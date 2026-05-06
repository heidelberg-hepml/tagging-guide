import os

import h5py
import torch
from hydra.core.hydra_config import HydraConfig
from lgatr.layers.linear import EquiLinear
from lgatr.nets.lgatr_slim import Linear as LorentzLinear
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from experiments.distributed import gather_concat
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow
from experiments.tagging.experiment import BinaryTaggingExperiment, TaggingExperiment
from experiments.tagging.omniloader import load_data

# omniloader subtracts these per-source label shifts to put each source's pid
# into a local 0..N-1 range (mirrors omniloader.load_data:353-359). For the
# union "pretrain" dataset_name, no shift is applied (labels are already global).
_LABEL_SHIFT = {
    "jetclass": 2,
    "jetclass2": 12,
    "aspen": 200,
    "cms_qcd": 201,
    "cms_bsm": 202,
}
_PRETRAIN_UNION = ["atlas", "aspen", "jetclass", "jetclass2", "h1", "cms_qcd", "cms_bsm"]


def _detect_data_shape(cfg, dataset_name):
    """Read training h5 files to determine (num_classes, num_features F).

    For the "pretrain" union we take min(F) across sources so per-particle
    feature slicing is consistent across the union. num_classes is
    max(label) + 1 after applying the per-source label_shift.
    """
    if dataset_name == "pretrain":
        names_and_shifts = [(n, 0) for n in _PRETRAIN_UNION]
    else:
        names_and_shifts = [(dataset_name, _LABEL_SHIFT.get(dataset_name, 0))]

    max_label = -1
    min_F = None
    for name, shift in names_and_shifts:
        path = os.path.join(cfg.data.data_dir, name, "train")
        if not os.path.isdir(path):
            raise ValueError(f"Cannot detect data shape: {path} does not exist")
        h5_files = [f for f in os.listdir(path) if f.endswith((".h5", ".hdf5"))]
        if not h5_files:
            raise ValueError(f"Cannot detect data shape: no h5 files in {path}")
        for fname in h5_files:
            with h5py.File(os.path.join(path, fname), "r") as f:
                m = int(f["pid"][:].max()) - shift
                max_label = max(max_label, m)
                F = f["data"].shape[-1]
                min_F = F if min_F is None else min(min_F, F)
    return max_label + 1, min_F


def _extract_cli_overrides(cfg, prefix):
    """Pull `prefix.*` overrides from the active hydra command line."""
    if not HydraConfig.initialized():
        return OmegaConf.create()
    out = OmegaConf.create()
    for s in HydraConfig.get().overrides.task:
        s_ = s.lstrip("+~")
        if not s_.startswith(prefix):
            continue
        key = s_.split("=", 1)[0]
        val = OmegaConf.select(cfg, key)
        rel = key[len(prefix) :]
        OmegaConf.update(out, rel, val, merge=True)
    return out


class _OmniBackbone:
    """Mixin: data loading + extraction for OmniLearned-format h5 datasets.

    Subclass knobs:
      * ``DATASET_NAME`` (optional): override ``cfg.data.dataset_name``.
      * ``LABEL_DTYPE`` (optional): torch dtype for the label tensor; defaults
        to ``self.dtype`` (float, suitable for BCE). PretrainExperiment sets
        it to ``torch.long`` for cross-entropy.

    DDP follows omnilearned (omnilearned/train.py:418-455 +
    omnilearned/dataloader.py:300-385): rank/size is passed to ``load_data``
    so each rank holds only its strided slice of file_indices, and we use a
    plain DataLoader without a DistributedSampler (which would shard twice).
    """

    DATASET_NAME: str | None = None
    LABEL_DTYPE: torch.dtype | None = None

    @property
    def _dataset_name(self):
        return self.DATASET_NAME if self.DATASET_NAME is not None else self.cfg.data.dataset_name

    def init_data(self):
        for split in ("train", "test", "val"):
            loader = load_data(
                dataset_name=self._dataset_name,
                path=self.cfg.data.data_dir,
                dataset_type=split,
                num_workers=0,
                batch=1,
                rank=self.rank,
                size=self.world_size,
                shuffle=(split == "train"),
            )
            setattr(self, f"data_{split}", loader.dataset)
        LOGGER.info(
            f"Loaded omniloader datasets ({self._dataset_name}): "
            f"train={len(self.data_train)}, test={len(self.data_test)}, val={len(self.data_val)}"
        )

    def _init_dataloader(self):
        per_rank_train = self.cfg.training.batchsize // self.world_size
        per_rank_eval = self.cfg.evaluation.batchsize // self.world_size
        self.train_loader = DataLoader(
            self.data_train, batch_size=per_rank_train, shuffle=True, drop_last=False
        )
        self.test_loader = DataLoader(
            self.data_test, batch_size=per_rank_eval, shuffle=False, drop_last=False
        )
        self.val_loader = DataLoader(
            self.data_val, batch_size=per_rank_eval, shuffle=False, drop_last=False
        )
        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, "
            f"val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), "
            f"{self.cfg.evaluation.batchsize} (evaluation)"
        )
        self._record_train_size()
        self.init_standardization()

    def _extract_batch(self, batch):
        """Convert an omniloader batch dict to (fourmomenta, scalars, label, weights).

        OmniLearned X format: X[..., 0]=deta_jet, X[..., 1]=dphi_jet,
        X[..., 2]=log(pT). The 4-momentum is reconstructed under the massless
        approximation (consistent with how OmniLearned itself derives invariant
        mass; see omnilearned/layers.py:65-76).

        X[..., 3 : 3 + extra_scalars] are passed through as continuous scalar
        features. This mirrors omnilearned's add_info pathway
        (omnilearned/network.py:462-473), which feeds them through an MLP
        rather than via nn.Embedding. The pid column (when present in the
        underlying data) is omitted unless extra_scalars covers it; pid would
        ideally need an embedding layer (omnilearned/network.py:475-480),
        which our wrappers do not provide.
        """
        n_feat = 3 + self.extra_scalars
        X = batch["X"][..., :n_feat].to(self.device, self.momentum_dtype)
        # Robust mask: padded slots are exact zeros across all columns; a real
        # particle can have any single column at zero (e.g. pT=1 GeV makes
        # log(pT)=0) but cannot have all of (deta, dphi, log(pT)) simultaneously
        # at exactly zero except by astronomical coincidence.
        mask = (X != 0).any(dim=-1)

        pt = torch.exp(X[..., 2])
        eta = X[..., 0]
        phi = X[..., 1]
        E = pt * torch.cosh(eta)
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        fourmomenta = torch.stack([E, px, py, pz], dim=-1) * mask.unsqueeze(-1)

        if self.extra_scalars > 0:
            scalars = X[..., 3:].to(self.dtype) * mask.unsqueeze(-1).to(self.dtype)
        else:
            scalars = torch.empty(*fourmomenta.shape[:-1], 0, device=self.device, dtype=self.dtype)

        ldtype = self.LABEL_DTYPE if self.LABEL_DTYPE is not None else self.dtype
        label = batch["y"].to(self.device, ldtype)
        weights = torch.ones_like(label, dtype=self.dtype)
        return fourmomenta, scalars, label, weights


class PretrainExperiment(_OmniBackbone, TaggingExperiment):
    """Multi-class supervised pretraining over OmniLearned-format h5 data.

    Loss = softmax cross-entropy (mirrors omnilearned/train.py:569). The
    diffusion / perturbed-classification / CLIP auxiliaries from OmniLearned
    are intentionally dropped.

    num_classes and extra_scalars are inferred from the training data.
    """

    LABEL_DTYPE = torch.long

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = None  # populated in init_physics
        self.extra_scalars = None  # populated in init_physics

    def init_physics(self):
        if self.num_outputs is None or self.extra_scalars is None:
            n_classes, n_feat = _detect_data_shape(self.cfg, self._dataset_name)
            self.num_outputs = n_classes
            self.extra_scalars = max(n_feat - 3, 0)
            LOGGER.info(
                f"Auto-detected num_classes={self.num_outputs}, extra_scalars={self.extra_scalars}"
            )
        # Persist for downstream finetune2 (reads from the saved warmstart cfg).
        with open_dict(self.cfg):
            self.cfg.data.extra_scalars = self.extra_scalars
            self.cfg.data.num_classes = self.num_outputs
        super().init_physics()

    def _init_loss(self):
        self.loss = torch.nn.CrossEntropyLoss(reduction="none")

    @torch.inference_mode()
    def _evaluate_single(self, loader, title, mode, step=None):
        """Report only the cross-entropy loss.

        Mirrors omnilearned/train.py:val_step (138-196), which only accumulates
        loss values; it does not compute accuracy or AUC for pretraining.
        Per-class metrics are not particularly informative on the union dataset.
        """
        assert mode in ["val", "eval"]
        if mode == "eval":
            LOGGER.info(f"### Starting to evaluate model on {title} dataset ###")

        self.model.eval()
        loss_sum = torch.zeros((), device=self.device, dtype=self.dtype)
        n_total = torch.zeros((), device=self.device, dtype=self.dtype)
        for batch in loader:
            y_pred, label, _, _, weights = self._get_ypred_and_label(batch)
            ce = torch.nn.functional.cross_entropy(y_pred, label, reduction="none")
            loss_sum += (weights * ce).sum()
            n_total += weights.sum()

        loss_sum = gather_concat(loss_sum.unsqueeze(0)).sum().item()
        n_total = gather_concat(n_total.unsqueeze(0)).sum().item()

        metrics = {"loss": loss_sum / max(n_total, 1.0)}
        if mode == "eval":
            LOGGER.info(f"CELoss on {title} dataset: {metrics['loss']:.4f}")

        if self.cfg.use_mlflow:
            name = f"{mode}.{title}" if mode == "eval" else "val"
            log_mlflow(f"{name}.loss", metrics["loss"], step=step)
        return metrics


class Finetune2Experiment(_OmniBackbone, BinaryTaggingExperiment):
    """Binary BCE finetune of a PretrainExperiment backbone on top tagging.

    Distinct from TopTaggingFineTuneExperiment, which uses the npz top dataset;
    this one consumes the same h5 data layout that the pretrain backbone saw,
    so a single pretrained backbone can be reused across the OmniLearned
    ecosystem.

    Hard-coded to ``dataset_name='top'`` (binary signal/background); set
    ``cfg.finetune.backbone_path`` to a pretrain run directory.
    """

    DATASET_NAME = "top"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1

        warmstart_path = os.path.join(
            self.cfg.finetune.backbone_path, self.cfg.finetune.backbone_cfg
        )
        self.warmstart_cfg = OmegaConf.load(warmstart_path)
        assert self.warmstart_cfg.exp_type == "pretrain", (
            f"Finetune2Experiment expects warmstart from a 'pretrain' run, "
            f"got exp_type={self.warmstart_cfg.exp_type}"
        )

        if self.warmstart_cfg.model._target_ not in [
            "experiments.tagging.wrappers.TransformerWrapper",
            "experiments.tagging.wrappers.ParTWrapper",
            "experiments.tagging.wrappers.LGATrWrapper",
            "experiments.tagging.wrappers.LGATrSlimWrapper",
        ]:
            raise NotImplementedError(
                f"Finetune2Experiment does not support model {self.warmstart_cfg.model._target_}"
            )

        # Match the pretrained backbone's per-particle feature count so
        # linear_in.weight loads cleanly. The pretrain run persisted this in
        # cfg.data.extra_scalars during its own init_physics.
        self.extra_scalars = int(self.warmstart_cfg.data.extra_scalars)

        with open_dict(self.cfg):
            model_cli = _extract_cli_overrides(self.cfg, "model.")
            self.cfg.model = OmegaConf.merge(self.warmstart_cfg.model, model_cli)

            # Carry over backbone-shaping data fields. Honor user CLI overrides
            # (mirrors the model.* pattern above) and skip keys absent from the
            # warmstart cfg so older pretrain runs don't crash here.
            data_cli = _extract_cli_overrides(self.cfg, "data.")
            for key in (
                "tagging_features",
                "canonicalize",
                "beam_reference",
                "two_beams",
                "add_time_reference",
                "spurion_scale",
                "momentum_float64",
            ):
                if key in data_cli:
                    continue
                warm_val = OmegaConf.select(self.warmstart_cfg.data, key)
                if warm_val is not None:
                    self.cfg.data[key] = warm_val

    def init_model(self):
        # Match the pretrained head shape so backbone weights load cleanly.
        self.cfg.model.out_channels = self.warmstart_cfg.model.out_channels

        self._create_model()

        if not self.warm_start:
            model_path = os.path.join(
                self.warmstart_cfg.run_dir,
                "models",
                f"model_run{self.warmstart_cfg.run_idx}.pt",
            )
            try:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=False)["model"]
            except FileNotFoundError as err:
                raise ValueError(f"Cannot load model from {model_path}") from err
            LOGGER.info(f"Loading pretrained model from {model_path}")
            self.model.load_state_dict(state_dict)

            target = self.warmstart_cfg.model._target_
            if target == "experiments.tagging.wrappers.TransformerWrapper":
                self.model.net.linear_out = torch.nn.Linear(
                    self.model.net.hidden_channels, self.num_outputs
                )
            elif target == "experiments.tagging.wrappers.ParTWrapper":
                self.model.net.fc[-1] = torch.nn.Linear(self.model.net.embed_dim, self.num_outputs)
                for module in self.model.net.fc.modules():
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()
            elif target == "experiments.tagging.wrappers.LGATrWrapper":
                self.model.net.linear_out = EquiLinear(
                    in_mv_channels=self.cfg.model.net.hidden_mv_channels,
                    out_mv_channels=self.num_outputs,
                    in_s_channels=self.cfg.model.net.hidden_s_channels,
                    out_s_channels=self.cfg.model.net.out_s_channels,
                )
            elif target == "experiments.tagging.wrappers.LGATrSlimWrapper":
                self.model.net.linear_out = LorentzLinear(
                    in_v_channels=self.cfg.model.net.hidden_v_channels,
                    out_v_channels=self.cfg.model.net.out_v_channels,
                    in_s_channels=self.cfg.model.net.hidden_s_channels,
                    out_s_channels=self.num_outputs,
                )
            else:
                raise NotImplementedError

        self._finalize_model()

    def _init_optimizer(self, param_groups=None):
        assert param_groups is None, "Finetune2Experiment constructs param_groups manually"

        target = self.warmstart_cfg.model._target_
        if target == "experiments.tagging.wrappers.TransformerWrapper":
            params_backbone_framesnet = list(self._model.framesnet.parameters())
            params_backbone_main = list(self._model.net.linear_in.parameters()) + list(
                self._model.net.blocks.parameters()
            )
            params_head = self._model.net.linear_out.parameters()

            param_groups = [
                {
                    "params": params_backbone_framesnet,
                    "lr": self.cfg.finetune.lr_backbone * self.cfg.training.lr_factor_framesnet,
                    "weight_decay": self.cfg.training.weight_decay_framesnet,
                },
                {
                    "params": params_backbone_main,
                    "lr": self.cfg.finetune.lr_backbone,
                    "weight_decay": self.cfg.training.weight_decay,
                },
                {
                    "params": params_head,
                    "lr": self.cfg.finetune.lr_head,
                    "weight_decay": self.cfg.training.weight_decay,
                },
            ]
        elif target == "experiments.tagging.wrappers.ParTWrapper":
            decay, no_decay, head_decay, head_nodecay = {}, {}, {}, {}
            for name, param in self._model.net.named_parameters():
                if not param.requires_grad:
                    continue
                if (
                    len(param.shape) == 1
                    or name.endswith(".bias")
                    or (hasattr(self._model.net, "no_weight_decay") and name in {"cls_token"})
                ):
                    if name.startswith("fc."):
                        head_nodecay[name] = param
                    else:
                        no_decay[name] = param
                else:
                    if name.startswith("fc."):
                        head_decay[name] = param
                    else:
                        decay[name] = param
            param_groups = [
                {
                    "params": list(no_decay.values()),
                    "weight_decay": 0.0,
                    "lr": self.cfg.finetune.lr_backbone,
                },
                {
                    "params": list(decay.values()),
                    "weight_decay": self.cfg.training.weight_decay,
                    "lr": self.cfg.finetune.lr_backbone,
                },
                {
                    "params": self._model.framesnet.parameters(),
                    "weight_decay": self.cfg.training.weight_decay_framesnet,
                    "lr": self.cfg.finetune.lr_backbone * self.cfg.training.lr_factor_framesnet,
                },
                {
                    "params": list(head_nodecay.values()),
                    "weight_decay": 0.0,
                    "lr": self.cfg.finetune.lr_head,
                },
                {
                    "params": list(head_decay.values()),
                    "weight_decay": self.cfg.training.weight_decay,
                    "lr": self.cfg.finetune.lr_head,
                },
            ]
        elif target in [
            "experiments.tagging.wrappers.LGATrWrapper",
            "experiments.tagging.wrappers.LGATrSlimWrapper",
        ]:
            params_backbone = list(self._model.net.linear_in.parameters()) + list(
                self._model.net.blocks.parameters()
            )
            params_head = self._model.net.linear_out.parameters()

            param_groups = [
                {"params": params_backbone, "lr": self.cfg.finetune.lr_backbone},
                {"params": params_head, "lr": self.cfg.finetune.lr_head},
            ]
        else:
            raise NotImplementedError

        super()._init_optimizer(param_groups=param_groups)
