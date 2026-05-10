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
from experiments.tagging.omniloader import (
    _LABEL_SHIFT,
    _PRETRAIN_SOURCES,
    _hepdataset_worker_init,
    load_data,
)


def _detect_data_shape(cfg, dataset_name):
    """Scan h5 shards to infer (num_classes, n_feat); n_feat is the min across sources."""
    if dataset_name == "pretrain":
        names_and_shifts = [(n, 0) for n in _PRETRAIN_SOURCES]
    else:
        names_and_shifts = [(dataset_name, _LABEL_SHIFT.get(dataset_name, 0))]

    max_label = -1
    n_feat = None
    for name, shift in names_and_shifts:
        train_path = os.path.join(cfg.data.data_dir, name, "train")
        if not os.path.isdir(train_path):
            raise ValueError(f"Cannot detect data shape: {train_path} does not exist")
        per_source_n_feat = None
        for split in ("train", "val", "test"):
            path = os.path.join(cfg.data.data_dir, name, split)
            if not os.path.isdir(path):
                LOGGER.warning(f"Skipping label/n_feat scan: {path} missing")
                continue
            h5_files = [f for f in os.listdir(path) if f.endswith((".h5", ".hdf5"))]
            if not h5_files:
                if split == "train":
                    raise ValueError(f"Cannot detect data shape: no h5 files in {path}")
                continue
            for fname in h5_files:
                with h5py.File(os.path.join(path, fname), "r") as f:
                    file_max_label = int(f["pid"][:].max()) - shift
                    max_label = max(max_label, file_max_label)
                    file_n_feat = f["data"].shape[-1]
                    assert per_source_n_feat in (None, file_n_feat), (
                        f"{name}: n_feat mismatch across splits "
                        f"({per_source_n_feat} vs {file_n_feat} in {fname})"
                    )
                    per_source_n_feat = file_n_feat
                    n_feat = file_n_feat if n_feat is None else min(n_feat, file_n_feat)
    return max_label + 1, n_feat


def _extract_cli_overrides(cfg, prefix):
    """Pull `prefix.*` overrides from the hydra CLI (strips `+`/`~` markers)."""
    if not HydraConfig.initialized():
        return OmegaConf.create()
    out = OmegaConf.create()
    for raw in HydraConfig.get().overrides.task:
        override = raw.lstrip("+~")
        if not override.startswith(prefix):
            continue
        key = override.split("=", 1)[0]
        val = OmegaConf.select(cfg, key)
        rel = key[len(prefix) :]
        OmegaConf.update(out, rel, val, merge=True)
    return out


class _OmniDataMixin:
    """Data-loading + batch-extraction mixin for OmniLearned h5 (rank-partitioned in load_data)."""

    DATASET_NAME: str | None = None
    LABEL_DTYPE: torch.dtype | None = None

    @property
    def _dataset_name(self):
        return self.DATASET_NAME if self.DATASET_NAME is not None else self.cfg.data.dataset_name

    def init_data(self):
        """Build per-split datasets; the loader's batching is rebuilt in `_init_dataloader`."""
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

    def _check_omnilearned_canonicalization(self):
        """Assert YAML pin (beam_y as intent marker) and override to None: data is already jet-centered."""
        assert self.cfg.data.canonicalize in (None, "beam_y"), (
            f"OmniLearned data is jet-centered by construction; "
            f"cfg.data.canonicalize must be 'beam_y' (intent marker, "
            f"overridden to None at runtime) or already None, got "
            f"{self.cfg.data.canonicalize}"
        )
        with open_dict(self.cfg):
            self.cfg.data.canonicalize = None

    def _init_dataloader(self):
        per_rank_train = self.cfg.training.batchsize // self.world_size
        per_rank_eval = self.cfg.evaluation.batchsize // self.world_size
        num_workers = int(self.cfg.data.num_workers)
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if num_workers > 0:
            loader_kwargs["worker_init_fn"] = _hepdataset_worker_init
            loader_kwargs["persistent_workers"] = True
            prefetch = self.cfg.data.get("prefetch_factor", None)
            if prefetch is not None:
                loader_kwargs["prefetch_factor"] = int(prefetch)

        # drop_last=True equalizes per-rank batch counts (avoids DDP all-reduce hang on train).
        self.train_loader = DataLoader(
            self.data_train,
            batch_size=per_rank_train,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
        self.test_loader = DataLoader(
            self.data_test,
            batch_size=per_rank_eval,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        self.val_loader = DataLoader(
            self.data_val,
            batch_size=per_rank_eval,
            shuffle=False,
            drop_last=False,
            **loader_kwargs,
        )
        LOGGER.info(
            f"Constructed dataloaders with "
            f"train_batches={len(self.train_loader)}, test_batches={len(self.test_loader)}, "
            f"val_batches={len(self.val_loader)}, "
            f"batch_size={self.cfg.training.batchsize} (training), "
            f"{self.cfg.evaluation.batchsize} (evaluation), "
            f"num_workers={num_workers}"
        )
        self._record_train_size()
        self.init_standardization()

    def _extract_batch(self, batch):
        """Convert an omniloader batch dict to (fourmomenta, scalars, label, weights).

        X column layout (h5 width=9 on the OmniLearned shards):
            col 0    deta_jet (continuous)
            col 1    dphi_jet (continuous)
            col 2    log(pT)  (continuous; ==0 marks padding)
            col 3    log(E)   (continuous) -- consumed here to set the
                              4-momentum energy and NOT forwarded to scalars
            col 4    particle PID class index 0..8 (continuous-embedded)
            cols 5-8 impact parameters (source-dependent; zero on atlas/h1)

        Cols 0-3 reconstruct 4-momenta in the jet-centered frame (using the
        on-disk log E so that the multivector embedding for LGATr / LGATrSlim
        is properly massive). Cols 4+ pass through as scalars.
        """
        n_feat = 4 + self.extra_scalars
        X_full = batch["X"][..., :n_feat]
        # Drop all-padding events: an all-False mask softmaxes to NaN inside the network.
        nonempty = (X_full[..., 2] != 0).any(dim=-1)
        if not bool(nonempty.all()):
            X_full = X_full[nonempty]
            y = batch["y"][nonempty]
        else:
            y = batch["y"]

        # Zero-pad trailing scalar columns when on-disk width < expected n_feat (e.g. finetune
        # on n_feat=4 top with a backbone pretrained on n_feat=9). Matching columns reuse the
        # pretrained linear_in weights; missing columns multiply zero. Upstream omnilearned
        # reinits linear_in via filter_partial_model on shape mismatch; we keep it intact.
        if X_full.shape[-1] < n_feat:
            if not getattr(self, "_warned_scalar_pad", False):
                LOGGER.warning(
                    f"Per-particle feature width {X_full.shape[-1]} < expected "
                    f"{n_feat} (4 kinematic + {self.extra_scalars} extra scalars); "
                    f"zero-padding the trailing {n_feat - X_full.shape[-1]} columns."
                )
                self._warned_scalar_pad = True
            X_full = torch.nn.functional.pad(X_full, (0, n_feat - X_full.shape[-1]))

        # Mask in fp32 so a small log(pT) doesn't round to 0 under bf16/fp16.
        mask = (X_full[..., 2] != 0).to(self.device)
        X = X_full.to(self.device, self.momentum_dtype)

        pt = torch.exp(X[..., 2])
        eta = X[..., 0]
        phi = X[..., 1]
        E = torch.exp(X[..., 3])
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        fourmomenta = torch.stack([E, px, py, pz], dim=-1) * mask.unsqueeze(-1)

        if self.extra_scalars > 0:
            scalars = X[..., 4:].to(self.dtype) * mask.unsqueeze(-1).to(self.dtype)
        else:
            scalars = torch.empty(*fourmomenta.shape[:-1], 0, device=self.device, dtype=self.dtype)

        ldtype = self.LABEL_DTYPE if self.LABEL_DTYPE is not None else self.dtype
        label = y.to(self.device, ldtype)
        weights = torch.ones_like(label, dtype=self.dtype)

        return fourmomenta, scalars, label, weights


class PretrainExperiment(_OmniDataMixin, TaggingExperiment):
    """Multi-class CE pretraining over the OmniLearned union; num_classes/extra_scalars auto-detected."""

    LABEL_DTYPE = torch.long

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = None
        self.extra_scalars = None

    def init_physics(self):
        self._check_omnilearned_canonicalization()
        if self.num_outputs is None or self.extra_scalars is None:
            n_classes, n_feat = _detect_data_shape(self.cfg, self._dataset_name)
            self.num_outputs = n_classes
            self.extra_scalars = max(n_feat - 4, 0)
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

    def _class_balanced_weights(self, label):
        """Per-batch 1/count class weighting (mirrors omnilearned/utils.py:get_loss)."""
        counts = torch.bincount(label, minlength=self.num_outputs).float() + 1e-6
        w = (1.0 / counts)[label]
        w = (w / w.mean()).to(self.dtype)
        return w

    def _batch_loss(self, batch):
        y_pred, label, tracker, _, _ = self._get_ypred_and_label(batch)
        w = self._class_balanced_weights(label)
        loss = torch.mean(w * self.loss(y_pred, label))
        return loss, tracker

    @torch.inference_mode()
    def _evaluate_single(self, loader, title, mode, step=None):
        """CE eval with the same per-batch class weighting as training."""
        assert mode in ["val", "eval"]
        if mode == "eval":
            LOGGER.info(f"### Starting to evaluate model on {title} dataset ###")

        self.model.eval()
        # fp32 accumulators: bf16/fp16 lose precision over 100s-1000s of summed batches.
        loss_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        n_batches = torch.zeros((), device=self.device, dtype=torch.float32)
        for batch in loader:
            y_pred, label, _, _, _ = self._get_ypred_and_label(batch)
            w = self._class_balanced_weights(label)
            ce = torch.nn.functional.cross_entropy(y_pred, label, reduction="none")
            loss_sum += (w * ce).mean().float()
            n_batches += 1

        loss_sum = gather_concat(loss_sum.unsqueeze(0)).sum().item()
        n_batches = gather_concat(n_batches.unsqueeze(0)).sum().item()

        metrics = {"loss": loss_sum / max(n_batches, 1.0)}
        if mode == "eval":
            LOGGER.info(f"CELoss on {title} dataset: {metrics['loss']:.4f}")

        if self.cfg.use_mlflow:
            name = f"{mode}.{title}" if mode == "eval" else "val"
            log_mlflow(f"{name}.loss", metrics["loss"], step=step)
        return metrics


class Finetune2Experiment(_OmniDataMixin, BinaryTaggingExperiment):
    """Binary BCE finetune of a PretrainExperiment backbone on the h5 `top` data."""

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

        # Match backbone's per-particle feature count so linear_in.weight loads cleanly.
        self.extra_scalars = int(self.warmstart_cfg.data.extra_scalars)

        with open_dict(self.cfg):
            model_cli = _extract_cli_overrides(self.cfg, "model.")
            self.cfg.model = OmegaConf.merge(self.warmstart_cfg.model, model_cli)

            # Validate _target_ AFTER the CLI merge so a CLI override can't bypass it.
            if self.cfg.model._target_ not in [
                "experiments.tagging.wrappers.TransformerWrapper",
                "experiments.tagging.wrappers.ParTWrapper",
                "experiments.tagging.wrappers.LGATrWrapper",
                "experiments.tagging.wrappers.LGATrSlimWrapper",
            ]:
                raise NotImplementedError(
                    f"Finetune2Experiment does not support model {self.cfg.model._target_}"
                )

            # Carry over backbone-shaping data fields unless CLI-overridden.
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
                if key in self.warmstart_cfg.data and self.warmstart_cfg.data[key] is not None:
                    self.cfg.data[key] = self.warmstart_cfg.data[key]

    def init_physics(self):
        self._check_omnilearned_canonicalization()
        super().init_physics()

    def init_model(self):
        # Match pretrained head shape so backbone weights load cleanly.
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

            target = self.cfg.model._target_
            if target == "experiments.tagging.wrappers.TransformerWrapper":
                self.model.net.linear_out = torch.nn.Linear(
                    self.model.net.hidden_channels, self.num_outputs
                )
            elif target == "experiments.tagging.wrappers.ParTWrapper":
                # Reset the entire fc head: pretrained head is task-specific.
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
        """Group params: backbone -> lr_backbone, input embed + head -> lr_head (mirrors upstream `new_layer`)."""
        assert param_groups is None, "Finetune2Experiment constructs param_groups manually"

        target = self.cfg.model._target_
        if target == "experiments.tagging.wrappers.TransformerWrapper":
            params_backbone_framesnet = list(self._model.framesnet.parameters())
            params_backbone_main = list(self._model.net.blocks.parameters())
            params_embed = list(self._model.net.linear_in.parameters())
            params_head = list(self._model.net.linear_out.parameters())

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
                    "params": params_embed,
                    "lr": self.cfg.finetune.lr_head,
                    "weight_decay": self.cfg.training.weight_decay,
                },
                {
                    "params": params_head,
                    "lr": self.cfg.finetune.lr_head,
                    "weight_decay": self.cfg.training.weight_decay,
                },
            ]
        elif target == "experiments.tagging.wrappers.ParTWrapper":
            # 1D-vs-rest decay split applied separately to backbone, embed (`embed.*`, `pair_embed.*`), head (`fc.*`).
            no_decay_names = (
                self._model.net.no_weight_decay()
                if hasattr(self._model.net, "no_weight_decay")
                else set()
            )
            decay, no_decay = {}, {}
            embed_decay, embed_nodecay = {}, {}
            head_decay, head_nodecay = {}, {}
            for name, param in self._model.net.named_parameters():
                if not param.requires_grad:
                    continue
                is_no_decay = (
                    len(param.shape) == 1 or name.endswith(".bias") or name in no_decay_names
                )
                is_head = name.startswith("fc.")
                is_embed = name.startswith(("embed.", "pair_embed."))
                if is_head:
                    (head_nodecay if is_no_decay else head_decay)[name] = param
                elif is_embed:
                    (embed_nodecay if is_no_decay else embed_decay)[name] = param
                else:
                    (no_decay if is_no_decay else decay)[name] = param
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
                    "params": list(embed_nodecay.values()),
                    "weight_decay": 0.0,
                    "lr": self.cfg.finetune.lr_head,
                },
                {
                    "params": list(embed_decay.values()),
                    "weight_decay": self.cfg.training.weight_decay,
                    "lr": self.cfg.finetune.lr_head,
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
            params_backbone = list(self._model.net.blocks.parameters())
            params_embed = list(self._model.net.linear_in.parameters())
            params_head = list(self._model.net.linear_out.parameters())

            param_groups = [
                {"params": params_backbone, "lr": self.cfg.finetune.lr_backbone},
                {"params": params_embed, "lr": self.cfg.finetune.lr_head},
                {"params": params_head, "lr": self.cfg.finetune.lr_head},
            ]
        else:
            raise NotImplementedError

        super()._init_optimizer(param_groups=param_groups)
