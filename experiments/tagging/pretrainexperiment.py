import os

import h5py
import torch
from omegaconf import open_dict
from torch.utils.data import DataLoader

from experiments.distributed import gather_concat
from experiments.logger import LOGGER
from experiments.mlflow import log_mlflow
from experiments.tagging.experiment import BinaryTaggingExperiment, TaggingExperiment
from experiments.tagging.omniloader import (
    _LABEL_SHIFT,
    _PRETRAIN_LABEL_OFFSET,
    _PRETRAIN_SOURCES,
    load_data,
)


def _detect_data_shape(cfg, dataset_name):
    """Scan h5 shards to infer (num_classes, n_feat); n_feat is the min across sources."""
    if dataset_name == "pretrain":
        sources = [(n, 0, _PRETRAIN_LABEL_OFFSET.get(n, 0)) for n in _PRETRAIN_SOURCES]
    else:
        sources = [(dataset_name, _LABEL_SHIFT.get(dataset_name, 0), 0)]

    max_label = -1
    n_feat = None
    for name, shift, offset in sources:
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
                    file_max_label = int(f["pid"][:].max()) - shift + offset
                    max_label = max(max_label, file_max_label)
                    file_n_feat = f["data"].shape[-1]
                    assert per_source_n_feat in (None, file_n_feat), (
                        f"{name}: n_feat mismatch across splits "
                        f"({per_source_n_feat} vs {file_n_feat} in {fname})"
                    )
                    per_source_n_feat = file_n_feat
                    n_feat = file_n_feat if n_feat is None else min(n_feat, file_n_feat)
    return max_label + 1, n_feat


class _OmniDataMixin:
    """Data-loading + batch-extraction mixin for OmniLearned h5 (rank-partitioned in load_data)."""

    DATASET_NAME: str | None = None
    LABEL_DTYPE: torch.dtype | None = None

    @property
    def _dataset_name(self):
        return self.DATASET_NAME if self.DATASET_NAME is not None else self.cfg.data.dataset_name

    def init_data(self):
        """Build per-split datasets; the loader's batching is rebuilt in `_init_dataloader`."""
        # `*_frac` only applies to pretrain; single-source experiments always use the full split.
        if self._dataset_name == "pretrain":
            fractions = {
                "train": float(self.cfg.data.train_frac),
                "val": float(self.cfg.data.val_frac),
                "test": float(self.cfg.data.test_frac),
            }
            for split, f in fractions.items():
                assert 0.0 < f <= 1.0, f"data.{split}_frac must be in (0, 1], got {f}"
        else:
            fractions = {"train": 1.0, "val": 1.0, "test": 1.0}

        seed = self.cfg.seed if self.cfg.seed is not None else 0
        slab_events = int(self.cfg.data.slab_events)
        buffer_bytes = int(self.cfg.data.buffer_mb) * 1024**2
        for split in ("train", "test", "val"):
            # shuffle=True everywhere: eval metrics are order-independent, and rank
            # coverage is exact via slab striping rather than shuffling.
            dataset = load_data(
                dataset_name=self._dataset_name,
                path=self.cfg.data.data_dir,
                dataset_type=split,
                rank=self.rank,
                size=self.world_size,
                shuffle=True,
                fraction=fractions[split],
                seed=seed,
                slab_events=slab_events,
                buffer_bytes=buffer_bytes,
            )
            setattr(self, f"data_{split}", dataset)
        LOGGER.info(
            f"Loaded omniloader datasets ({self._dataset_name}): "
            f"train={len(self.data_train)}, test={len(self.data_test)}, val={len(self.data_val)}"
        )

    def _check_omnilearned_canonicalization(self):
        """Assert YAML pin (beam_eta as intent marker) and override to None: _extract_batch
        already produces beam_eta-equivalent 4-vectors (massless, jet-centered)."""
        assert self.cfg.data.canonicalize in (None, "beam_eta"), (
            f"OmniLearned data is jet-centered by construction; "
            f"cfg.data.canonicalize must be 'beam_eta' (intent marker, "
            f"overridden to None at runtime) or already None, got "
            f"{self.cfg.data.canonicalize}"
        )
        with open_dict(self.cfg):
            self.cfg.data.canonicalize = None

    def _save_config(self, *args, **kwargs):
        """Dump canonicalize=beam_eta (the intent) instead of the runtime None, so the downstream
        finetune carry-over picks up the frame."""
        with open_dict(self.cfg):
            runtime = self.cfg.data.canonicalize
            self.cfg.data.canonicalize = "beam_eta"
        try:
            super()._save_config(*args, **kwargs)
        finally:
            with open_dict(self.cfg):
                self.cfg.data.canonicalize = runtime

    def _init_dataloader(self):
        per_rank_train = self.cfg.training.batchsize // self.world_size
        per_rank_eval = self.cfg.evaluation.batchsize // self.world_size
        num_workers = int(self.cfg.data.num_workers)
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if num_workers > 0:
            # persistent_workers=False so each epoch re-forks workers that inherit the
            # dataset's updated epoch (set_epoch), giving a fresh per-epoch shuffle.
            loader_kwargs["persistent_workers"] = False
            prefetch = self.cfg.data.get("prefetch_factor", None)
            if prefetch is not None:
                loader_kwargs["prefetch_factor"] = int(prefetch)

        # shuffle lives in the dataset (DataLoader shuffle is forbidden for IterableDataset).
        self.train_loader = DataLoader(
            self.data_train,
            batch_size=per_rank_train,
            drop_last=True,
            **loader_kwargs,
        )
        self.test_loader = DataLoader(
            self.data_test,
            batch_size=per_rank_eval,
            drop_last=False,
            **loader_kwargs,
        )
        self.val_loader = DataLoader(
            self.data_val,
            batch_size=per_rank_eval,
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
            col 3    log(E_lab) -- stored but DROPPED: lab-frame E paired with jet-
                              centered (deta, dphi) is not a Lorentz vector.
            col 4    particle PID class index 0..8 (continuous-embedded)
            cols 5-8 impact parameters (source-dependent; zero on atlas/h1)

        Cols 0-2 reconstruct massless 4-momenta in the jet-centered frame
        (E = sqrt(pT² + pz²)); this matches `canonicalize=beam_eta` on lab
        data in the m->0 limit. Cols 4+ pass through as scalars.
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
        px = pt * torch.cos(phi)
        py = pt * torch.sin(phi)
        pz = pt * torch.sinh(eta)
        E = torch.sqrt(pt * pt + pz * pz)
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
            self.extra_scalars = max(n_feat - 4, 0) if self.cfg.data.use_scalars else 0
            LOGGER.info(
                f"Auto-detected num_classes={self.num_outputs}, "
                f"extra_scalars={self.extra_scalars} (use_scalars={self.cfg.data.use_scalars})"
            )
        # persist for downstream finetune runs that read extra_scalars/num_classes from the warmstart cfg
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


class TopOmniExperiment(_OmniDataMixin, BinaryTaggingExperiment):
    """Binary BCE training from scratch on the h5 `top` data."""

    DATASET_NAME = "top"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_outputs = 1
        self.extra_scalars = None

    def init_physics(self):
        self._check_omnilearned_canonicalization()
        if self.extra_scalars is None:
            _, n_feat = _detect_data_shape(self.cfg, self._dataset_name)
            self.extra_scalars = max(n_feat - 4, 0) if self.cfg.data.use_scalars else 0
            LOGGER.info(
                f"Auto-detected extra_scalars={self.extra_scalars} "
                f"(use_scalars={self.cfg.data.use_scalars})"
            )
        super().init_physics()
