import copy
import os

import torch
from lgatr.layers.linear import EquiLinear
from lgatr.nets.lgatr_slim import Linear as LorentzLinear
from omegaconf import OmegaConf, open_dict

from experiments.logger import LOGGER


class _FinetuneMixin:
    """Shared finetune logic: load a pretrained backbone, swap the output head, build layer-wise-LR
    param groups. Mix in front of a fourmomenta-only dataset experiment, e.g.
    ``class TopTaggingFineTuneExperiment(_FinetuneMixin, TopTaggingExperiment)``.
    """

    ALLOWED_WARMSTART_EXP_TYPES = {"jetclass", "toptagxl", "pretrain"}
    BACKBONE_DATA_FIELDS = (
        "tagging_features",
        "canonicalize",
        "beam_reference",
        "two_beams",
        "add_time_reference",
        "spurion_scale",
        "momentum_float64",
        "units",
        "mass_reg",
        "canonicalize_spurions",
        "max_particles",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert self.extra_scalars == 0, (
            f"{type(self).__name__} supports only fourmomenta-only targets "
            f"(extra_scalars==0), got extra_scalars={self.extra_scalars}"
        )

        warmstart_path = os.path.join(
            self.cfg.finetune.backbone_path, self.cfg.finetune.backbone_cfg
        )
        self.warmstart_cfg = OmegaConf.load(warmstart_path)
        assert self.warmstart_cfg.exp_type in self.ALLOWED_WARMSTART_EXP_TYPES, (
            f"{type(self).__name__} expects warmstart from "
            f"{sorted(self.ALLOWED_WARMSTART_EXP_TYPES)}, got "
            f"exp_type={self.warmstart_cfg.exp_type}"
        )
        wd = self.warmstart_cfg.data
        n_extra = int(wd.get("extra_scalars", 0) or 0)
        assert n_extra == 0, (
            f"finetune supplies no per-particle scalars, but backbone expects extra_scalars={n_extra}"
        )

        supported_wrappers = (
            "experiments.tagging.wrappers.TransformerWrapper",
            "experiments.tagging.wrappers.ParTWrapper",
            "experiments.tagging.wrappers.LGATrWrapper",
            "experiments.tagging.wrappers.LGATrSlimWrapper",
        )
        with open_dict(self.cfg):
            self.cfg.model = copy.deepcopy(self.warmstart_cfg.model)

            if self.cfg.model._target_ not in supported_wrappers:
                raise NotImplementedError(
                    f"{type(self).__name__} does not support model {self.cfg.model._target_}"
                )

            for key in self.BACKBONE_DATA_FIELDS:
                ws_val = self.warmstart_cfg.data.get(key, None)
                if ws_val is None:
                    continue
                cur_val = self.cfg.data.get(key, None)
                if cur_val != ws_val:
                    LOGGER.warning(
                        f"overriding cfg.data.{key}={cur_val!r} with warmstart value {ws_val!r}"
                    )
                self.cfg.data[key] = ws_val

    def init_model(self):
        if not self.warm_start:
            self.cfg.model.out_channels = self.warmstart_cfg.model.out_channels
        self._create_model()

        if not self.warm_start:
            # load from backbone_path (where the config came from), not the saved run_dir
            model_path = os.path.join(
                self.cfg.finetune.backbone_path,
                "models",
                f"model_run{self.warmstart_cfg.run_idx}.pt",
            )
            try:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=False)["model"]
            except FileNotFoundError as err:
                raise ValueError(f"Cannot load model from {model_path}") from err
            LOGGER.info(f"Loading pretrained model from {model_path}")
            self.model.load_state_dict(state_dict)

            # output-layer surgery (must happen before _finalize_model wraps with DDP)
            target = self.cfg.model._target_
            if target == "experiments.tagging.wrappers.TransformerWrapper":
                self.model.net.linear_out = torch.nn.Linear(
                    self.model.net.hidden_channels, self.num_outputs
                )
            elif target == "experiments.tagging.wrappers.ParTWrapper":
                # Reset the entire fc head: pretrained head is task-specific; only the backbone transfers.
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
        assert param_groups is None, "FineTuneExperiment constructs param_groups manually"

        target = self.cfg.model._target_
        if target == "experiments.tagging.wrappers.ParTWrapper":
            head_prefix = "fc."
        elif target in (
            "experiments.tagging.wrappers.TransformerWrapper",
            "experiments.tagging.wrappers.LGATrWrapper",
            "experiments.tagging.wrappers.LGATrSlimWrapper",
        ):
            head_prefix = "linear_out."
        else:
            raise NotImplementedError(f"{type(self).__name__} does not support model {target}")

        def _is_decay(param):
            return param.squeeze().ndim > 1

        backbone_decay, backbone_nodecay = [], []
        head_decay, head_nodecay = [], []
        for name, param in self._model.net.named_parameters():
            if not param.requires_grad:
                continue
            decay = _is_decay(param)
            if name.startswith(head_prefix):
                (head_decay if decay else head_nodecay).append(param)
            else:
                (backbone_decay if decay else backbone_nodecay).append(param)

        framesnet_decay, framesnet_nodecay = [], []
        for _, param in self._model.framesnet.named_parameters():
            if not param.requires_grad:
                continue
            (framesnet_decay if _is_decay(param) else framesnet_nodecay).append(param)

        wd = self.cfg.training.weight_decay
        wd_fn = self.cfg.training.weight_decay_framesnet
        lr_bb = self.cfg.finetune.lr_backbone
        lr_head = self.cfg.finetune.lr_head
        lr_fn = lr_bb * self.cfg.training.lr_factor_framesnet

        param_groups = [
            {"params": backbone_decay, "lr": lr_bb, "weight_decay": wd},
            {"params": backbone_nodecay, "lr": lr_bb, "weight_decay": 0.0},
            {"params": framesnet_decay, "lr": lr_fn, "weight_decay": wd_fn},
            {"params": framesnet_nodecay, "lr": lr_fn, "weight_decay": 0.0},
            {"params": head_decay, "lr": lr_head, "weight_decay": wd},
            {"params": head_nodecay, "lr": lr_head, "weight_decay": 0.0},
        ]

        super()._init_optimizer(param_groups=param_groups)
