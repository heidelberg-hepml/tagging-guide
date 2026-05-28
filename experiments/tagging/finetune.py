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

        embed_lr_group = self.cfg.finetune.get("embed_lr_group", "head")
        assert embed_lr_group in ("head", "backbone"), (
            f"finetune.embed_lr_group must be 'head' or 'backbone', got {embed_lr_group}"
        )

        target = self.cfg.model._target_
        if target == "experiments.tagging.wrappers.TransformerWrapper":
            params_backbone_framesnet = list(self._model.framesnet.parameters())
            params_backbone_main = list(self._model.net.blocks.parameters())
            params_embed = list(self._model.net.linear_in.parameters())
            params_head = list(self._model.net.linear_out.parameters())
            if embed_lr_group == "backbone":
                params_backbone_main += params_embed
                params_embed = []

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
            if embed_lr_group == "backbone":
                decay.update(embed_decay)
                no_decay.update(embed_nodecay)
                embed_decay, embed_nodecay = {}, {}
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
        elif target in (
            "experiments.tagging.wrappers.LGATrWrapper",
            "experiments.tagging.wrappers.LGATrSlimWrapper",
        ):
            params_backbone = list(self._model.net.blocks.parameters())
            params_embed = list(self._model.net.linear_in.parameters())
            params_head = list(self._model.net.linear_out.parameters())
            if embed_lr_group == "backbone":
                params_backbone += params_embed
                params_embed = []

            param_groups = [
                {"params": params_backbone, "lr": self.cfg.finetune.lr_backbone},
                {"params": params_embed, "lr": self.cfg.finetune.lr_head},
                {"params": params_head, "lr": self.cfg.finetune.lr_head},
            ]
        else:
            raise NotImplementedError

        super()._init_optimizer(param_groups=param_groups)
