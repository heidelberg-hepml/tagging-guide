"""Translate a composed hydra config into keyword arguments for cost_estimate.estimate."""

import torch
from lloca.reps.tensorreps import TensorReps
from omegaconf import OmegaConf

from experiments.tagging.embedding import get_num_auxiliary_scalars, get_spurion

# config/model/*.yaml -> family
FAMILIES = {
    "tr": "tr",
    "tr_4block": "tr",
    "tag_transformer": "tr",
    "tag_top_transformer": "tr",
    "lloca": "tr",
    "lloca_4block": "tr",
    "part": "part",
    "tag_ParT": "part",
    "slim": "slim",
    "slim_4block": "slim",
    "tag_slim": "slim",
    "lgatr": "lgatr",
    "lgatr_4block": "lgatr",
    "lgatr-sparse": "lgatr",
    "tag_lgatr": "lgatr",
    "gn3": "gn3",
    "gn3_4block": "gn3",
    "GN3V01": "gn3",
    "pet2_s": "pet2",
    "pet2_m": "pet2",
    "pet2_l": "pet2",
    "tag_lorentznet": "lorentznet",
}


def get_arch_kwargs(arch, cfg, jet_size, **extra):
    """Return (architecture, arch_kwargs) for cost_estimate.estimate, where arch is the name
    of the config/model/*.yaml used to compose cfg, jet_size excludes the spurions, and extra
    holds settings that are not part of the model config (e.g. the PET2 mode)."""
    family = FAMILIES.get(arch)
    if family is None:
        raise ValueError(f"architecture {arch} not implemented")

    def select(key, default=None):
        return OmegaConf.select(cfg, f"model.net.{key}", default=default)

    # LLoCa networks see the spurions like the internally equivariant ones,
    # see TaggingExperiment.init_physics
    frames = "equivectors" in OmegaConf.select(cfg, "model.framesnet", default={})
    equivariant = frames or family in ["lgatr", "slim", "lorentznet"]

    # spurions and global token are added by the cost functions, because
    # they are visible to different parts of the network
    kwargs = dict(seqlen=jet_size)
    if equivariant:
        kwargs["num_spurions"] = num_spurions(cfg.data)
    if family == "tr":
        glu = 3 / 2 if "transformer_v2" in cfg.model.net._target_ else 1
        kwargs["blocks"] = cfg.model.net.num_blocks
        kwargs["channels"] = TensorReps(cfg.model.net.attn_reps).dim * cfg.model.net.num_heads
        kwargs["mlp_ratio"] = cfg.model.net.mlp_factor * glu
        architecture = "llocatransformer" if frames else "transformer"
        if frames:
            kwargs["channels_framesnet"] = cfg.model.framesnet.equivectors.hidden_channels
            kwargs["layers_framesnet"] = cfg.model.framesnet.equivectors.num_layers_mlp
            kwargs["is_global"] = cfg.model.framesnet.is_global
    elif family == "part":
        architecture = "particletransformer"
        glu = 3 / 2 if select("version", 1) > 1 else 1
        kwargs["blocks"] = cfg.model.net.num_layers + cfg.model.net.num_cls_layers
        embed_dims = select("embed_dims", [])
        kwargs["channels"] = (
            embed_dims[-1] if len(embed_dims) > 0 else select("helpers.hidden_dims")
        )
        kwargs["mlp_ratio"] = select("ffn_ratio", 4) * glu
        kwargs["channels_pair"] = cfg.model.net.pair_embed_dims[0]
        kwargs["layers_pair"] = len(cfg.model.net.pair_embed_dims)
    elif family == "slim":
        architecture = "lgatr-slim"
        kwargs["blocks"] = cfg.model.net.num_blocks
        kwargs["channels_v"] = cfg.model.net.hidden_v_channels
        kwargs["channels_s"] = cfg.model.net.hidden_s_channels
        kwargs["mlp_ratio"] = cfg.model.net.mlp_ratio
        kwargs["attn_ratio"] = cfg.model.net.attn_ratio
    elif family == "lgatr":
        architecture = "lgatr"
        kwargs["blocks"] = cfg.model.net.num_blocks
        kwargs["channels_mv"] = cfg.model.net.hidden_mv_channels
        kwargs["channels_s"] = cfg.model.net.hidden_s_channels
        kwargs["mlp_ratio"] = cfg.model.net.mlp.mlp_ratio
        kwargs["attn_ratio"] = cfg.model.net.attention.attn_ratio
        kwargs["sparse_gp"] = cfg.model.net.primitives.sparse_gp
        kwargs["sparse_linear"] = cfg.model.net.primitives.sparse_linear
        kwargs["subgroup"] = cfg.model.net.primitives.subgroup
    elif family == "lorentznet":
        architecture = "lorentznet"
        kwargs["blocks"] = cfg.model.net.n_layers
        kwargs["channels"] = cfg.model.net.n_hidden
        kwargs["num_scalars"] = get_num_auxiliary_scalars(cfg.data.auxiliary_scalars)
    elif family == "gn3":
        architecture = "gn3"
        kwargs["seqlen"] += cfg.model.net.encoder.num_registers
        kwargs["blocks"] = cfg.model.net.encoder.num_layers
        kwargs["channels"] = cfg.model.net.encoder.embed_dim
        kwargs["mlp_ratio"] = 2 * 3 / 2  # ffn_ratio * GLU
    elif family == "pet2":
        architecture = "pet2"
        kwargs["blocks"] = cfg.model.net.num_transformers
        kwargs["blocks_head"] = cfg.model.net.num_transformers_head
        kwargs["channels"] = cfg.model.net.base_dim
        kwargs["num_heads"] = cfg.model.net.num_heads
        kwargs["num_tokens"] = cfg.model.net.num_tokens
        kwargs["mlp_ratio"] = cfg.model.net.mlp_ratio
        kwargs["K"] = cfg.model.net.K
        kwargs["num_coord"] = cfg.model.net.num_coord
        kwargs["use_int"] = cfg.model.net.use_int

    kwargs.update(extra)
    return architecture, kwargs


def num_spurions(cfg_data):
    """Number of spurions that embed_tagging_data prepends to each jet."""
    spurion = get_spurion(
        beam_reference=cfg_data.beam_reference,
        add_time_reference=cfg_data.add_time_reference,
        two_beams=cfg_data.two_beams,
        device="cpu",
        dtype=torch.float32,
    )
    return spurion.shape[0]
