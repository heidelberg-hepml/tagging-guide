import json

import hydra
import numpy as np
from lloca.reps.tensorreps import TensorReps
from omegaconf import OmegaConf

from cost_estimate.estimate import estimate_energy, estimate_flops

ARCHS = ["tr", "lloca", "part", "slim", "gn3"]
SIZES = np.arange(-3.0, 3.1, step=1.0)
DTYPES = ["float32", "float16"]
JETSIZE = 50

OmegaConf.register_new_resolver("eval", eval, replace=True)


def main(save=True, jet_size=JETSIZE):
    results = dict()
    for size in SIZES:
        print(f"################ {size} ################")
        results[size] = dict()
        for arch in ARCHS:
            results[size][arch] = single_model(arch, size, jet_size=jet_size)

    if save:
        with open("cost_estimate/energy_model.json", "w") as file:
            json.dump(results, file, indent=2)


def single_model(arch, size, jet_size=JETSIZE):
    # create experiment environment
    with hydra.initialize(config_path="../config", version_base=None):
        overrides = [
            f"model={arch}",
            f"model.net.size={size}",
            "save=false",
            "training.batchsize=1",
            "data.dataset=mini",
            "gpus=0",
        ]
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)

    kwargs = dict(seqlen=jet_size)
    if arch in ["tr", "lloca"]:
        kwargs["blocks"] = cfg.model.net.num_blocks
        kwargs["channels"] = TensorReps(cfg.model.net.attn_reps).dim * cfg.model.net.num_heads
        kwargs["mlp_ratio"] = cfg.model.net.mlp_factor * 3 / 4  # GLU
        if arch == "lloca":
            architecture = "llocatransformer"
            kwargs["channels_framesnet"] = cfg.model.framesnet.equivectors.hidden_channels
            kwargs["layers_framesnet"] = cfg.model.framesnet.equivectors.num_layers_mlp
            kwargs["is_global"] = cfg.model.framesnet.is_global
        else:
            architecture = "transformer"
    elif arch == "part":
        architecture = "particletransformer"
        kwargs["blocks"] = cfg.model.net.num_layers + cfg.model.net.num_cls_layers
        kwargs["channels"] = cfg.model.net.helpers.hidden_dims
        kwargs["mlp_ratio"] = cfg.model.net.ffn_ratio * 3 / 4  # GLU
        kwargs["channels_pair"] = cfg.model.net.pair_embed_dims[0]
        kwargs["layers_pair"] = len(cfg.model.net.pair_embed_dims)
    elif arch == "slim":
        architecture = "lgatr-slim"
        kwargs["blocks"] = cfg.model.net.num_blocks
        kwargs["channels_v"] = cfg.model.net.hidden_v_channels
        kwargs["channels_s"] = cfg.model.net.hidden_s_channels
        kwargs["mlp_ratio"] = cfg.model.net.mlp_ratio * 3 / 4  # GLU
        kwargs["attn_ratio"] = cfg.model.net.attn_ratio
    elif arch == "gn3":
        architecture = "gn3"
        kwargs["blocks"] = cfg.model.net.encoder.num_layers
        kwargs["channels"] = cfg.model.net.encoder.embed_dim
        kwargs["mlp_ratio"] = 2 * 3 / 4  # ffn_ratio * GLU
    else:
        raise ValueError(f"architecture {arch} not implemented")

    results = dict()
    results["flops"] = estimate_flops(architecture=architecture, arch_kwargs=kwargs)
    for dtype in DTYPES:
        results[dtype] = estimate_energy(
            architecture=architecture,
            arch_kwargs=kwargs,
            dtype_a=dtype,
            dtype_w=dtype,
            dtype_default="float32",
            mode="H100-estimate",
        )

    print(
        f"{arch:<6} {size:>6.1f}: flops = {results['flops']:.2e} energy = {results['float16']:.2e}"
    )
    return results


if __name__ == "__main__":
    main()
