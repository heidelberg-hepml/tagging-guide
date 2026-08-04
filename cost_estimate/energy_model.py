import json

import hydra
import numpy as np
from omegaconf import OmegaConf

from cost_estimate.arch_kwargs import get_arch_kwargs
from cost_estimate.estimate import FLOAT32_ARCHS, estimate_energy, estimate_flops

ARCHS = ["tr", "lloca", "part", "slim", "lgatr", "lgatr-sparse"]
SIZES = np.arange(-2.0, 2.1, step=1.0)
DTYPES = ["float32", "bfloat16"]
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
            "gpu=false",
        ]
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)

    architecture, kwargs = get_arch_kwargs(arch, cfg, jet_size)

    results = dict()
    results["flops"] = estimate_flops(architecture=architecture, arch_kwargs=kwargs)
    for dtype in DTYPES:
        results[f"energy_{dtype}"] = estimate_energy(
            architecture=architecture,
            arch_kwargs=kwargs,
            dtype_a=dtype,
            dtype_w=dtype,
            dtype_default=dtype,
            mode="H100-estimate",
        )
    results["energy"] = (
        results["energy_float32"] if arch in FLOAT32_ARCHS else results["energy_bfloat16"]
    )

    print(
        f"{arch:<6} {size:>6.1f}: flops = {results['flops']:.2e} energy = {results['energy']:.2e}"
    )
    return results


if __name__ == "__main__":
    main()
