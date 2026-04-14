import json

import hydra
import numpy as np
from torch.utils.flop_counter import FlopCounterMode

import experiments.logger
from cost_estimate.utils import get_rnd_batch
from experiments.tagging.experiment import TopTaggingExperiment

ARCHS = ["tr", "lloca", "part", "slim", "gn3"]
SIZES = np.arange(-3.0, 3.1, step=1.0)
JETSIZE = 50


def main(save=True):
    results = dict()
    for size in SIZES:
        print(f"################ {size} ################")
        results[size] = dict()
        for arch in ARCHS:
            params, flops = single_model(arch, size)
            results[size][arch] = dict(params=params, flops=flops)

    if save:
        with open("cost_estimate/basics.json", "w") as file:
            json.dump(results, file, indent=2)


def single_model(arch, size):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../config", version_base=None):
        overrides = [
            f"model={arch}",
            f"model.net.size={size}",
            "save=false",
            "training.batchsize=1",
            "data.dataset=mini",
            "model.zeropad=true",
        ]
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
        exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()

    params = sum(p.numel() for p in exp.model.parameters())

    embedding = get_rnd_batch(exp.cfg.data, batchsize=1, jet_size=JETSIZE, device=exp.device)

    with FlopCounterMode(display=False) as flop_counter:
        exp.model(*embedding)
    flops = flop_counter.get_total_flops()

    print(f"{arch:<6} {size:>6.1f}: params= {params:>10}\t flops(bs=1)= {flops:.2e}")
    # print(flop_counter.get_table(depth=5))

    return params, flops


if __name__ == "__main__":
    main()
