# Should be evaluated on GPU
# otherwise the transformer FLOPs will be off, because it is not using flash-attention
import json
import time

import hydra
import numpy as np
import torch

import experiments.logger
from cost_estimate.utils import get_system_info
from experiments.tagging.experiment import TopTaggingExperiment

ARCHS = ["tr", "lloca", "part", "slim"]
SIZES = ["xxs", "xs", "s", "m", "l", "xl"]
STEPS = 100
JETSIZE = 50


def main(save=True, steps=STEPS, jet_size=JETSIZE):
    results = dict()

    results["system_info"] = get_system_info()
    print(results["system_info"])
    results["benchmarking"] = {"steps": steps, "jet_size": jet_size}

    for size in SIZES:
        print(f"################ {size} ################")
        results[size] = dict()
        for arch in ARCHS:
            modelname = f"{arch}_{size}"
            results[size][arch] = single_model(modelname, steps=steps, jet_size=jet_size)

    if save:
        with open("cost_estimate/inference_cpu.json", "w") as file:
            json.dump(results, file, indent=2)


@torch.no_grad()
def single_model(modelname, steps=STEPS, warmup_steps=10, jet_size=JETSIZE):
    experiments.logger.LOGGER.disabled = True  # turn off logging

    # create experiment environment
    with hydra.initialize(config_path="../config", version_base=None):
        overrides = [
            f"model=tag_{modelname}",
            "save=false",
            "training.batchsize=1",
            "data.dataset=mini",
            "gpus=0",
        ]
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
        exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    exp.model.eval()

    times = []
    iterator = iter(exp.train_loader)
    for step in range(warmup_steps + steps):
        data = next(iterator)
        if jet_size is not None:
            while data.x.shape[0] < jet_size:
                data = next(iterator)
            data.x = data.x[:jet_size]
            data.scalars = data.scalars[:jet_size]
            data.batch = data.batch[:jet_size]
            data.ptr[-1] = jet_size
        t0 = time.perf_counter_ns()
        exp._get_ypred_and_label(data)
        dt = (time.perf_counter_ns() - t0) * 1e-6
        if step > warmup_steps:
            times.append(dt)

    quants = np.quantile(times, [0.2, 0.5, 0.8])
    mean = quants[1]
    std_minus = quants[1] - quants[0]
    std_plus = quants[2] - quants[1]

    print(f"{modelname:<10}: time = {mean:.2f} -{std_minus:.2f} +{std_plus:.2f} ms")
    return dict(mean=mean, std_minus=std_minus, std_plus=std_plus)


if __name__ == "__main__":
    main()
