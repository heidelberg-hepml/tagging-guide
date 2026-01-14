# Should be evaluated on GPU
# otherwise the transformer FLOPs will be off, because it is not using flash-attention
import json
import time

import hydra
import numpy as np

import experiments.logger
from experiments.tagging.experiment import TopTaggingExperiment

ARCHS = ["tr", "lloca", "part", "slim"]
SIZES = ["xxs", "xs", "s", "m", "l", "xl"]


def main(save=True, steps=10, jet_size=50):
    results = {a: dict() for a in ARCHS}
    for size in SIZES:
        print(f"################ {size} ################")
        for arch in ARCHS:
            modelname = f"{arch}_{size}"
            mean, std_minus, std_plus = single_model(modelname, steps=steps, jet_size=jet_size)
            results[arch][size] = dict(mean=mean, std_minus=std_minus, std_plus=std_plus)

    if save:
        with open("cost_estimate/time_cpu.json", "w") as file:
            json.dump(results, file, indent=2)


def single_model(modelname, steps=10, jet_size=50):
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

    iterator = iter(exp.train_loader)
    times = []
    for _ in range(steps):
        data = next(iterator)
        if jet_size is not None:
            while data.x.shape[0] < jet_size:
                data = next(iterator)
            data.x = data.x[:jet_size]
            data.scalars = data.scalars[:jet_size]
            data.batch = data.batch[:jet_size]
            data.ptr[-1] = jet_size
        t0 = time.time()
        exp._get_ypred_and_label(data)
        dt = time.time() - t0
        times.append(dt)
    times = times[int(0.1 * steps) :]

    quants = np.quantile(times, [0.2, 0.5, 0.8]) * 1e3
    mean = quants[1]
    std_minus = quants[1] - quants[0]
    std_plus = quants[2] - quants[1]

    print(f"{modelname:<10}: time = {mean:.2f} -{std_minus:.2f} +{std_plus:.2f} ms")
    # print(flop_counter.get_table(depth=5))

    return mean, std_minus, std_plus


if __name__ == "__main__":
    main()
