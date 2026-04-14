# Should be evaluated on GPU
# otherwise the transformer FLOPs will be off, because it is not using flash-attention
import json
import math
import time

import hydra
import numpy as np
import torch

import experiments.logger
from cost_estimate.utils import get_rnd_batch, get_system_info
from experiments.tagging.experiment import TopTaggingExperiment

ARCHS = ["tr", "lloca", "part", "slim", "gn3"]
SIZES = np.arange(-2.0, 2.1, step=1.0)
STEPS = 100
JETSIZE = 50
BATCHSIZE = 1


def main(save=True, steps=STEPS):
    results = dict()

    results["system_info"] = get_system_info()
    print(results["system_info"])
    results["benchmarking"] = {"steps": steps, "jet_size": JETSIZE, "batchsize": BATCHSIZE}

    t0 = time.time()
    for size in SIZES:
        print(f"################ {size} ################")
        results[size] = dict()
        for arch in ARCHS:
            all_dicts = {}
            best_dict = {"mean": math.inf}
            for amp in [False, True]:
                for compile in [False, True]:
                    mode = f"{'' if amp else 'no-'}amp,{'' if compile else 'no-'}compile"
                    current_dict = single_model(arch, size, amp, compile, mode, steps=steps)
                    all_dicts[mode] = current_dict.copy()

                    if current_dict["mean"] < best_dict["mean"]:
                        current_dict["best_mode"] = mode
                        best_dict = current_dict

            results[size][arch] = best_dict.copy()
            for key, value in all_dicts.items():
                results[size][arch][key] = value
            print(
                f"best {arch:<6} {size:>6.1f}: time = {best_dict['mean']:.2f} -{best_dict['std_plus']:.2f} +{best_dict['std_minus']:.2f} ms ({best_dict['best_mode']})"
            )

    dt = time.time() - t0
    print(f"Finished scan after {dt:.2f}s")
    results["total_time"] = dt
    if save:
        with open("cost_estimate/inference_cpu.json", "w") as file:
            json.dump(results, file, indent=2)


@torch.no_grad()
def single_model(arch, size, amp, compile, mode, steps=STEPS, warmup_steps=10):
    experiments.logger.LOGGER.disabled = True  # turn off logging
    torch.manual_seed(42)

    # create experiment environment
    with hydra.initialize(config_path="../config", version_base=None):
        overrides = [
            f"model={arch}",
            f"model.net.size={size}",
            "save=false",
            f"training.batchsize={BATCHSIZE}",
            "data.dataset=mini",
            "gpus=0",
            f"model.use_amp={amp}",
            "model.zeropad=true",
        ]
        if arch == "gn3":
            overrides.append(f"model.compile={compile}")
            overrides.append("model.attention_backend=torch-meff")
        else:
            overrides.append(f"model.net.compile={compile}")
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
    for step in range(warmup_steps + steps):
        embedding = get_rnd_batch(
            exp.cfg.data, batchsize=BATCHSIZE, jet_size=JETSIZE, device=exp.device
        )
        t0 = time.perf_counter_ns()
        exp.model(*embedding)
        dt = (time.perf_counter_ns() - t0) * 1e-6
        if step > warmup_steps:
            times.append(dt)

    quants = np.quantile(times, [0.2, 0.5, 0.8])
    mean = quants[1]
    std_minus = quants[1] - quants[0]
    std_plus = quants[2] - quants[1]

    torch.compiler.reset()  # otherwise torch does recompiles

    print(f"{arch:<6} {size:>6.1f}: time = {mean:.2f} -{std_minus:.2f} +{std_plus:.2f} ms ({mode})")
    return dict(mean=mean, std_minus=std_minus, std_plus=std_plus)


if __name__ == "__main__":
    main()
