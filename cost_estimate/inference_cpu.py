import ctypes
import gc
import json
import math
import multiprocessing
import threading
import time

import hydra
import numpy as np
import psutil
import torch

import experiments.logger
from cost_estimate.estimate import FLOAT32_ARCHS
from cost_estimate.utils import get_rnd_batch, get_system_info
from experiments.tagging.experiment import TopTaggingExperiment

ARCHS = ["tr", "lloca", "part", "slim", "lgatr", "lgatr-sparse"]
SIZES = np.arange(-2.0, 2.1, step=1.0)
STEPS = 100
JETSIZE = 50
BATCHSIZE = 1
MEMORY_STEPS = 20

try:
    _LIBC = ctypes.CDLL("libc.so.6")
except OSError:
    _LIBC = None


def release_memory():
    gc.collect()
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


class RSSPeakSampler:
    def __init__(self, interval=5e-4):
        self.interval = interval
        self.process = psutil.Process()
        self.max_rss = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.max_rss = max(self.max_rss, self.process.memory_info().rss)
            time.sleep(self.interval)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join()
        self.max_rss = max(self.max_rss, self.process.memory_info().rss)


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
                    current_dict = single_model_subprocess(
                        arch, size, amp, compile, mode, steps=steps
                    )
                    all_dicts[mode] = current_dict.copy()

                    if current_dict["mean"] < best_dict["mean"] and (
                        not amp or arch not in FLOAT32_ARCHS
                    ):
                        current_dict["best_mode"] = mode
                        best_dict = current_dict

            results[size][arch] = best_dict.copy()
            for key, value in all_dicts.items():
                results[size][arch][key] = value
            print(
                f"best {arch:<6} {size:>6.1f}: time = {best_dict['mean']:.2f} +{best_dict['std_plus']:.2f} -{best_dict['std_minus']:.2f} ms; memory = {best_dict['memory_rss']:.2e} GB ({best_dict['best_mode']})"
            )

    dt = time.time() - t0
    print(f"Finished scan after {dt:.2f}s")
    results["total_time"] = dt
    if save:
        with open("cost_estimate/inference_cpu.json", "w") as file:
            json.dump(results, file, indent=2)


def _memory_worker(args, queue):
    queue.put(single_model(*args))


def single_model_subprocess(arch, size, amp, compile, mode, steps=STEPS):
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    args = (arch, size, amp, compile, mode, steps)
    p = ctx.Process(target=_memory_worker, args=(args, queue))
    p.start()
    p.join()
    if p.exitcode != 0 or queue.empty():
        raise RuntimeError(f"memory worker failed ({arch}, {size}, {mode}): exit {p.exitcode}")
    return queue.get()


@torch.inference_mode()
def single_model(
    arch, size, amp, compile, mode, steps=STEPS, warmup_steps=10, memory_steps=MEMORY_STEPS
):
    experiments.logger.LOGGER.disabled = True  # turn off logging
    torch.manual_seed(42)
    if compile:
        # import + initialize the dynamo/inductor stack before taking the RSS
        # baseline, so memory_rss only contains model-specific compile artifacts
        dummy = torch.compile(torch.nn.Linear(8, 8))
        dummy(torch.randn(1, 8))
        torch.compiler.reset()
        del dummy
    release_memory()
    baseline_rss = psutil.Process().memory_info().rss

    # create experiment environment
    with hydra.initialize(config_path="../config", version_base=None):
        overrides = [
            f"model={arch}",
            f"model.net.size={size}",
            "save=false",
            f"training.batchsize={BATCHSIZE}",
            "data.dataset=mini",
            "gpu=false",
            f"model.use_amp={amp}",
            "model.zeropad=true",
        ]
        if arch == "gn3":
            overrides.append(f"model.compile={compile}")
        else:
            overrides.append(f"model.net.compile={compile}")
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
        exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.model.eval()

    times = []
    for step in range(warmup_steps + steps):
        embedding = get_rnd_batch(
            exp.cfg.data, batchsize=BATCHSIZE, jet_size=JETSIZE, device=exp.device
        )
        t0 = time.perf_counter_ns()
        exp.model(*embedding)
        dt = (time.perf_counter_ns() - t0) * 1e-6
        if step >= warmup_steps:
            times.append(dt)

    quants = np.quantile(times, [0.2, 0.5, 0.8])
    mean = quants[1]
    std_minus = quants[1] - quants[0]
    std_plus = quants[2] - quants[1]

    release_memory()
    sampler = RSSPeakSampler()
    sampler.start()
    for _ in range(memory_steps):
        embedding = get_rnd_batch(
            exp.cfg.data, batchsize=BATCHSIZE, jet_size=JETSIZE, device=exp.device
        )
        exp.model(*embedding)
    sampler.stop()
    memory_rss = (sampler.max_rss - baseline_rss) / 1024**3

    torch.compiler.reset()  # otherwise torch does recompiles

    print(
        f"{arch:<6} {size:>6.1f}: time = {mean:.2f} -{std_minus:.2f} +{std_plus:.2f} ms; memory = {memory_rss:.2e} GB ({mode})"
    )
    return dict(mean=mean, std_minus=std_minus, std_plus=std_plus, memory_rss=memory_rss)


if __name__ == "__main__":
    main()
