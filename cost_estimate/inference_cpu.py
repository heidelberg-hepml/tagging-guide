import ctypes
import gc
import json
import math
import multiprocessing
import time

import hydra
import numpy as np
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


def read_memory_status():
    with open("/proc/self/status") as file:
        fields = dict(line.split(":", 1) for line in file)
    return {key: int(fields[key].split()[0]) * 1024 for key in ("VmRSS", "VmHWM", "RssAnon")}


def reset_peak_rss():
    with open("/proc/self/clear_refs", "w") as file:
        file.write("5")


def max_tensor_allocated(model, embedding):
    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(activities=activities, profile_memory=True) as prof:
        model(*embedding)
    events = [
        e
        for e in prof.profiler.kineto_results.events()
        if e.name() == "[memory]" and e.device_type() == torch.autograd.DeviceType.CPU
    ]
    events.sort(key=lambda e: e.start_ns())
    return np.cumsum([0] + [e.nbytes() for e in events]).max()


def main(save=True, steps=STEPS):
    results = dict()

    results["system_info"] = get_system_info()
    print(results["system_info"])
    results["benchmarking"] = {
        "steps": steps,
        "jet_size": JETSIZE,
        "batchsize": BATCHSIZE,
        "num_threads": torch.get_num_threads(),
    }

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
                f"best {arch:<6} {size:>6.1f}: time = {best_dict['mean']:.2f} +{best_dict['std_plus']:.2f} -{best_dict['std_minus']:.2f} ms; memory_rss = {best_dict['memory_rss']:.2e} GB; memory_alloc = {best_dict['memory_alloc']:.2e} GB ({best_dict['best_mode']})"
            )

    dt = time.time() - t0
    print(f"Finished scan after {dt:.2f}s")
    results["total_time"] = dt
    if save:
        with open("cost_estimate/inference_cpu.json", "w") as file:
            json.dump(results, file, indent=2)


def _memory_worker(args, conn):
    conn.send(single_model(*args))
    conn.close()


def single_model_subprocess(arch, size, amp, compile, mode, steps=STEPS):
    ctx = multiprocessing.get_context("spawn")
    recv_conn, send_conn = ctx.Pipe(duplex=False)
    args = (arch, size, amp, compile, mode, steps)
    p = ctx.Process(target=_memory_worker, args=(args, send_conn))
    p.start()
    send_conn.close()
    try:
        result = recv_conn.recv()
    except EOFError:
        result = None
    p.join()
    if p.exitcode != 0 or result is None:
        raise RuntimeError(f"memory worker failed ({arch}, {size}, {mode}): exit {p.exitcode}")
    return result


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
    baseline = read_memory_status()

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
    if hasattr(exp._model, "init_standardization"):
        # normally done with a real batch in _init_dataloader; buffer values
        # don't matter for cost benchmarking, only shapes do
        embedding = get_rnd_batch(
            exp.cfg.data, batchsize=BATCHSIZE, jet_size=JETSIZE, device=exp.device
        )
        exp._model.init_standardization(embedding[0], mask=embedding[-1], is_spurion=embedding[3])
    exp.model.eval()
    weights = sum(t.numel() * t.element_size() for t in exp.model.parameters())
    weights += sum(t.numel() * t.element_size() for t in exp.model.buffers())
    release_memory()
    initialized = read_memory_status()

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

    embedding = get_rnd_batch(
        exp.cfg.data, batchsize=BATCHSIZE, jet_size=JETSIZE, device=exp.device
    )
    release_memory()
    warm = read_memory_status()
    reset_peak_rss()
    for _ in range(memory_steps):
        exp.model(*embedding)
    peak_rss = read_memory_status()["VmHWM"]
    memory_rss = (
        initialized["RssAnon"] - baseline["RssAnon"] + peak_rss - warm["VmRSS"]
    ) / 1024**3
    memory_rss_warmup = (warm["RssAnon"] - initialized["RssAnon"]) / 1024**3
    memory_alloc = (weights + max_tensor_allocated(exp.model, embedding)) / 1024**3

    torch.compiler.reset()  # otherwise torch does recompiles

    print(
        f"{arch:<6} {size:>6.1f}: time = {mean:.2f} -{std_minus:.2f} +{std_plus:.2f} ms; memory_rss = {memory_rss:.2e} GB; memory_alloc = {memory_alloc:.2e} GB; memory_rss_warmup = {memory_rss_warmup:.2e} GB ({mode})"
    )
    return dict(
        mean=mean,
        std_minus=std_minus,
        std_plus=std_plus,
        memory_rss=memory_rss,
        memory_alloc=memory_alloc,
        memory_rss_warmup=memory_rss_warmup,
    )


if __name__ == "__main__":
    main()
