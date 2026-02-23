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
from experiments.tagging.embedding import embed_tagging_data
from experiments.tagging.experiment import TopTaggingExperiment

ARCHS = ["tr", "lloca", "part", "slim"]
SIZES = np.arange(-3.0, 2.1, step=1.0)
BATCHSIZES = [512]
STEPS = 10
JETSIZE = 50


def main(save=True, steps=STEPS):
    for bs in BATCHSIZES:
        print(f"################ batchsize={bs} ################")
        single_batchsize(bs, save=save, steps=steps)


def single_batchsize(bs, save=True, steps=STEPS):
    results = dict()
    results["system_info"] = get_system_info()
    print(results["system_info"])
    results["benchmarking"] = {"steps": steps, "jet_size": JETSIZE, "batchsize": bs}

    t0 = time.time()
    for size in SIZES:
        print(f"################ {size} ################")
        results[size] = dict()
        for arch in ARCHS:
            all_dicts = {}
            best_dict = {"mean": math.inf}
            for amp in [False, True]:
                for compile in [False, True]:
                    for checkpoint in [False, True]:
                        mode = f"{'' if amp else 'no-'}amp,{'' if compile else 'no-'}compile,{'' if checkpoint else 'no-'}checkpoint"
                        current_dict = single_model(
                            arch,
                            size,
                            amp,
                            compile,
                            checkpoint,
                            mode,
                            bs=bs,
                            steps=steps,
                        )
                        all_dicts[mode] = current_dict.copy()
                        if current_dict["mean"] < best_dict["mean"]:
                            current_dict["best_mode"] = mode
                            best_dict = current_dict

            results[size][arch] = best_dict.copy()
            for key, value in all_dicts.items():
                results[size][arch][key] = value
            print(
                f"best {arch:<6} {size:>6.1f}: time = {best_dict['mean']:.2f} -{best_dict['std_minus']:.2f} +{best_dict['std_plus']:.2f} ms; memory_alloc = {best_dict['memory_alloc']:.2e} GB; memory reserved = {best_dict['memory_resvd']:.2e} GB ({best_dict['best_mode']})"
            )

    dt = time.time() - t0
    print(f"Finished scan after {dt:.2f}s")
    results["total_time"] = dt
    if save:
        with open(f"cost_estimate/train_gpu_bs{bs}.json", "w") as file:
            json.dump(results, file, indent=2)


def single_model(
    arch, size, amp, compile, checkpoint, mode, bs, steps=STEPS, warmup_steps=100
):
    experiments.logger.LOGGER.disabled = True  # turn off logging
    torch.manual_seed(42)
    assert torch.cuda.is_available()

    # create experiment environment
    with hydra.initialize(config_path="../config", version_base=None):
        overrides = [
            f"model={arch}",
            f"model.net.size={size}",
            "save=false",
            f"training.batchsize={bs}",
            "data.dataset=mini",
            "gpus=1",
            f"model.use_amp={amp}",
            f"model.net.compile={compile}",
            f"model.net.checkpoint_blocks={checkpoint}",
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
    optimizer = torch.optim.Adam(exp.model.parameters(), lr=1e-3)

    if JETSIZE is None:

        def cycle(iterable):
            while True:
                yield from iterable

        iterator = iter(cycle(exp.train_loader))

    times = []
    torch.cuda.reset_peak_memory_stats(exp.device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    for step in range(warmup_steps + steps):
        if JETSIZE is not None:
            embedding = get_rnd_batch(
                exp.cfg.data, batchsize=bs, jet_size=JETSIZE, device=exp.device
            )
        else:
            while True:
                # to avoid incomplete batches
                batch = next(iterator)
                fourmomenta, scalars, ptr, label = exp._extract_batch(batch)
                if label.shape[0] == bs:
                    break
            embedding = embed_tagging_data(
                fourmomenta,
                scalars,
                ptr,
                exp.cfg.data,
            )
            embedding["num_graphs"] = label.shape[0]
        start.record()
        out, _, _ = exp.model(embedding)
        label = torch.randn(bs, 1, device=exp.device)
        loss = exp.loss(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end.record()
        end.synchronize()
        if step > warmup_steps:
            times.append(start.elapsed_time(end))
    memory_alloc = torch.cuda.max_memory_allocated(exp.device) / 1024**3
    memory_resvd = torch.cuda.max_memory_reserved(exp.device) / 1024**3

    quants = np.quantile(times, [0.2, 0.5, 0.8])
    mean = quants[1]
    std_minus = quants[1] - quants[0]
    std_plus = quants[2] - quants[1]

    torch.compiler.reset()  # otherwise torch does recompiles

    print(
        f"{arch:<6} {size:>6.1f}: time = {mean:.2f} -{std_minus:.2f} +{std_plus:.2f} ms; memory_alloc = {memory_alloc:.2e} GB; memory reserved = {memory_resvd:.2e} GB ({mode})"
    )
    return dict(
        mean=mean,
        std_minus=std_minus,
        std_plus=std_plus,
        memory_alloc=memory_alloc,
        memory_resvd=memory_resvd,
    )


if __name__ == "__main__":
    main()
