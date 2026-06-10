# Should be evaluated on GPU
import json
import time

import hydra
import numpy as np
import torch

import experiments.logger
from cost_estimate.utils import get_rnd_batch, get_system_info
from experiments.tagging.embedding import embed_tagging_data
from experiments.tagging.experiment import TopTaggingExperiment

ARCHS = ["tr", "lloca", "part", "slim", "lgatr"]
SIZES = np.arange(-2.0, 2.1, step=1.0)
BATCHSIZE = 512
STEPS = 100
WARMUP_STEPS = 100
JETSIZE = None
TRAIN = False


def main(save=True, steps=STEPS, warmup_steps=WARMUP_STEPS, sizes=SIZES):
    results = dict()
    results["system_info"] = get_system_info()
    print(results["system_info"])
    results["benchmarking"] = {
        "steps": steps,
        "jet_size": JETSIZE,
        "batchsize": BATCHSIZE,
        "sizes": list(sizes),
        "train": TRAIN,
    }

    t0 = time.time()
    for size in sizes:
        print(f"################ size={size} ################")
        results[str(size)] = dict()
        for arch in ARCHS:
            results[str(size)][arch] = dict()
            for zp in [True, False]:
                if arch == "part" and not zp:
                    continue
                mode = "zeropad" if zp else "no-zeropad"
                all_dicts = {}
                best_dict = None
                for compile in [False, True]:
                    compile_key = "compile" if compile else "no-compile"
                    current_dict = single_model(
                        arch, zp, compile, size, steps=steps, warmup_steps=warmup_steps
                    )
                    all_dicts[compile_key] = current_dict.copy()
                    if best_dict is None or current_dict["mean"] < best_dict["mean"]:
                        best_dict = current_dict.copy()
                        best_dict["best_compile"] = compile_key
                for key, value in all_dicts.items():
                    best_dict[key] = value
                results[str(size)][arch][mode] = best_dict

    dt = time.time() - t0
    print(f"Finished scan after {dt:.2f}s")
    results["total_time"] = dt
    if save:
        with open("cost_estimate/inference_gpu_zeropad.json", "w") as file:
            json.dump(results, file, indent=2)


def single_model(arch, zeropad, compile, size, steps=STEPS, warmup_steps=WARMUP_STEPS):
    bs = BATCHSIZE
    experiments.logger.LOGGER.disabled = True  # turn off logging
    torch.manual_seed(0)
    assert torch.cuda.is_available()

    with hydra.initialize(config_path="../config", version_base=None):
        overrides = [
            f"model={arch}",
            f"model.net.size={size}",
            "save=false",
            f"training.batchsize={bs}",
            "data.dataset=mini",
            "gpu=true",
            "model.use_amp=false",
            f"model.zeropad={zeropad}",
            f"model.net.compile={compile}",
            "float32_matmul_precision=high",
        ]
        cfg = hydra.compose(config_name="toptagging", overrides=overrides)
        exp = TopTaggingExperiment(cfg)
    exp._init()
    exp.init_physics()
    exp.init_model()
    exp.init_data()
    exp._init_dataloader()
    exp._init_loss()
    if TRAIN:
        exp.model.train()
        optimizer = torch.optim.Adam(exp.model.parameters(), lr=1e-3)
    else:
        exp.model.eval()

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
    grad_ctx = torch.enable_grad() if TRAIN else torch.inference_mode()
    with grad_ctx:
        for step in range(warmup_steps + steps):
            if JETSIZE is not None:
                embedding = get_rnd_batch(
                    exp.cfg.data, batchsize=bs, jet_size=JETSIZE, device=exp.device
                )
            else:
                while True:
                    batch = next(iterator)
                    fourmomenta, scalars, label_real, _ = exp._extract_batch(batch)
                    if label_real.shape[0] == bs:
                        break
                embedding = embed_tagging_data(fourmomenta, scalars, exp.cfg.data)
            start.record()
            if TRAIN:
                out, _, _ = exp.model(*embedding)
                label = torch.randn(bs, 1, device=exp.device)
                loss = exp.loss(out, label).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                exp.model(*embedding)
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

    torch.compiler.reset()

    mode = "zeropad" if zeropad else "no-zeropad"
    compile_key = "compile" if compile else "no-compile"
    print(
        f"{arch:<6} size={size:>5.1f} {mode:<11} {compile_key:<10}: time = {mean:.2f} -{std_minus:.2f} +{std_plus:.2f} ms; memory_alloc = {memory_alloc:.2e} GB; memory reserved = {memory_resvd:.2e} GB"
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
