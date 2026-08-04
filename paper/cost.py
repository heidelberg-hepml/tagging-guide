from matplotlib.backends.backend_pdf import PdfPages

from paper.metrics import load_data, merge_data, plot_metric_scatter
from paper.plot import COST_LABELS

ARCHS = ["tr", "part", "slim", "lloca", "lgatr", "lgatr-sparse"]
SIZES = [-2.0, -1.0, 0.0, 1.0, 2.0]


def main():
    basics = load_data("cost_estimate/basics.json")
    energy = load_data("cost_estimate/energy_model.json")
    for entry in energy.values():
        for arch in entry.values():
            arch["energy"] *= 1e-12
    cpu = merge_data(load_data("cost_estimate/inference_cpu.json"), basics)
    gpu = merge_data(load_data("cost_estimate/inference_gpu.json"), basics)
    train = merge_data(load_data("cost_estimate/train_gpu_bs512.json"), basics)
    # energy_model also has a "flops" key (estimated); keep it last so it wins
    # over the measured "flops" in basics, while "params" survives from basics.
    basics_energy = merge_data(basics, energy)

    pages = [
        (basics, "params", "flops", COST_LABELS["flops_measured"], False),
        (basics_energy, "params", "flops", COST_LABELS["flops_estimated"], False),
        (cpu, "params", "mean", COST_LABELS["inference_cpu"], True),
        (gpu, "params", "mean", COST_LABELS["inference_gpu_bs512"], True),
        (cpu, "params", "memory_rss", COST_LABELS["memory_cpu"], False),
        (gpu, "params", "memory_alloc", COST_LABELS["memory_gpu_bs512"], False),
        (basics_energy, "params", "energy", COST_LABELS["energy"], False),
        (train, "params", "mean", COST_LABELS["train_gpu_bs512"], True),
        (train, "params", "memory_alloc", COST_LABELS["memory_train_gpu_bs512"], True),
    ]

    with PdfPages("paper/cost.pdf") as pdf:
        for data, x_key, y_key, ylabel, yerr in pages:
            plot_metric_scatter(
                pdf,
                data,
                SIZES,
                xlabel=COST_LABELS["params"],
                ylabel=ylabel,
                archs=ARCHS,
                x_key=x_key,
                y_key=y_key,
                yerr=yerr,
                xscale="log",
                yscale="log",
            )


if __name__ == "__main__":
    main()
