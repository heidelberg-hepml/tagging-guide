from matplotlib.backends.backend_pdf import PdfPages

from paper.metrics import load_data, merge_data, plot_gain_scatter
from paper.plot import COST_LABELS

ARCHS = ["tr", "part", "slim", "lloca", "lgatr", "lgatr-sparse"]
SIZES = [-2.0, -1.0, 0.0, 1.0, 2.0]
ABLATE_LABELS = {"compile": "compile", "amp": "AMP", "zeropad": "sparse reps"}


def main():
    basics = load_data("cost_estimate/basics.json")
    cpu = merge_data(load_data("cost_estimate/inference_cpu.json"), basics)
    gpu = merge_data(load_data("cost_estimate/inference_gpu.json"), basics)
    train = merge_data(load_data("cost_estimate/train_gpu_bs512.json"), basics)

    pages = [
        (cpu, "mean", ["compile", "amp"], "CPU inference time", False),
        (cpu, "memory_rss", ["compile", "amp"], "CPU inference memory", True),
        (gpu, "mean", ["compile", "amp", "zeropad"], "GPU inference time", False),
        (gpu, "memory_alloc", ["compile", "amp", "zeropad"], "GPU inference memory", True),
        (train, "mean", ["compile", "amp"], "GPU training time", False),
        (train, "memory_alloc", ["compile", "amp"], "GPU training memory", True),
    ]

    with PdfPages("paper/cost_gains.pdf") as pdf:
        for data, y_key, ablations, metric_label, reduction in pages:
            for ablate in ablations:
                kind = "reduction" if reduction else "gain"
                plot_gain_scatter(
                    pdf,
                    data,
                    SIZES,
                    xlabel=COST_LABELS["params"],
                    ylabel=f"{metric_label} {kind}",
                    corner=ABLATE_LABELS[ablate],
                    ablate=ablate,
                    archs=ARCHS,
                    x_key="params",
                    y_key=y_key,
                    reduction=reduction,
                    xscale="log",
                )


if __name__ == "__main__":
    main()
