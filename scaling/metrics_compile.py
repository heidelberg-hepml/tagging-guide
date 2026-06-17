from matplotlib.backends.backend_pdf import PdfPages

from scaling.metrics import load_data, merge_data, plot_metric_scatter

ARCHS = ["slim", "lgatr", "lgatr-sparse"]
SIZES = [-2.0, -1.0, 0.0, 1.0, 2.0]


def main():
    gpu = load_data("cost_estimate/inference_gpu.json")
    cpu = merge_data(
        load_data("cost_estimate/inference_cpu.json"),
        load_data("cost_estimate/basics.json"),
    )

    with PdfPages("scaling/metrics_compile.pdf") as pdf:
        plot_metric_scatter(
            pdf,
            gpu,
            SIZES,
            xlabel="GPU memory [GB]",
            ylabel="GPU inference time [ms]",
            archs=ARCHS,
            x_key="memory_alloc",
            y_key="mean",
            ablate="compile",
            series_labels=("best", "no compile"),
        )
        plot_metric_scatter(
            pdf,
            cpu,
            SIZES,
            xlabel="FLOPs",
            ylabel="CPU inference time [ms]",
            archs=ARCHS,
            x_key="flops",
            y_key="mean",
            ablate="compile",
            series_labels=("best", "no compile"),
        )


if __name__ == "__main__":
    main()
