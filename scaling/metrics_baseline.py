from matplotlib.backends.backend_pdf import PdfPages

from scaling.metrics import load_data, merge_data, plot_metric_scatter

ARCHS = ["tr", "part", "slim", "lloca", "lgatr", "lgatr-sparse"]
SIZES = [-2.0, -1.0, 0.0, 1.0, 2.0]


def main():
    gpu = load_data("cost_estimate/inference_gpu_bs512.json")
    cpu_energy = merge_data(
        load_data("cost_estimate/inference_cpu.json"),
        load_data("cost_estimate/energy_model.json"),
    )

    with PdfPages("scaling/metrics_baseline.pdf") as pdf:
        plot_metric_scatter(
            pdf,
            gpu,
            SIZES,
            xlabel="GPU memory [GB]",
            ylabel="GPU inference time [ms]",
            archs=ARCHS,
            x_key="memory_alloc",
            y_key="mean",
        )
        plot_metric_scatter(
            pdf,
            cpu_energy,
            SIZES,
            xlabel="Energy [pJ]",
            ylabel="CPU inference time [ms]",
            archs=ARCHS,
            x_key="energy",
            y_key="mean",
        )


if __name__ == "__main__":
    main()
