from scaling.metrics import load_data, plot_metric_scatter

ARCHS = ["tr", "slim", "lloca", "lgatr", "lgatr-sparse"]
SIZES = [-2.0, -1.0, 0.0, 1.0, 2.0]


def main():
    data = load_data("cost_estimate/inference_gpu.json")
    plot_metric_scatter(
        "scaling/metrics_zeropad.pdf",
        data,
        SIZES,
        xlabel="GPU memory [GB]",
        ylabel="GPU inference time [ms]",
        archs=ARCHS,
        x_key="memory_alloc",
        y_key="mean",
        ablate="zeropad",
        series_labels=("best", "zero-pad"),
    )


if __name__ == "__main__":
    main()
