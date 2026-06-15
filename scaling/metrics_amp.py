from scaling.metrics import load_data, plot_metric_scatter

ARCHS = ["tr", "part", "slim", "lloca", "lgatr", "lgatr-sparse"]
SIZES = [-2.0, -1.0, 0.0, 1.0, 2.0]
FLOAT32_ARCHS = ["lgatr", "lloca", "slim", "lgatr-sparse"]


def main():
    data = load_data("cost_estimate/inference_gpu.json")
    plot_metric_scatter(
        "scaling/metrics_amp.pdf",
        data,
        SIZES,
        xlabel="GPU memory [GB]",
        ylabel="GPU inference time [ms]",
        archs=ARCHS,
        x_key="memory_alloc",
        y_key="mean",
        series=(("no-amp,no-compile", "-"), ("amp,no-compile", "--")),
        series_labels=("FP32", "AMP"),
        skip=lambda model, mode: mode == "amp,no-compile" and model in FLOAT32_ARCHS,
    )


if __name__ == "__main__":
    main()
