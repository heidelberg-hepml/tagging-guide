from scaling.metrics import load_data, plot_metric_scatter

ARCHS = ["tr", "part", "slim", "lloca", "lgatr", "lgatr-sparse"]
SIZES = [-2.0, -1.0, 0.0, 1.0, 2.0]
COMPILE_ARCHS = ["lgatr", "lgatr-sparse", "slim"]


def main():
    data = load_data("cost_estimate/inference_gpu.json")
    plot_metric_scatter(
        "scaling/metrics_compile.pdf",
        data,
        SIZES,
        xlabel="GPU memory [GB]",
        ylabel="GPU inference time [ms]",
        archs=ARCHS,
        x_key="memory_alloc",
        y_key="mean",
        series=(("no-amp,no-compile", "-"), ("no-amp,compile", "--")),
        series_labels=("no compile", "compile"),
        skip=lambda model, mode: mode == "no-amp,compile" and model not in COMPILE_ARCHS,
    )


if __name__ == "__main__":
    main()
