import csv
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
 
SRC_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SRC_DIR / "accuracy_results_table.txt"
FULL_OUTPUT_FILE = SRC_DIR / "full_accuracy_results_table.txt"
PLOT_FILE = SRC_DIR / "average_accuracy_across_noise_settings.png"

RESULT_FILES = [
    "results_dense_junk.csv",
    "results_feature_dropout.csv",
    "results_bit_flip.csv",
]

PREPROCESSING_METHODS = {
    "Raw l1",
    "Graph-aware l1",
    "Raw mutual_info",
    "Graph-aware mutual_info",
}


def load_rows():
    # Read all three experiment result files into one list.
    rows = []
    for filename in RESULT_FILES:
        path = SRC_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing results file: {path}")
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["noise_level"] = float(row["noise_level"])
                row["noise_percent"] = float(row["noise_percent"])
                row["accuracy_mean"] = float(row["accuracy_mean"])
                row["accuracy_std"] = float(row["accuracy_std"])
                rows.append(row)

    return rows

def format_accuracy(value):
    # round accuracy values to four decimals
    return f"{float(value):.4f}"


def format_noise_level(row):
    # Show both the raw noise value used in code and its percentage equivalent
    noise_level = row["noise_level"]
    noise_percent = row["noise_percent"]
    return f"{noise_level:g} ({noise_percent:g}%)"

def make_ascii_table(headers, table_rows):
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(headers))
    ]

    header_line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    divider = "-+-".join("-" * width for width in widths)
    lines = [header_line, divider]
    lines.extend(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))) for row in table_rows)
    return "\n".join(lines) + "\n"


def make_full_table(rows):
    # This keeps the original detailed table, but it is too long for the report.
    headers = [
        "Noise Type",
        "Noise Level",
        "Method",
        "Accuracy Mean",
        "Accuracy Std",
    ]

    table_rows = [
        [
            row["noise_type"],
            format_noise_level(row),
            row["method"],
            format_accuracy(row["accuracy_mean"]),
            format_accuracy(row["accuracy_std"]),
        ]
        for row in rows
    ]

    widths = [max(len(headers[i]), *(len(row[i]) for row in table_rows)) for i in range(len(headers))]
    header_line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    divider = "-+-".join("-" * width for width in widths)
    lines = [header_line, divider]

    current_noise_type = None
    for row in table_rows:
        # Add dividers between the three noise types.
        if current_noise_type is not None and row[0] != current_noise_type:
            lines.append(divider)
        current_noise_type = row[0]
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))

    return "\n".join(lines) + "\n"


def rows_for_noise(rows, noise_type):
    return [row for row in rows if row["noise_type"] == noise_type]


def hardest_rows(rows, noise_type):
    noise_rows = rows_for_noise(rows, noise_type)
    hardest_level = max(row["noise_level"] for row in noise_rows)
    return [row for row in noise_rows if row["noise_level"] == hardest_level]


def clean_row_for_method(rows, noise_type, method):
    for row in rows_for_noise(rows, noise_type):
        if row["noise_level"] == 0.0 and row["method"] == method:
            return row
    return None


def best_row(rows):
    return max(rows, key=lambda row: row["accuracy_mean"])


def takeaway_for_noise(noise_type):
    if noise_type == "dense_junk":
        return "Preprocessing helps most when noise is extra irrelevant columns."
    if noise_type == "feature_dropout":
        return "Selection stays close, but missing signal is hard to recover."
    if noise_type == "bit_flip":
        return "Corrupted original features hurt most methods."
    return "Highest-noise setting summary."


def make_hardest_noise_summary(rows):
    # Report-friendly table where one row per noise type at the hardest setting.
    headers = [
        "Noise Type",
        "Hardest Level",
        "Best Overall",
        "Best Acc.",
        "Best Preprocessing",
        "Preproc Acc.",
        "Takeaway",
    ]

    table_rows = []
    for noise_type in sorted({row["noise_type"] for row in rows}):
        hardest = hardest_rows(rows, noise_type)
        best_overall = best_row(hardest)
        preprocessing_rows = [row for row in hardest if row["method"] in PREPROCESSING_METHODS]
        best_preprocessing = best_row(preprocessing_rows)

        table_rows.append(
            [
                noise_type,
                format_noise_level(best_overall),
                best_overall["method"],
                format_accuracy(best_overall["accuracy_mean"]),
                best_preprocessing["method"],
                format_accuracy(best_preprocessing["accuracy_mean"]),
                takeaway_for_noise(noise_type),
            ]
        )

    return make_ascii_table(headers, table_rows)


def make_robustness_drop_summary(rows):
    headers = [
        "Noise Type",
        "Method",
        "Clean Acc.",
        "Hardest Acc.",
        "Drop",
    ]

    key_methods = [
        "No selection",
        "Graph-aware l1",
        "Graph-aware mutual_info",
        "Autoencoder",
    ]

    table_rows = []
    for noise_type in sorted({row["noise_type"] for row in rows}):
        hardest = hardest_rows(rows, noise_type)
        hardest_by_method = {row["method"]: row for row in hardest}

        for method in key_methods:
            clean = clean_row_for_method(rows, noise_type, method)
            hard = hardest_by_method.get(method)
            if clean is None or hard is None:
                continue

            drop = clean["accuracy_mean"] - hard["accuracy_mean"]
            table_rows.append(
                [
                    noise_type,
                    method,
                    format_accuracy(clean["accuracy_mean"]),
                    format_accuracy(hard["accuracy_mean"]),
                    format_accuracy(drop),
                ]
            )

    return make_ascii_table(headers, table_rows)


def make_summary_table(rows):
    return (
        "Highest-Noise Summary\n"
        "=====================\n"
        + make_hardest_noise_summary(rows)
        + "\nAccuracy Drop From Clean to Hardest Noise\n"
        + "=========================================\n"
        + make_robustness_drop_summary(rows)
    )


def plot_average_accuracy_across_noise_settings(rows):
    if plt is None:
        print("Skipping plot because matplotlib is not installed.")
        return

    # Average each method across all nonzero noise levels from all three noise types
    accuracies_by_method = {}
    for row in rows:
        if row["noise_level"] == 0.0:
            continue

        method = row["method"]
        accuracy = row["accuracy_mean"]
        accuracies_by_method.setdefault(method, []).append(accuracy)

    methods = list(accuracies_by_method.keys())
    averages = [
        sum(accuracies_by_method[method]) / len(accuracies_by_method[method])
        for method in methods
    ]

    sorted_results = sorted(zip(methods, averages), key=lambda item: item[1], reverse=True)
    methods = [method for method, _ in sorted_results]
    averages = [average for _, average in sorted_results]

    plt.figure(figsize=(11, 6))
    plt.bar(methods, averages)
    plt.xlabel("Method")
    plt.ylabel("Average test accuracy")
    plt.title("Average Accuracy Across All Noise Settings")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_FILE)
    plt.close()
    print(f"Wrote average accuracy plot to {PLOT_FILE}")


def main():
    rows = load_rows()
    summary_table = make_summary_table(rows)
    full_table = make_full_table(rows)

    OUTPUT_FILE.write_text(summary_table)
    FULL_OUTPUT_FILE.write_text(full_table)
    print(f"Wrote compact summary to {OUTPUT_FILE}")
    print(f"Wrote full table with {len(rows)} result rows to {FULL_OUTPUT_FILE}")

    plot_average_accuracy_across_noise_settings(rows)

if __name__ == "__main__":
    main()
