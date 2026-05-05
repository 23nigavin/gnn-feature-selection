import csv
from pathlib import Path
import matplotlib.pyplot as plt
 
SRC_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SRC_DIR / "accuracy_results_table.txt"    # just output into the src direc
PLOT_FILE = SRC_DIR / "average_accuracy_across_noise_settings.png"

RESULT_FILES = ["results_dense_junk.csv",
    "results_feature_dropout.csv",
    "results_bit_flip.csv",
]

'''
reads all three experiment result files into one list of CSV rows
'''
def load_rows():
    rows = []
    for filename in RESULT_FILES:
        path = SRC_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing results file: {path}")
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows.extend(reader)

    return rows

def format_accuracy(value):
    # round accuracy values to four decimals
    return f"{float(value):.4f}"


def format_noise_level(row):
    # show both the raw noise value used in code and its percentage equivalent
    noise_level = float(row["noise_level"])
    noise_percent = float(row["noise_percent"])
    return f"{noise_level:g} ({noise_percent:g}%)"

def make_table(rows):
    # convert the CSV rows into a table
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

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(headers))
    ]

    # build a simple ASCII table
    header_line = " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers)))
    divider = "-+-".join("-" * width for width in widths)
    lines = [header_line, divider]

    current_noise_type = None
    for row in table_rows:
        # add divider between dense_junk, feature_dropout, and bit_flip b
        if current_noise_type is not None and row[0] != current_noise_type:
            lines.append(divider)
        current_noise_type = row[0]
        lines.append(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))

    return "\n".join(lines) + "\n"


def plot_average_accuracy_across_noise_settings(rows):
    # average each method across all nonzero noise levels from all three noise types
    accuracies_by_method = {}
    for row in rows:
        if float(row["noise_level"]) == 0.0:
            continue

        method = row["method"]
        accuracy = float(row["accuracy_mean"])
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
    table = make_table(rows)
    # save the output .txt file
    OUTPUT_FILE.write_text(table)
    print(f"Wrote {len(rows)} result rows to {OUTPUT_FILE}")

    plot_average_accuracy_across_noise_settings(rows)

if __name__ == "__main__":
    main()
