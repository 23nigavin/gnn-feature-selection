import csv
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
OUTPUT_FILE = SRC_DIR / "accuracy_results_table.txt"    # just output into the src direc

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


def main():
    rows = load_rows()
    table = make_table(rows)
    # save the output .txt file
    OUTPUT_FILE.write_text(table)
    print(f"Wrote {len(rows)} result rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
