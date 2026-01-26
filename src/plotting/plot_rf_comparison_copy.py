import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

DATASETS = ['Calibration', 'Control', 'eICU', 'MIMIC-III', 'MIMIC-IV']
THRESHOLDS = ['15', '50']


def create_rf_comparison_plot(csv_path: str, output_dir: str = None):
    """
    Create a single bar chart where databases are on the x-axis and
    metrics for threshold 15 and 50 are plotted next to each other.
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded RF comparison CSV from {csv_path}")
    except Exception as e:
        logger.error(f"Failed to load CSV {csv_path}: {e}")
        return

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)

    os.makedirs(output_dir, exist_ok=True)

    # Build long-format dataframe
    records = []

    for dataset in DATASETS:
        for threshold in THRESHOLDS:
            col = f"{dataset}_{threshold}"
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping.")
                continue

            for _, row in df.iterrows():
                records.append({
                    "Dataset": dataset,
                    "Metric_Threshold": f"{row['Metric']} ({threshold})",
                    "Metric": row["Metric"],
                    "Threshold": threshold,
                    "Value": row[col]
                })

    long_df = pd.DataFrame(records)

    # Plot
    sns.set_style("whitegrid")
    plt.figure(figsize=(14, 7))

    ax = sns.barplot(
        data=long_df,
        x="Dataset",
        y="Value",
        hue="Metric_Threshold",
        edgecolor="black"
    )

    ax.set_xlabel("Database", fontsize=14, weight="bold")
    ax.set_ylabel("Value", fontsize=14, weight="bold")
    ax.set_ylim(0.5, 1.0)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(
        title="Metric (Threshold)",
        fontsize=11,
        title_fontsize=12,
        loc="lower right"
    )

    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(output_dir, "rf_metrics_thresholds_by_database.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    logger.info(f"Saved combined RF comparison plot to {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Create combined RF comparison plot from CSV")
    parser.add_argument("--csv", "-c", required=True, help="Path to the RF overview comparison CSV file")
    parser.add_argument("--output", "-o", help="Output directory for the chart")

    args = parser.parse_args()

    create_rf_comparison_plot(args.csv, args.output)
