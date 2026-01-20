import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

legend_labels = ["Calibration Data", "Control Data", "eICU Data", "MIMIC-III Data", "MIMIC-IV Data"]

logger = logging.getLogger(__name__)

# Set pastel color palette for models
PASTEL_COLORS = {
    'AdaBoost': '#407fb7',
    'LightGBM': '#fabe50',
    'LR': '#d85c41',
    'RandomForest': '#d0d95c',
    'XGBoost': '#834e75'
}


def create_overview_barchart(csv_path: str, output_dir: str = None):
    """
    Create a bar chart from the overview CSV showing F1 scores across datasets and models.
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded overview CSV from {csv_path}")
    except Exception as e:
        logger.error(f"Failed to load CSV {csv_path}: {e}")
        return

    # Melt the dataframe for plotting
    df_melted = df.melt(id_vars='Model', var_name='Dataset', value_name='F1_Score')

    # Clean dataset names
    df_melted['Dataset'] = df_melted['Dataset'].str.replace('_F1_15', '')
    df_melted['Dataset'] = df_melted['Dataset'].str.replace('_F1_50', '')

    # Create the plot
    plt.figure(figsize=(14, 8))
    sns.barplot(data=df_melted, x='Dataset', y='F1_Score', hue='Model', palette=PASTEL_COLORS, edgecolor='black')

    plt.ylabel('F1-Score', fontsize=14, weight='bold')
    plt.xlabel('Dataset', fontsize=14, weight='bold')
    plt.legend(title='Model', fontsize=14, title_fontsize=14, loc='lower left')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.ylim(0.3, 1)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.tight_layout()

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    base_name = os.path.basename(csv_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_path = os.path.join(output_dir, f'{name_without_ext}.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved overview bar chart to {output_path}")
    plt.show()
    plt.close()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Create overview bar chart from CSV")
    parser.add_argument("--csv", "-c", help="Path to the overview CSV file")
    parser.add_argument("--output", "-o", help="Output directory for the chart (default: same as CSV)")

    args = parser.parse_args()

    create_overview_barchart(args.csv, args.output)

# Example usage: python plot_overview.py --c E:\Promotion\ml-framework\Results\All_models_overview_50.csv --o E:\Promotion\ml-framework\Results\Plots