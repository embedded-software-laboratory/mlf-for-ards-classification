import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

# Set color palette for thresholds
THRESHOLD_COLORS = {
    '15': '#fabe50',
    '50': '#d85c41'
}

# Set pastel color palette for models
PASTEL_COLORS = {
    'AdaBoost': '#407fb7',
    'LightGBM': '#fabe50',
    'LR': '#d85c41',
    'RandomForest': '#d0d95c',
    'XGBoost': '#834e75'
}

DATASETS = ['Calibration', 'Control', 'eICU', 'MIMIC-III', 'MIMIC-IV']


def create_rf_comparison_plots(csv_path: str, output_dir: str = None):
    """
    Create a single bar chart comparing _15 and _50 thresholds across all datasets and metrics.
    """
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded RF comparison CSV from {csv_path}")
    except Exception as e:
        logger.error(f"Failed to load CSV {csv_path}: {e}")
        return

    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Reshape data for plotting
    data_list = []
    for metric in df['Metric']:
        for dataset in DATASETS:
            col_15 = f'{dataset}_15'
            col_50 = f'{dataset}_50'
            
            if col_15 in df.columns and col_50 in df.columns:
                value_15 = df.loc[df['Metric'] == metric, col_15].values[0]
                value_50 = df.loc[df['Metric'] == metric, col_50].values[0]
                
                data_list.append({
                    'Metric': metric,
                    'Dataset': dataset,
                    'Threshold': '15',
                    'Value': value_15
                })
                data_list.append({
                    'Metric': metric,
                    'Dataset': dataset,
                    'Threshold': '50',
                    'Value': value_50
                })
    
    df_plot = pd.DataFrame(data_list)
    
    base_name = os.path.basename(csv_path)
    name_without_ext = os.path.splitext(base_name)[0]
    
    # Plot 1: Accuracy and F1-Score
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 6))
    
    metrics_group1 = ['Accuracy', 'F1-Score']
    
    for idx, metric in enumerate(metrics_group1):
        df_metric = df_plot[df_plot['Metric'] == metric]
        
        sns.barplot(
            data=df_metric, 
            x='Dataset', 
            y='Value', 
            hue='Threshold', 
            palette=THRESHOLD_COLORS, 
            edgecolor='black',
            ax=axes1[idx]
        )
        
        axes1[idx].set_ylabel('Value', fontsize=18, weight='bold')
        axes1[idx].set_xlabel('Dataset', fontsize=18, weight='bold')
        axes1[idx].set_title(metric, fontsize=20, weight='bold')
        axes1[idx].legend(title='ARDS prevalence (%)', fontsize=16, title_fontsize=16, loc='lower left')
        axes1[idx].grid(axis='y', alpha=0.3, linestyle='--')
        axes1[idx].set_ylim(0.5, 1.0)
        axes1[idx].tick_params(axis='x', labelsize=16, rotation=15)
        axes1[idx].tick_params(axis='y', labelsize=16)
    
    plt.tight_layout()
    
    output_path1 = os.path.join(output_dir, f'{name_without_ext}_accuracy_f1.pdf')
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    logger.info(f"Saved Accuracy and F1-Score comparison plot to {output_path1}")
    plt.close()
    
    # Plot 2: Sensitivity and Specificity
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))
    
    metrics_group2 = ['Sensitivity', 'Specificity']
    
    for idx, metric in enumerate(metrics_group2):
        df_metric = df_plot[df_plot['Metric'] == metric]
        
        sns.barplot(
            data=df_metric, 
            x='Dataset', 
            y='Value', 
            hue='Threshold', 
            palette=THRESHOLD_COLORS, 
            edgecolor='black',
            ax=axes2[idx]
        )
        
        axes2[idx].set_ylabel('Value', fontsize=18, weight='bold')
        axes2[idx].set_xlabel('Dataset', fontsize=18, weight='bold')
        axes2[idx].set_title(metric, fontsize=20, weight='bold')
        axes2[idx].legend(title='ARDS prevalence (%)', fontsize=16, title_fontsize=16, loc='lower left')
        axes2[idx].grid(axis='y', alpha=0.3, linestyle='--')
        axes2[idx].set_ylim(0.5, 1.0)
        axes2[idx].tick_params(axis='x', labelsize=16, rotation=15)
        axes2[idx].tick_params(axis='y', labelsize=16)
    
    plt.tight_layout()
    
    output_path2 = os.path.join(output_dir, f'{name_without_ext}_sensitivity_specificity.pdf')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    logger.info(f"Saved Sensitivity and Specificity comparison plot to {output_path2}")
    plt.close()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Create RF comparison bar charts from CSV")
    parser.add_argument("--csv", "-c", help="Path to the RF overview comparison CSV file")
    parser.add_argument("--output", "-o", help="Output directory for the charts (default: same as CSV)")

    args = parser.parse_args()

    create_rf_comparison_plots(args.csv, args.output)