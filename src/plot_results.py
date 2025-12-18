import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import numpy as np
from sklearn.metrics import roc_curve, auc

logger = logging.getLogger(__name__)

# Set pastel color palette
PASTEL_COLORS = {
    'AdaBoost': '#8ebae5',      # Pastel red
    'LightGBM': '#b8d698',      # Pastel green
    'LR': '#a8859e',            # Pastel blue
    'RandomForest': '#fdd48f',  # Pastel yellow
    'XGBoost': '#e69679'        # Pastel orange
}

PASTEL_PALETTE = list(PASTEL_COLORS.values())


class ResultsAnalyzer:
    """
    Analyzes and visualizes ML evaluation results from JSON files.
    """

    def __init__(self, results_json_path: str, output_dir: str = None):
        """
        Initialize the ResultsAnalyzer.

        Args:
            results_json_path: Path to the main results JSON file
            output_dir: Directory to save output plots and tables
        """
        self.results_json_path = results_json_path

        # Ensure output directory exists (fix for FileNotFoundError)
        if output_dir:
            self.output_dir = str(Path(output_dir).resolve())
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = self._create_output_dir()

        self.results_data = None
        self.models_data = {}

        logger.info(f"Initializing ResultsAnalyzer with results file: {results_json_path}")
        logger.info(f"Output directory: {self.output_dir}")


    def _create_output_dir(self) -> str:
        """Create output directory for results."""
        # Use absolute path to avoid issues with relative paths
        results_file_path = Path(self.results_json_path).resolve()
        output_dir = results_file_path.parent / "results_visualization"
        output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Created output directory: {output_dir}")
        return str(output_dir)

    def load_results(self) -> bool:
        """
        Load the main results JSON file.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Resolve to absolute path
            json_path = Path(self.results_json_path).resolve()
            with open(json_path, 'r') as f:
                self.results_data = json.load(f)
            logger.info(f"Successfully loaded results from {json_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading results JSON: {e}")
            return False

    def extract_model_metrics(self) -> Dict:
        """
        Extract key metrics for each model.

        Returns:
            Dictionary containing model metrics
        """
        if not self.results_data:
            logger.error("Results data not loaded")
            return {}

        metrics_dict = {}
        contained_models = self.results_data.get('contained_model_results', {})

        for model_name, model_data in contained_models.items():
            try:
                # Get evaluation data
                contained_evals = model_data.get('contained_evals', {})
                evaluation = contained_evals.get('Evaluation', {})

                # Get Standard optimizer data (most common): Optimizers possible: Standard, MaxTPR, MaxTPRMinFPR, GeometricRoot
                contained_optimizers = evaluation.get('contained_optimizers', {})
                standard_optimizer = contained_optimizers.get('GeometricRoot', {})
                contained_splits = standard_optimizer.get('contained_splits', {})
                eval_split = contained_splits.get('Evaluation split', {})
                contained_metrics = eval_split.get('contained_metrics', {})

                # Extract metrics
                metrics_dict[model_name] = {
                    'AUC': contained_metrics.get('AUC', {}).get('metric_value', 0),
                    'Accuracy': contained_metrics.get('Accuracy', {}).get('metric_value', 0),
                    'F1Score': contained_metrics.get('F1Score', {}).get('metric_value', 0),
                    'Sensitivity': contained_metrics.get('Sensitivity', {}).get('metric_value', 0),
                    'Specificity': contained_metrics.get('Specificity', {}).get('metric_value', 0),
                    'PPV': contained_metrics.get('PPV', {}).get('metric_value', 0),
                    'NPV': contained_metrics.get('NPV', {}).get('metric_value', 0),
                    'MCC': contained_metrics.get('MCC', {}).get('metric_value', 0),
                    'FPR': contained_metrics.get('FPR', {}).get('metric_value', []),
                    'TPR': contained_metrics.get('TPR', {}).get('metric_value', []),
                }
                logger.debug(f"Extracted metrics for model: {model_name}")

            except Exception as e:
                logger.error(f"Error extracting metrics for model {model_name}: {e}")

        self.models_data = metrics_dict
        return metrics_dict

    def create_metrics_comparison_table(self):
        """Create a table comparing key metrics across models."""
        if not self.models_data:
            logger.warning("No models data available for table creation")
            return

        logger.info("Creating metrics comparison table...")

        # Create DataFrame
        df = pd.DataFrame({
            model: {
                'AUC': metrics['AUC'],
                'Accuracy': metrics['Accuracy'],
                'F1-Score': metrics['F1Score'],
                'Sensitivity': metrics['Sensitivity'],
                'Specificity': metrics['Specificity'],
                'PPV': metrics['PPV'],
                'NPV': metrics['NPV'],
                'MCC': metrics['MCC']
            }
            for model, metrics in self.models_data.items()
        }).T

        # Round to 4 decimal places
        df = df.round(4)

        # Create figure and table
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')

        # Create table
        table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index,
                        cellLoc='center', loc='center', colWidths=[0.12] * len(df.columns))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)

        # Color header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')

        # Color rows with alternating pastel colors
        for i, model in enumerate(df.index):
            table[(i + 1, -1)].set_facecolor(PASTEL_COLORS.get(model, '#FFFFFF'))

        plt.title('Model Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        # Save figure
        output_path = os.path.join(self.output_dir, 'metrics_comparison_table.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metrics comparison table to {output_path}")
        plt.close()

        # Also save as CSV
        csv_path = os.path.join(self.output_dir, 'metrics_comparison_table.csv')
        df.to_csv(csv_path)
        logger.info(f"Saved metrics comparison table CSV to {csv_path}")

    def create_metrics_barchart(self):
        """Create bar charts for individual metrics."""
        if not self.models_data:
            logger.warning("No models data available for bar chart creation")
            return

        logger.info("Creating metrics bar charts...")

        metrics_to_plot = ['AUC', 'Accuracy', 'F1Score', 'Sensitivity', 'Specificity']
        models = list(self.models_data.keys())
        model_colors = [PASTEL_COLORS.get(m, '#CCCCCC') for m in models]

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            values = [self.models_data[model].get(metric, 0) for model in models]

            bars = ax.bar(models, values, color=model_colors, edgecolor='black', linewidth=1.5)

            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=9)

            ax.set_ylabel(metric, fontsize=11, fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)

        # Remove extra subplot
        fig.delaxes(axes[5])

        plt.suptitle('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'metrics_barchart.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metrics bar chart to {output_path}")
        plt.close()

    def create_roc_curves(self):
        """Create ROC curves for each model."""
        if not self.models_data:
            logger.warning("No models data available for ROC curves")
            return

        logger.info("Creating ROC curves...")

        fig, ax = plt.subplots(figsize=(10, 8))
        models = list(self.models_data.keys())

        for model in models:
            fpr = self.models_data[model].get('FPR', [])
            tpr = self.models_data[model].get('TPR', [])

            # Handle both single values and arrays
            if not isinstance(fpr, (list, np.ndarray)) or len(fpr) == 0:
                logger.debug(f"Skipping ROC curve for {model}: invalid FPR data")
                continue

            fpr = np.array(fpr) if isinstance(fpr, list) else fpr
            tpr = np.array(tpr) if isinstance(tpr, list) else tpr

            # Calculate AUC
            roc_auc = self.models_data[model].get('AUC', 0)

            ax.plot(fpr, tpr, linewidth=2.5, label=f'{model} (AUC = {roc_auc:.4f})',
                   color=PASTEL_COLORS.get(model, '#CCCCCC'))

        # Plot diagonal
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')

        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'roc_curves.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curves to {output_path}")
        plt.close()

    def create_performance_heatmap(self):
        """Create a heatmap of all metrics."""
        if not self.models_data:
            logger.warning("No models data available for heatmap")
            return

        logger.info("Creating performance heatmap...")

        metrics_to_include = ['AUC', 'Accuracy', 'F1Score', 'Sensitivity', 'Specificity', 'PPV', 'NPV', 'MCC']
        df = pd.DataFrame({
            model: {metric: self.models_data[model].get(metric, 0) for metric in metrics_to_include}
            for model in self.models_data.keys()
        }).T

        fig, ax = plt.subplots(figsize=(12, 6))

        sns.heatmap(df, annot=True, fmt='.4f', cmap='Blues', vmin=0, vmax=1,
                   cbar_kws={'label': 'Metric Value'}, linewidths=0.5, ax=ax,
                   cbar=True, square=False)

        ax.set_title('Model Performance Heatmap', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Models', fontsize=12, fontweight='bold')

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'performance_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved performance heatmap to {output_path}")
        plt.close()

    def create_radar_chart(self):
        """Create a radar chart comparing models."""
        if not self.models_data:
            logger.warning("No models data available for radar chart")
            return

        logger.info("Creating radar chart...")

        metrics = ['AUC', 'Accuracy', 'F1Score', 'Sensitivity', 'Specificity']
        num_vars = len(metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        for model in self.models_data.keys():
            values = [self.models_data[model].get(metric, 0) for metric in metrics]
            values += values[:1]  # Complete the circle

            ax.plot(angles, values, 'o-', linewidth=2, label=model,
                   color=PASTEL_COLORS.get(model, '#CCCCCC'))
            ax.fill(angles, values, alpha=0.25, color=PASTEL_COLORS.get(model, '#CCCCCC'))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'radar_chart.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved radar chart to {output_path}")
        plt.close()

    def create_auc_comparison(self):
        """Create a focused comparison of AUC values."""
        if not self.models_data:
            logger.warning("No models data available for AUC comparison")
            return

        logger.info("Creating AUC comparison chart...")

        models = list(self.models_data.keys())
        auc_values = [self.models_data[model].get('AUC', 0) for model in models]
        colors = [PASTEL_COLORS.get(m, '#CCCCCC') for m in models]

        fig, ax = plt.subplots(figsize=(10, 6))

        bars = ax.barh(models, auc_values, color=colors, edgecolor='black', linewidth=1.5)

        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, auc_values)):
            ax.text(value - 0.02, bar.get_y() + bar.get_height() / 2,
                   f'{value:.4f}', ha='right', va='center', fontsize=11, fontweight='bold')

        ax.set_xlabel('AUC Score', fontsize=12, fontweight='bold')
        ax.set_title('Area Under Curve (AUC) Comparison', fontsize=14, fontweight='bold')
        ax.set_xlim([0.5, 1.0])
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        plt.tight_layout()

        output_path = os.path.join(self.output_dir, 'auc_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved AUC comparison to {output_path}")
        plt.close()

    def generate_summary_report(self):
        """Generate a text summary report."""
        if not self.models_data:
            logger.warning("No models data available for summary report")
            return

        logger.info("Generating summary report...")

        report_lines = [
            "=" * 80,
            "MODEL EVALUATION SUMMARY REPORT",
            "=" * 80,
            f"\nResults File: {self.results_json_path}",
            f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "\n" + "=" * 80,
            "MODEL RANKINGS BY AUC",
            "=" * 80
        ]

        # Sort by AUC
        sorted_models = sorted(self.models_data.items(),
                              key=lambda x: x[1]['AUC'], reverse=True)

        for rank, (model, metrics) in enumerate(sorted_models, 1):
            report_lines.append(f"\n{rank}. {model}")
            report_lines.append(f"   AUC:        {metrics['AUC']:.4f}")
            report_lines.append(f"   Accuracy:   {metrics['Accuracy']:.4f}")
            report_lines.append(f"   F1-Score:   {metrics['F1Score']:.4f}")
            report_lines.append(f"   Sensitivity:{metrics['Sensitivity']:.4f}")
            report_lines.append(f"   Specificity:{metrics['Specificity']:.4f}")
            report_lines.append(f"   PPV:        {metrics['PPV']:.4f}")
            report_lines.append(f"   NPV:        {metrics['NPV']:.4f}")
            report_lines.append(f"   MCC:        {metrics['MCC']:.4f}")

        report_lines.extend([
            "\n" + "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])

        report_text = "\n".join(report_lines)
        logger.info(report_text)

        # Save report
        report_path = os.path.join(self.output_dir, 'summary_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)

        logger.info(f"Saved summary report to {report_path}")

    def create_feature_selection_heatmap(self, comparison_csv_path: str = None):
        """
        Create a heatmap from a Comparison.csv containing Model, Feature_Number, F1.
        If comparison_csv_path is None, try to find Comparison.csv in the same folder as results_json_path
        under a Feature_Selection subfolder.
        """
        # determine path
        if comparison_csv_path:
            csv_path = Path(comparison_csv_path).resolve()
        else:
            # try a sensible default next to the results json
            json_parent = Path(self.results_json_path).resolve().parent
            csv_path = (json_parent / "Feature_Selection" / "Comparison.csv").resolve()

        if not csv_path.exists():
            logger.warning(f"Comparison CSV not found: {csv_path}")
            return

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            logger.error(f"Failed to read Comparison CSV {csv_path}: {e}")
            return

        # pivot table: rows = Model, cols = Feature_Number, values = F1
        try:
            pivot = df.pivot_table(index="Model", columns="Feature_Number", values="F1", aggfunc="mean")
        except Exception as e:
            logger.error(f"Failed to pivot Comparison CSV: {e}")
            return

        # sort columns numerically
        try:
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        except Exception:
            pass

        # ensure output dir exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # create heatmap
        plt.figure(figsize=(max(6, pivot.shape[1] * 0.8), max(4, pivot.shape[0] * 0.6)))
        cmap = sns.color_palette(PASTEL_PALETTE, as_cmap=True)
        ax = sns.heatmap(pivot, annot=True, fmt=".4f", cmap=cmap, vmin=0.0, vmax=1.0, cbar_kws={"label": "F1"})
        ax.set_title("Feature Selection — F1 by Model and Feature Count", fontsize=12, fontweight="bold")
        ax.set_xlabel("Number of Features", fontsize=11)
        ax.set_ylabel("Model", fontsize=11)
        plt.tight_layout()

        out_png = os.path.join(self.output_dir, "feature_selection_heatmap.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved feature selection heatmap to {out_png}")

        # also save pivot as csv for downstream use
        out_csv = os.path.join(self.output_dir, "feature_selection_pivot.csv")
        pivot.to_csv(out_csv)
        logger.info(f"Saved pivot CSV to {out_csv}")

        return out_png, out_csv

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        logger.info("Starting full analysis...")

        if not self.load_results():
            logger.error("Failed to load results")
            return False

        self.extract_model_metrics()

        logger.info("Creating visualizations...")
        self.create_metrics_comparison_table()
        self.create_metrics_barchart()
        self.create_roc_curves()
        self.create_performance_heatmap()
        self.create_radar_chart()
        self.create_auc_comparison()
        self.generate_summary_report()

        logger.info(f"Analysis complete! Results saved to {self.output_dir}")
        return True


def main(results_json_path: str, output_dir: str = None):
    """
    Main function to run the results analysis.

    Args:
        results_json_path: Path to the results JSON file
        output_dir: Optional output directory for results
    """
    analyzer = ResultsAnalyzer(results_json_path, output_dir)
    return analyzer.run_full_analysis()


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Analyze and visualize ML evaluation results")
    parser.add_argument("results_json", help="Path to the results JSON file")
    parser.add_argument("--output", "-o", help="Output directory for visualizations", default=None)

    args = parser.parse_args()

    main(args.results_json, args.output)