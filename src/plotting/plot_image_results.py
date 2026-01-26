#!/usr/bin/env python3
"""
Image Model Results Visualization Script

This script analyzes and visualizes evaluation results from Image Classification models
(ResNet, DenseNet, Vision Transformer) across different training methods (Last Layer, Last Block, Full Model).

Usage:
    python plot_image_results.py --results_dir "Results/ImageModels" --output_dir "Results/ImageModels/Analysis"
    
    Or for a single method:
    python plot_image_results.py --results_dir "Results/ImageModels/Last_Block" --output_dir "Results/ImageModels/Last_Block/plots"
"""

import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from typing import Dict, List, Tuple
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pastel color scheme for models (matching plot_overview.py)
MODEL_COLORS = {
    'ResNet': '#d0d95c',
    'DenseNet': '#834e75',
    'ViT': '#ced1f8',
}

# Pastel color scheme for methods
METHOD_COLORS = {
    'Last_Layer': '#407fb7',
    'Last_Block': '#fabe50',
    'Model': '#d85c41'
}


class ImageResultsAnalyzer:
    """
    Analyzes and visualizes Image Model evaluation results from JSON files.
    """

    def __init__(self, results_dir: str, output_dir: str = None):
        """
        Initialize the ImageResultsAnalyzer.

        Args:
            results_dir: Directory containing results JSON files (can be specific method or parent dir)
            output_dir: Directory to save output plots and CSV files
        """
        self.results_dir = Path(results_dir).resolve()
        
        if output_dir:
            self.output_dir = Path(output_dir).resolve()
        else:
            self.output_dir = self.results_dir / "analysis"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.all_results = {}  # Dict: {method: {model: metrics}}
        self.methods_found = []
        
        logger.info(f"Initialized ImageResultsAnalyzer")
        logger.info(f"Results directory: {self.results_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def discover_result_files(self) -> List[Tuple[str, Path]]:
        """
        Discover all image result JSON files in the directory.
        
        Returns:
            List of tuples (method_name, file_path)
        """
        result_files = []
        
        # Check if results_dir is a specific method folder (contains JSON directly)
        json_files = list(self.results_dir.glob('*_image_results.json'))
        if json_files:
            # Single method directory
            method_name = self.results_dir.name
            for json_file in json_files:
                result_files.append((method_name, json_file))
                logger.info(f"Found result file: {json_file.name} for method: {method_name}")
        else:
            # Parent directory containing multiple method folders
            for method_dir in self.results_dir.iterdir():
                if method_dir.is_dir():
                    json_files = list(method_dir.glob('*_image_results.json'))
                    for json_file in json_files:
                        result_files.append((method_dir.name, json_file))
                        logger.info(f"Found result file: {json_file.name} for method: {method_dir.name}")
        
        if not result_files:
            logger.warning(f"No image results JSON files found in {self.results_dir}")
        
        return result_files

    def load_all_results(self) -> bool:
        """
        Load all discovered result JSON files.
        
        Returns:
            True if at least one file was loaded successfully
        """
        result_files = self.discover_result_files()
        
        if not result_files:
            logger.error("No result files found to load")
            return False
        
        success_count = 0
        
        for method_name, json_path in result_files:
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                # Extract metrics for each model
                method_results = self._extract_metrics_from_json(data)
                
                if method_results:
                    self.all_results[method_name] = method_results
                    self.methods_found.append(method_name)
                    success_count += 1
                    logger.info(f"Successfully loaded {len(method_results)} model(s) from {method_name}")
                else:
                    logger.warning(f"No valid results extracted from {json_path}")
                    
            except Exception as e:
                logger.error(f"Error loading {json_path}: {e}")
        
        logger.info(f"Successfully loaded {success_count} result file(s)")
        return success_count > 0

    def _extract_metrics_from_json(self, data: Dict) -> Dict:
        """
        Extract metrics from a loaded JSON results file.
        
        Args:
            data: Loaded JSON data
            
        Returns:
            Dictionary mapping model names to their metrics
        """
        metrics_dict = {}
        contained_models = data.get('contained_model_results', {})
        
        for model_name, model_data in contained_models.items():
            try:
                # Navigate the nested structure
                contained_evals = model_data.get('contained_evals', {})
                evaluation = contained_evals.get('Evaluation', {})
                contained_optimizers = evaluation.get('contained_optimizers', {})
                standard_optimizer = contained_optimizers.get('Standard', {})
                contained_splits = standard_optimizer.get('contained_splits', {})
                eval_split = contained_splits.get('Evaluation split', {})
                contained_metrics = eval_split.get('contained_metrics', {})
                
                # Get additional info
                training_dataset = evaluation.get('training_dataset', {})
                test_dataset = evaluation.get('test_dataset', {})
                additional_info = training_dataset.get('additional_information', '')
                
                # Extract metrics
                metrics_dict[model_name] = {
                    'Accuracy': contained_metrics.get('Accuracy', {}).get('metric_value', 0),
                    'AUC': contained_metrics.get('AUC', {}).get('metric_value', 0),
                    'F1Score': contained_metrics.get('F1Score', {}).get('metric_value', 0),
                    'Sensitivity': contained_metrics.get('Sensitivity', {}).get('metric_value', 0),
                    'Specificity': contained_metrics.get('Specificity', {}).get('metric_value', 0),
                    'MCC': contained_metrics.get('MCC', {}).get('metric_value', 0),
                    'num_train_images': training_dataset.get('number_of_images', 0),
                    'num_test_images': test_dataset.get('number_of_images', 0),
                    'additional_info': additional_info
                }
                logger.debug(f"Extracted metrics for model: {model_name}")
                
            except Exception as e:
                logger.error(f"Error extracting metrics for model {model_name}: {e}")
        
        return metrics_dict

    def create_comprehensive_csv(self):
        """Create a comprehensive CSV with all results."""
        if not self.all_results:
            logger.warning("No results available for CSV creation")
            return
        
        logger.info("Creating comprehensive results CSV...")
        
        rows = []
        for method, models in self.all_results.items():
            for model_name, metrics in models.items():
                row = {
                    'Method': method,
                    'Model': model_name,
                    'Accuracy': metrics['Accuracy'],
                    'AUC': metrics['AUC'],
                    'F1-Score': metrics['F1Score'],
                    'Sensitivity': metrics['Sensitivity'],
                    'Specificity': metrics['Specificity'],
                    'MCC': metrics['MCC'],
                    'Train_Images': metrics['num_train_images'],
                    'Test_Images': metrics['num_test_images']
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by Method and Model
        df = df.sort_values(['Method', 'Model'])
        
        # Save CSV
        csv_path = self.output_dir / 'image_models_results.csv'
        df.to_csv(csv_path, index=False, float_format='%.4f')
        logger.info(f"Saved comprehensive CSV to {csv_path}")
        
        # Also create a pivot table for easier comparison
        pivot_metrics = ['Accuracy', 'AUC', 'F1-Score', 'Sensitivity', 'Specificity', 'MCC']
        
        for metric in pivot_metrics:
            pivot_df = df.pivot(index='Model', columns='Method', values=metric)
            pivot_path = self.output_dir / f'comparison_{metric.lower()}.csv'
            pivot_df.to_csv(pivot_path, float_format='%.4f')
            logger.info(f"Saved {metric} comparison CSV to {pivot_path}")
        
        return df

    def plot_metrics_by_method(self):
        """Create bar plots comparing metrics across methods for each model."""
        if not self.all_results:
            logger.warning("No results available for plotting")
            return
        
        logger.info("Creating metrics comparison plots by method...")
        
        metrics_to_plot = ['Accuracy', 'AUC', 'F1Score', 'Sensitivity', 'Specificity', 'MCC']
        
        # Collect all unique models across all methods
        all_models = set()
        for method_results in self.all_results.values():
            all_models.update(method_results.keys())
        all_models = sorted(list(all_models))
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            # Prepare data for grouped bar chart
            x = np.arange(len(all_models))
            width = 0.8 / len(self.methods_found)
            
            for i, method in enumerate(self.methods_found):
                method_results = self.all_results[method]
                values = [method_results.get(model, {}).get(metric, 0) for model in all_models]
                
                offset = (i - len(self.methods_found)/2 + 0.5) * width
                bars = ax.bar(x + offset, values, width, 
                             label=method, 
                             color=METHOD_COLORS.get(method, '#CCCCCC'),
                             edgecolor='black', linewidth=1)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2., height,
                               f'{height:.3f}', ha='center', va='bottom', 
                               fontsize=14, rotation=0)
            
            ax.set_ylabel(metric, fontsize=14, fontweight='bold')
            ax.set_xlabel('Model', fontsize=14, fontweight='bold')
            ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(all_models, rotation=45, ha='right', fontsize=14)
            ax.set_ylim([0, 1.05])
            ax.tick_params(axis='y', labelsize=14)
            ax.legend(title='Method', loc='lower left', fontsize=14, title_fontsize=14, framealpha=1, facecolor='white')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
        
        # plt.suptitle('Image Model Performance Across Methods', 
        #              fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        output_path = self.output_dir / 'metrics_by_method.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved metrics by method plot to {output_path}")
        plt.close()

    def plot_metrics_by_model(self):
        """Create bar plots comparing metrics across models for each method."""
        if not self.all_results:
            logger.warning("No results available for plotting")
            return
        
        logger.info("Creating metrics comparison plots by model...")
        
        metrics_to_plot = ['Accuracy', 'AUC', 'F1Score', 'Sensitivity', 'Specificity', 'MCC']
        
        for method_name, models_data in self.all_results.items():
            fig, axes = plt.subplots(2, 3, figsize=(16, 10))
            axes = axes.flatten()
            
            models = list(models_data.keys())
            model_colors = [MODEL_COLORS.get(m, '#CCCCCC') for m in models]
            
            for idx, metric in enumerate(metrics_to_plot):
                ax = axes[idx]
                values = [models_data[model].get(metric, 0) for model in models]
                
                bars = ax.bar(models, values, color=model_colors, 
                             edgecolor='black', linewidth=1.5, alpha=0.8)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                           f'{height:.4f}', ha='center', va='bottom', fontsize=14)
                
                ax.set_ylabel(metric, fontsize=14, fontweight='bold')
                ax.set_xlabel('Model', fontsize=14, fontweight='bold')
                ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
                ax.set_ylim([0, 1.05])
                ax.set_xticklabels(models, rotation=45, ha='right', fontsize=14)
                ax.tick_params(axis='y', labelsize=14)
                ax.grid(axis='y', alpha=0.3, linestyle='--')
                ax.set_axisbelow(True)
            
            # plt.suptitle(f'Model Performance - {method_name}', 
            #             fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            output_path = self.output_dir / f'metrics_{method_name}.pdf'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved metrics plot for {method_name} to {output_path}")
            plt.close()

    def plot_heatmap(self):
        """Create a heatmap showing all metrics across methods and models."""
        if not self.all_results:
            logger.warning("No results available for heatmap")
            return
        
        logger.info("Creating performance heatmap...")
        
        metrics_to_include = ['Accuracy', 'AUC', 'F1Score', 'Sensitivity', 'Specificity', 'MCC']
        
        # Build data for heatmap
        rows = []
        index_labels = []
        
        for method in self.methods_found:
            method_results = self.all_results[method]
            for model_name in sorted(method_results.keys()):
                metrics = method_results[model_name]
                row = [metrics.get(metric, 0) for metric in metrics_to_include]
                rows.append(row)
                index_labels.append(f"{method}_{model_name}")
        
        df = pd.DataFrame(rows, columns=metrics_to_include, index=index_labels)
        
        fig, ax = plt.subplots(figsize=(12, len(index_labels) * 0.5 + 2))
        
        sns.heatmap(df, annot=True, fmt='.4f', cmap='Blues', 
                   vmin=0, vmax=1, cbar_kws={'label': 'Metric Value'}, 
                   linewidths=0.5, ax=ax, cbar=True, square=False)
        
        ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Method_Model', fontsize=14, fontweight='bold')
        ax.tick_params(axis='both', labelsize=14)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'performance_heatmap.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved heatmap to {output_path}")
        plt.close()

    def plot_radar_chart(self):
        """Create radar charts comparing models across metrics."""
        if not self.all_results:
            logger.warning("No results available for radar chart")
            return
        
        logger.info("Creating radar charts...")
        
        metrics = ['Accuracy', 'AUC', 'F1Score', 'Sensitivity', 'Specificity', 'MCC']
        
        for method_name, models_data in self.all_results.items():
            if len(models_data) == 0:
                continue
            
            # Number of variables
            num_vars = len(metrics)
            
            # Compute angle for each axis
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            for model_name, model_metrics in models_data.items():
                values = [model_metrics.get(metric, 0) for metric in metrics]
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=model_name, color=MODEL_COLORS.get(model_name, '#CCCCCC'))
                ax.fill(angles, values, alpha=0.25, color=MODEL_COLORS.get(model_name, '#CCCCCC'))
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics, fontsize=11)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
            ax.grid(True, linestyle='--', alpha=0.5)
            
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, framealpha=1, facecolor='white')
            plt.title(f'Model Performance Radar Chart - {method_name}', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()

            output_path = self.output_dir / f'radar_chart_{method_name}.pdf'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved radar chart for {method_name} to {output_path}")
            plt.close()

    def plot_f1_comparison(self):
        """Create a focused bar chart comparing F1-Scores across fine-tuning methods for each model."""
        if not self.all_results:
            logger.warning("No results available for F1 comparison")
            return
        
        logger.info("Creating F1-Score comparison plot...")
        
        # Collect all unique models across all methods
        all_models = set()
        for method_results in self.all_results.values():
            all_models.update(method_results.keys())
        all_models = sorted(list(all_models))
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for grouped bar chart
        x = np.arange(len(all_models))
        width = 0.25  # Width of bars
        
        for i, method in enumerate(self.methods_found):
            method_results = self.all_results[method]
            f1_values = [method_results.get(model, {}).get('F1Score', 0) for model in all_models]
            
            offset = (i - len(self.methods_found)/2 + 0.5) * width
            bars = ax.bar(x + offset, f1_values, width, 
                         label=method, 
                         color=METHOD_COLORS.get(method, '#CCCCCC'),
                         edgecolor='black', linewidth=1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., height,
                           f'{height:.4f}', ha='center', va='bottom', 
                           fontsize=14, rotation=0)
        
        ax.set_ylabel('F1-Score', fontsize=14, fontweight='bold')
        ax.set_xlabel('Model', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(all_models, fontsize=14)
        ax.set_ylim([0, 1.05])
        ax.tick_params(axis='y', labelsize=14)
        ax.legend(title='Fine-tuning Method', loc='lower left', fontsize=14, title_fontsize=14, framealpha=1, facecolor='white')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        output_path = self.output_dir / 'f1_comparison.pdf'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved F1-Score comparison plot to {output_path}")
        plt.close()

    def plot_last_block_metrics(self):
        """Create two bar charts for Last_Block: one with Accuracy and F1-Score, another with Sensitivity and Specificity."""
        # Check if Last_Block results exist
        if 'Last_Block' not in self.all_results:
            logger.warning("Last_Block results not found, skipping Last_Block metrics plot")
            return
        
        logger.info("Creating Last_Block metrics comparison plots...")
        
        last_block_data = self.all_results['Last_Block']
        models = sorted(list(last_block_data.keys()))
        
        # First plot: Accuracy and F1-Score
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        metrics_plot1 = ['Accuracy', 'F1Score']
        labels_plot1 = ['Accuracy', 'F1-Score']
        
        for idx, (metric, label) in enumerate(zip(metrics_plot1, labels_plot1)):
            ax = axes[idx]
            values = [last_block_data[model].get(metric, 0) for model in models]
            model_colors = [MODEL_COLORS.get(m, '#CCCCCC') for m in models]
            
            bars = ax.bar(models, values, color=model_colors, 
                         edgecolor='black', linewidth=1.5, width=1.0)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=14)
            
            ax.set_ylabel(label, fontsize=14, fontweight='bold')
            ax.set_xlabel('Model', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.set_xticklabels(models, fontsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        output_path1 = self.output_dir / 'last_block_accuracy_f1.pdf'
        plt.savefig(output_path1, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Last_Block Accuracy and F1-Score plot to {output_path1}")
        plt.close()
        
        # Second plot: Sensitivity and Specificity
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        metrics_plot2 = ['Sensitivity', 'Specificity']
        labels_plot2 = ['Sensitivity', 'Specificity']
        
        for idx, (metric, label) in enumerate(zip(metrics_plot2, labels_plot2)):
            ax = axes[idx]
            values = [last_block_data[model].get(metric, 0) for model in models]
            model_colors = [MODEL_COLORS.get(m, '#CCCCCC') for m in models]
            
            bars = ax.bar(models, values, color=model_colors, 
                         edgecolor='black', linewidth=1.5, width=1.0)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                       f'{height:.4f}', ha='center', va='bottom', fontsize=14)
            
            ax.set_ylabel(label, fontsize=14, fontweight='bold')
            ax.set_xlabel('Model', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.set_xticklabels(models, fontsize=14)
            ax.tick_params(axis='y', labelsize=14)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        output_path2 = self.output_dir / 'last_block_sensitivity_specificity.pdf'
        plt.savefig(output_path2, dpi=300, bbox_inches='tight')
        logger.info(f"Saved Last_Block Sensitivity and Specificity plot to {output_path2}")
        plt.close()

    def generate_summary_report(self):
        """Generate a text summary report."""
        if not self.all_results:
            logger.warning("No results available for summary report")
            return
        
        logger.info("Generating summary report...")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("IMAGE MODEL EVALUATION RESULTS - SUMMARY REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for method_name in self.methods_found:
            models_data = self.all_results[method_name]
            report_lines.append(f"\n{'='*80}")
            report_lines.append(f"METHOD: {method_name}")
            report_lines.append(f"{'='*80}\n")
            
            for model_name, metrics in models_data.items():
                report_lines.append(f"  Model: {model_name}")
                report_lines.append(f"  {'-'*70}")
                report_lines.append(f"    Accuracy:    {metrics['Accuracy']:.4f}")
                report_lines.append(f"    AUC:         {metrics['AUC']:.4f}")
                report_lines.append(f"    F1-Score:    {metrics['F1Score']:.4f}")
                report_lines.append(f"    Sensitivity: {metrics['Sensitivity']:.4f}")
                report_lines.append(f"    Specificity: {metrics['Specificity']:.4f}")
                report_lines.append(f"    MCC:         {metrics['MCC']:.4f}")
                report_lines.append(f"    Train Size:  {metrics['num_train_images']} images")
                report_lines.append(f"    Test Size:   {metrics['num_test_images']} images")
                report_lines.append("")
        
        # Find best performing model for each metric
        report_lines.append(f"\n{'='*80}")
        report_lines.append("BEST PERFORMING MODELS PER METRIC")
        report_lines.append(f"{'='*80}\n")
        
        metrics = ['Accuracy', 'AUC', 'F1Score', 'Sensitivity', 'Specificity', 'MCC']
        for metric in metrics:
            best_score = 0
            best_model = ""
            best_method = ""
            
            for method_name, models_data in self.all_results.items():
                for model_name, model_metrics in models_data.items():
                    score = model_metrics.get(metric, 0)
                    if score > best_score:
                        best_score = score
                        best_model = model_name
                        best_method = method_name
            
            report_lines.append(f"  {metric:15s}: {best_model:15s} ({best_method:15s}) - {best_score:.4f}")
        
        report_lines.append("\n" + "=" * 80)
        
        # Save report
        report_path = self.output_dir / 'summary_report.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Saved summary report to {report_path}")
        
        # Also print to console
        print('\n'.join(report_lines))

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        logger.info("Starting full analysis pipeline...")
        
        # Load data
        if not self.load_all_results():
            logger.error("Failed to load results. Exiting.")
            return
        
        # Generate outputs
        self.create_comprehensive_csv()
        self.plot_metrics_by_method()
        self.plot_metrics_by_model()
        self.plot_heatmap()
        self.plot_radar_chart()
        self.plot_f1_comparison()
        self.plot_last_block_metrics()
        self.generate_summary_report()
        
        logger.info("=" * 80)
        logger.info("ANALYSIS COMPLETE")
        logger.info(f"All outputs saved to: {self.output_dir}")
        logger.info("=" * 80)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Analyze and visualize Image Model evaluation results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all methods in ImageModels directory
  python plot_image_results.py --results_dir Results/ImageModels
  
  # Analyze a specific method
  python plot_image_results.py --results_dir Results/ImageModels/Last_Block
  
  # Specify custom output directory
  python plot_image_results.py --results_dir Results/ImageModels --output_dir Results/Analysis
        """
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing image model results (can be parent dir or specific method)'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None,
        help='Directory to save output plots and CSV (default: results_dir/analysis)'
    )
    
    args = parser.parse_args()
    
    # Create analyzer and run
    analyzer = ImageResultsAnalyzer(args.results_dir, args.output_dir)
    analyzer.run_full_analysis()


if __name__ == '__main__':
    main()
