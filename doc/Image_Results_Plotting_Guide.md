# Image Model Results Visualization Guide

## Overview
The `plot_image_results.py` script analyzes and visualizes evaluation results from Image Classification models (ResNet, DenseNet, Vision Transformer) across different training methods (Last Layer, Last Block, Full Model).

## Features

### Generated Outputs
1. **CSV Files:**
   - `image_models_results.csv` - Comprehensive results table
   - `comparison_accuracy.csv` - Accuracy comparison pivot table
   - `comparison_auc.csv` - AUC comparison pivot table
   - `comparison_f1-score.csv` - F1-Score comparison pivot table
   - `comparison_sensitivity.csv` - Sensitivity comparison pivot table
   - `comparison_specificity.csv` - Specificity comparison pivot table
   - `comparison_mcc.csv` - MCC comparison pivot table

2. **Visualizations:**
   - `metrics_by_method.png` - Compare all models across methods (grouped bars)
   - `metrics_{method}.png` - Individual plots for each method
   - `performance_heatmap.png` - Heatmap of all metrics
   - `radar_chart_{method}.png` - Radar charts for each method
   - `summary_report.txt` - Text summary with best performers

3. **Metrics Analyzed:**
   - Accuracy
   - AUC (Area Under Curve)
   - F1-Score
   - Sensitivity (Recall/TPR)
   - Specificity (TNR)
   - MCC (Matthews Correlation Coefficient)

## Usage

### Command Line

```bash
# Basic usage - analyze all methods
python plot_image_results.py --results_dir "Results/ImageModels"

# Analyze specific method
python plot_image_results.py --results_dir "Results/ImageModels/Last_Block"

# Specify custom output directory
python plot_image_results.py --results_dir "Results/ImageModels" --output_dir "Results/Analysis"
```

### Using Run Scripts

**Linux/Mac:**
```bash
cd Runscripts
./run_plot_image_results.sh
```

**Windows:**
```batch
cd Runscripts
run_plot_image_results.bat
```

## Examples

### Example 1: Analyze All Methods
Analyzes ResNet, DenseNet, and ViT across Last_Layer, Last_Block, and Model training methods.

```bash
python plot_image_results.py \
    --results_dir "Results/ImageModels" \
    --output_dir "Results/ImageModels/Analysis"
```

**Output:**
- Results/ImageModels/Analysis/image_models_results.csv
- Results/ImageModels/Analysis/metrics_by_method.png
- Results/ImageModels/Analysis/performance_heatmap.png
- Results/ImageModels/Analysis/radar_chart_Last_Block.png
- Results/ImageModels/Analysis/radar_chart_Last_Layer.png
- Results/ImageModels/Analysis/radar_chart_Model.png
- Results/ImageModels/Analysis/summary_report.txt

### Example 2: Analyze Single Method
Analyzes only the Last_Block training method results.

```bash
python plot_image_results.py \
    --results_dir "Results/ImageModels/Last_Block" \
    --output_dir "Results/ImageModels/Last_Block/plots"
```

**Output:**
- Results/ImageModels/Last_Block/plots/image_models_results.csv
- Results/ImageModels/Last_Block/plots/metrics_Last_Block.png
- Results/ImageModels/Last_Block/plots/radar_chart_Last_Block.png
- Results/ImageModels/Last_Block/plots/summary_report.txt

### Example 3: Quick Analysis from Project Root

```bash
# From ml-framework root directory
cd src
python plot_image_results.py --results_dir ../Results/ImageModels
```

## Directory Structure Expected

```
Results/
└── ImageModels/
    ├── Last_Layer/
    │   ├── ImageModels_Last_Layer_image_results.json
    │   └── config.json
    ├── Last_Block/
    │   ├── ImageModels_Last_Block_image_results.json
    │   └── config.json
    └── Model/
        ├── ImageModels_Model_image_results.json
        └── config.json
```

## Output Interpretation

### CSV Results Table
The main CSV contains one row per model-method combination with columns:
- Method: Training method (Last_Layer, Last_Block, Model)
- Model: Model architecture (ResNet, DenseNet, ViT)
- Accuracy, AUC, F1-Score, Sensitivity, Specificity, MCC
- Train_Images: Number of training images
- Test_Images: Number of test images

### Visualizations

**Metrics by Method Plot:**
- Grouped bar chart comparing all models for each training method
- Useful for seeing which method works best for each model

**Individual Method Plots:**
- Bar charts showing model performance within each method
- Color-coded by model type

**Heatmap:**
- Grid view of all metrics across all method-model combinations
- Darker colors indicate better performance

**Radar Charts:**
- Spider/web charts showing model profiles across all metrics
- Easy to see strengths and weaknesses at a glance

**Summary Report:**
- Text file listing best performing model for each metric
- Quick reference for identifying top performers

## Troubleshooting

### No Results Found
**Error:** "No image results JSON files found"
**Solution:** Ensure your results directory contains `*_image_results.json` files

### Missing Metrics
**Error:** Some metrics show 0.0000
**Solution:** Check that your JSON files contain the full evaluation structure with all metrics

### Import Errors
**Error:** "ModuleNotFoundError: No module named 'matplotlib'"
**Solution:** Install required packages:
```bash
pip install matplotlib seaborn pandas numpy
```

## Advanced Usage

### Filtering Specific Models
Edit the script to filter models in the `_extract_metrics_from_json` method.

### Customizing Colors
Modify the `MODEL_COLORS` and `METHOD_COLORS` dictionaries at the top of the script.

### Adding New Metrics
Add metric names to `metrics_to_plot` lists in plotting functions and ensure they exist in the JSON structure.

## Requirements

- Python 3.7+
- matplotlib
- seaborn
- pandas
- numpy

Install with:
```bash
pip install matplotlib seaborn pandas numpy
```
