import json
import os
import sys
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np
from metrics.Models import EvalResult


def load_json(file_path: str):
    with open(file_path, 'r') as file:
        data = json.load(file)
        model_count = len(data)
    return data, model_count


def plot_data(data: dict, model_count: int) -> None:

    # create subplots
    fix, axs = plt.subplots(model_count + 1, 2, figsize=(10, 5 * model_count))
    plt.subplots_adjust(hspace=0.35, wspace=0.3)

    for index, model_name in enumerate(data):
        axs[index, 0].set_title(f"{model_name} ROC Curve")
    axs[model_count, 0].set_title("Metric Comparison")

    # Plot the ROC curve for each model
    for row_index, model_name in enumerate(data):
        model = data[model_name]
        plot_model_data(model, model_name, row_index, axs)

    plot_model_comparison(data, row_index + 1, axs)


def plot_model_data(model: dict, model_name: str, plot_row: int, axs) -> None:
    fprs: list = model["cross_validation"]["fprs"]
    tprs: list = model["cross_validation"]["tprs"]

    # Set the labels and limits
    axs[plot_row, 0].set(xlabel="False Positive Rate", ylabel="True Positive Rate")
    axs[plot_row, 0].set_xlim([-0.01, 1.01])
    axs[plot_row, 0].set_ylim([-0.01, 1.01])
    axs[plot_row, 0].plot([0, 1], [0, 1], linestyle="--", color="black")

    # Plot the ROC curve for each fold
    for fold, (tpr, fpr) in enumerate(zip(tprs, fprs)):
        axs[plot_row, 0].plot(fpr, tpr, label=f"Fold {fold + 1}, AUC = {metrics.auc(fpr, tpr):.2f}")

    axs[plot_row, 0].legend(loc="lower right")


def plot_model_comparison(data: dict, plot_row: int, axs) -> None:
    bar_chart_width = 0.25
    bar_chart_multipliers = 0
    bar_chart_x = np.arange(5)

    for model_name in data:
        model = data[model_name]
        fold_count = len(model["cross_validation"]["fprs"])

        # ----- Plot metric comparison ----- #
        axs[plot_row, 0].bar(
            bar_chart_x + bar_chart_multipliers * bar_chart_width,
            [model["cross_validation"][metric] for metric in ("acc", "sens", "spec", "f1", "mcc")],
            label=model_name,
            width=bar_chart_width,
        )
        bar_chart_multipliers += 1

        axs[plot_row, 0].legend()
        axs[plot_row, 0].set_xticks(bar_chart_x + bar_chart_width / 2,
                                    ["Accuracy", "Sensitivity", "Specificity", "F1", "MCC"])


def plot_eval(data: EvalResult, file_name: str) -> None:
    # Lädt file soll aber bereits daten entgegen nehmen

    model_count = len(data)
    plot_data(data, model_count)
    plt.savefig(file_name + ".pdf", bbox_inches='tight', format="pdf")

    #hallo
