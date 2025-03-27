import math
from copy import deepcopy
from re import split
from symbol import comparison

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import SubplotSpec
import seaborn as sns

from metrics import ExperimentResult

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# TODO change TPR and FPR to TPRs and FPRs and adjusted to these metrics being stored on evaluation level

class ResultVisualizer:

    def __init__(self, results: ExperimentResult, config, result_path):
        self.full_results = results
        self.result_name = results.result_name
        self.prepared_data = {}
        self.visualisation_settings = config["model_result_settings"]
        self.comparison_settings = config["model_comparison_settings"]
        self.active_setting = None
        self.result_path = result_path.replace("results", "")


    def visualize_results(self):
        self._visualize_metrics()
        self._visualize_comparison()

    def _visualize_metrics(self):

        self.result_path = self.result_path + "visualisation"
        contained_models = list(self.full_results.contained_model_results.keys())
        for setting_name in self.visualisation_settings.keys():
            needed_models = self.visualisation_settings[setting_name]["active_models"]
            present_models = []

            if "all" in needed_models:
                present_models = contained_models

            else:
                for model in needed_models:
                    if model in contained_models:
                        present_models.append(model)
                    else:
                        print(f"No information for {model} available. Skipping...")
            self.active_setting = self.visualisation_settings[setting_name]
            print(f"Visualizing {setting_name}...")
            self.active_setting["active_models"] = present_models
            metric_df = self._prepare_visualize_experiment()
            #metric_df = metric_df.sort_values(by=["model", "eval_type", "split"]).reset_index(drop=True)

            for model in present_models:
                model_metrics = metric_df[metric_df["model"] == model].reset_index(drop=True)
                self.result_path = f"{self.result_path}_{model}"


                n_rows,  n_optimizers, plot_index_type_dict, roc_plot_index_type_dict, optimizer_palette, model_metrics, \
                 bar_chart_metric_list, eval_types, fig_title, calculate_roc = self._setup_model_plot_visualisation(setting_name, model_metrics)
                self._visualize_model(n_rows,  n_optimizers, plot_index_type_dict, roc_plot_index_type_dict,optimizer_palette,model_metrics, \
                 bar_chart_metric_list, fig_title, calculate_roc)
                self.result_path = self.result_path.replace(f"_{model}", "")
        self.result_path = self.result_path.replace("visualisation", "")


    def _visualize_comparison(self):
        self.result_path = self.result_path + "comparisons"
        contained_models = list(self.full_results.contained_model_results.keys())
        supported_eval_types = ["Training", "Evaluation", "TrainingCrossValidation", "EvaluationCrossValidation"]
        for setting_name in self.comparison_settings.keys():
            setting = self.comparison_settings[setting_name]
            self.result_path = self.result_path + f"_{setting_name}"
            comparisons = setting["comparisons"]
            eval_type = setting["eval_type_to_compare"]

            if eval_type not in supported_eval_types:
                print(f"Comparison for {eval_type} not supported. Skipping...")
                continue
            metrics_requested = setting["metrics_to_compare"]

            needed_models = {}
            for given_comparison in comparisons:
                model_name = given_comparison.split(", ")[0].strip()
                optimizer_name = given_comparison.split(", ")[1].strip()
                if model_name == "all":
                    needed_models = { model: [optimizer_name] for model in contained_models}
                    break
                elif model_name in contained_models:
                    if model_name in needed_models.keys():
                        needed_models[model_name].append(optimizer_name)
                    else:
                        needed_models[model_name] = [optimizer_name]
                else:
                    print(f"No information for {model_name} available. Skipping...")

            metric_dict = None

            metric_dict, needed_metrics = self._prepare_visualize_comparisons(needed_models, eval_type, metrics_requested, metric_dict)
            metric_df = pd.DataFrame(metric_dict)



            self._create_comparison_charts(metric_df, setting_name, needed_metrics )
            self.result_path = self.result_path.replace(f"_{setting_name}", "")
        self.result_path = self.result_path.replace("comparisons", "")
    def _prepare_visualize_comparisons(self, models: dict, eval_type: str, metrics_requested: list, metric_dict: dict = None) -> (dict, list):
        if "crossvalidation" in eval_type.lower():
            split_name = "mean"
        else:
            split_name = eval_type + " split"
        available_metrics = set()
        not_in_all = []
        for model in models.keys():
            for optimizer in models[model]:
                model_metrics = list(self.full_results.contained_model_results[model].contained_evals[eval_type].contained_optimizers[optimizer].contained_splits[split_name].contained_metrics.keys())
                for metric in model_metrics:
                    if metric not in not_in_all:
                        available_metrics.add(metric)
                for metric in list(available_metrics):
                    if metric not in model_metrics:
                        available_metrics.remove(metric)
                        not_in_all.append(metric)

        needed_metrics = []
        if "all" in metrics_requested:
            needed_metrics = list(available_metrics)
            if "FPR" in needed_metrics and "TPR" in needed_metrics:
                needed_metrics.append("ROC")
                needed_metrics.remove("FPR")
                needed_metrics.remove("TPR")
        else:
            for metric in metrics_requested:
                if metric not in available_metrics:
                    print(f"No information for {metric} available. Skipping...")
                else:
                    needed_metrics.append(metric)
            if "FPR" in needed_metrics and "TPR" in needed_metrics:
                needed_metrics.remove("FPR")
                needed_metrics.remove("TPR")
                needed_metrics.append("ROC")


        for model in models.keys():
            for optimizer in models[model]:
                metric_dict = self._prepare_comparison_single_split(model, optimizer, eval_type, split_name, needed_metrics, metric_dict)
        return metric_dict, needed_metrics

    def _prepare_comparison_single_split(self, model: str, optimizer: str, eval_type: str, needed_split: str, needed_metrics: list, metric_dict: dict = None) -> dict:
        metrics = self.full_results.contained_model_results[model].contained_evals[eval_type].contained_optimizers[optimizer].contained_splits[needed_split].contained_metrics

        if not metric_dict:
            metric_dict = {"name": [model + " " + optimizer]}
            for metric in needed_metrics:
                if metric == "ROC":
                    metric_dict[metric] = [(metrics["FPR"].metric_value.metric_value, metrics["TPR"].metric_value.metric_value)]
                else:
                    metric_dict[metric] = [metrics[metric].metric_value.metric_value]
        else:
            metric_dict["name"].append(model + " " + optimizer)


            for metric in needed_metrics:
                if metric == "ROC":
                    metric_dict[metric].append((metrics["FPR"].metric_value.metric_value, metrics["TPR"].metric_value.metric_value))
                else:
                    metric_dict[metric].append(metrics[metric].metric_value.metric_value)
        return metric_dict

    def _visualize_model(self, n_rows: int, n_optimizers: int, metric_plot_index_type_dict: dict, roc_plot_index_type_dict: dict, optimizer_palette,
                         model_metrics: pd.DataFrame, bar_chart_metric_list: list,  fig_title: str, calculate_roc: bool):

        columns_metric = bar_chart_metric_list
        columns_roc = ["ROC"]

        self._create_visualisation_charts(model_metrics, n_rows, metric_plot_index_type_dict, optimizer_palette, fig_title,
                                          n_optimizers, columns_metric, "Metric")


        if calculate_roc:
            self._create_visualisation_charts(model_metrics, n_rows, roc_plot_index_type_dict, optimizer_palette, fig_title,
                                              n_optimizers, columns_roc, "ROC")

    def _prepare_visualize_experiment(self) -> pd.DataFrame:
        metric_dict = None
        for model in self.active_setting["active_models"]:
            metric_dict = self._prepare_visualize_model( model, metric_dict)
        metric_df = pd.DataFrame(metric_dict)
        return metric_df

    def _prepare_visualize_model(self,   model: str, metric_dict: dict) -> dict:
        needed_evals = []


        if "all" in self.active_setting["active_evals"]:
            needed_evals = list(self.full_results.contained_model_results[model].contained_evals.keys())
        else:
            for exp_eval in self.active_setting["active_evals"]:
                if exp_eval in self.full_results.contained_model_results[model].contained_evals.keys():
                    needed_evals.append(exp_eval)
                else:
                    print(f"No information for {exp_eval} available. Skipping...")

        for eval_type in needed_evals:

            metric_dict = self._prepare_visualize_optimizer( model, eval_type,  metric_dict)
        self.active_setting["active_evals"] = needed_evals
        return metric_dict

    def _prepare_visualize_optimizer(self,  model: str, eval_type: str,  metric_dict: dict) -> dict:
        needed_optimizers = []
        if "all" in self.active_setting["active_optimizers"]:
            needed_optimizers = list(self.full_results.contained_model_results[model].contained_evals[eval_type].contained_optimizers.keys())
        else:
            for optimizer in self.active_setting["active_optimizers"]:

                if optimizer in self.full_results.contained_model_results[model].contained_evals[eval_type].contained_optimizers.keys():
                    needed_optimizers.append(optimizer)
                else:
                    print(f"No information for {optimizer} available. Skipping...")

        for optimizer in needed_optimizers:


            if "crossvalidation" in eval_type.lower():
                split = list(self.full_results.contained_model_results[model].contained_evals[eval_type].contained_optimizers[optimizer].contained_splits.keys())
            else:
                split = eval_type + " split"
            if type(split) == list:
                for split_name in split:
                    metric_dict = self._prepare_visualize_single_split(model, eval_type, optimizer, split_name, metric_dict)

            elif type(split) == str:
                metric_dict = self._prepare_visualize_single_split(model, eval_type, optimizer, split, metric_dict)
            else:
                print(f"Split type {split} not supported. Skipping...")
        self.active_setting["active_optimizers"] = needed_optimizers
        return metric_dict

    def _prepare_visualize_single_split(self, model: str, eval_type: str, optimizer: str, split: str, metric_dict: dict = None) -> dict:
        active_metrics = self.active_setting["active_metrics"]
        metric_data = self.full_results.contained_model_results[model].contained_evals[eval_type].contained_optimizers[
            optimizer].contained_splits[split].contained_metrics
        available_metrics = list(metric_data.keys())
        metrics_to_visualize = []
        if "all" in active_metrics:
            metrics_to_visualize = available_metrics
            if "FPR" in metrics_to_visualize and "TPR" in metrics_to_visualize:
                metrics_to_visualize.append("ROC")
                metrics_to_visualize.remove("FPR")
                metrics_to_visualize.remove("TPR")
        else:
            for metric in active_metrics:
                if metric in available_metrics or (metric == "ROC" and "FPR" in available_metrics and "TPR" in available_metrics):
                    metrics_to_visualize.append(metric)
            if "FPR" in metrics_to_visualize and "TPR" in metrics_to_visualize:
                metrics_to_visualize.remove("FPR")
                metrics_to_visualize.remove("TPR")
                metrics_to_visualize.append("ROC")
        if not metric_dict:
            metric_dict = {"model": [model], "eval_type": [eval_type], "optimizer": [optimizer], "split": [split]}
            for metric in metrics_to_visualize:
                if metric == "ROC":
                    metric_dict[metric] = [(metric_data["FPR"].metric_value.metric_value, metric_data["TPR"].metric_value.metric_value)]
                else:
                    metric_dict[metric] = [metric_data[metric].metric_value.metric_value]
        else:
            metric_dict["model"].append(model)
            metric_dict["optimizer"].append(optimizer)
            metric_dict["split"].append(split)
            metric_dict["eval_type"].append(eval_type)
            for metric in metrics_to_visualize:
                if metric == "ROC":
                    metric_dict[metric].append((metric_data["FPR"].metric_value.metric_value, metric_data["TPR"].metric_value.metric_value))
                else:
                    metric_dict[metric].append(metric_data[metric].metric_value.metric_value)
        self.active_setting["active_metrics"] = metrics_to_visualize


        return metric_dict

    @staticmethod
    def _plot_metric_bar_chart(ax, data, palette, subfigtitle: str, hue: str):
        df_melted = data.melt(id_vars=hue, var_name="Metric", value_name="Value")
        ax.set_title(subfigtitle)
        sns.barplot(x="Metric", y="Value", hue=hue, data=df_melted, ax=ax, palette=palette)
        for bars in ax.containers:
            labels = [f'{v:.3f}' for v in bars.datavalues]
            ax.bar_label(bars, labels=labels, rotation=90, label_type='center')
        return ax

    @staticmethod
    def _plot_metric_bar_chart_cv(ax, data, palette, plot_name: str):
        df_melted = data.melt(id_vars="split", var_name="Metric", value_name="Value")
        ax.set_title(f"Metrics by Threshold Optimizer for {plot_name}")
        sns.barplot(x="Metric", y="Value", hue="split", data=df_melted, ax=ax, palette=palette)
        for bars in ax.containers:
            labels = [f'{v:.3f}' for v in bars.datavalues]
            ax.bar_label(bars, labels=labels, rotation=90, label_type='center')
        return ax

    @staticmethod
    def _plot_roc_curve(ax, data: pd.DataFrame, color_mapping, hue: str, subfigtitle: str):

        data["FPR"] = data["ROC"].apply(lambda x: x[0])
        data["TPR"] = data["ROC"].apply(lambda x: x[1])
        data.drop(columns=["ROC"], inplace=True)
        data.reset_index(drop=True, inplace=True)
        hue_list = []
        fpr_list = []
        tpr_list = []
        for index in range(len(data.index)):
            fpr = data["FPR"][index]
            tpr = data["TPR"][index]
            hue_list += [data[hue][index]] * len(fpr)
            fpr_list += fpr
            tpr_list += tpr

        roc_data = pd.DataFrame({"FPR": fpr_list, "TPR": tpr_list, hue: hue_list})
        sns.lineplot(x="FPR", y="TPR", hue=hue, data=roc_data, ax=ax, palette=color_mapping)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Coin Toss')

        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(subfigtitle)
        return ax

    @staticmethod
    def _plot_roc_curve_cv(ax, data: pd.DataFrame, color_mapping, plot_name: str):
        data["FPR"] = data["ROC"].apply(lambda x: x[0])
        data["TPR"] = data["ROC"].apply(lambda x: x[1])
        data.drop(columns=["ROC"], inplace=True)
        data.reset_index(drop=True, inplace=True)
        mean_fpr = data[data["split"] == "mean"]["FPR"].values[0]
        mean_tpr = data[data["split"] == "mean"]["TPR"].values[0]
        std_tpr = np.std(mean_tpr, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        split_list = []
        fpr_list = []
        tpr_list = []
        for index in range(len(data.index)):
            fpr = data["FPR"][index]
            tpr = data["TPR"][index]
            split_list += [data["split"][index]] * len(fpr)
            fpr_list += fpr
            tpr_list += tpr

        roc_data = pd.DataFrame({"FPR": fpr_list, "TPR": tpr_list, "split": split_list})
        sns.lineplot(x="FPR", y="TPR", hue="split", data=roc_data, ax=ax, palette=color_mapping)
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Coin Toss')
        #TODO: Check if label can be shown in plot
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.2, label="Mean +/- 1 std")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve by Threshold Optimizer for " + plot_name)
        return ax

    def _setup_model_plot_visualisation(self, setting, model_metrics: pd.DataFrame):
        model_name = model_metrics["model"].unique()[0]
        fig_title = self.result_name + "\n Evaluation for " + model_name + "\n with " + "Setting " + setting

        calculate_roc = "ROC" in model_metrics.columns
        if calculate_roc:
            bar_chart_metric_list = deepcopy(self.active_setting["active_metrics"])
            bar_chart_metric_list.remove("ROC")
        else:
            bar_chart_metric_list = self.active_setting["active_metrics"]


        eval_types = model_metrics["eval_type"].unique()
        n_eval_types = len(eval_types)
        optimizers = model_metrics["optimizer"].unique()
        n_optimizers = len(optimizers)
        splits = model_metrics["split"].unique()
        n_max_splits = 0
        for model_split in splits:
            if "crossvalidation" in model_split.lower():
                n_max_splits += 0.5
            if model_split == "mean":
                n_max_splits += 1
        n_max_splits = int(n_max_splits)


        compare_cv = self.active_setting["plot_cv_comparison"]
        crossvalidation_count = 0
        num_plots = n_eval_types
        if compare_cv:
            to_remove = []
            to_add = []
            for eval_type in eval_types:
                if "crossvalidation" in eval_type.lower():
                    crossvalidation_count += 1
                    to_remove.append(eval_type)
                    to_add += [f"{eval_type} {optimizer}" for optimizer in optimizers]
            eval_types = [eval_type for eval_type in eval_types if eval_type not in to_remove]
            eval_types += to_add

            temp_df_mean = model_metrics[model_metrics["split"].str.contains("mean", case=False)]
            temp_df_splits = model_metrics[model_metrics["split"].str.contains("crossvalidation", case=False)]

            rows_to_remove = temp_df_mean.index.to_list() + temp_df_splits.index.to_list()
            model_metrics = model_metrics.drop(rows_to_remove)
            model_metrics = pd.concat([model_metrics, temp_df_mean, temp_df_splits]).reset_index(drop=True)
            num_plots += crossvalidation_count*n_optimizers

        n_rows = math.ceil(num_plots / 2)
        metric_plot_index_type_dict = {}
        roc_plot_index_type_dict = {}
        for i in range(num_plots):
            eval_type = model_metrics["eval_type"][i*4]
            optimizer = model_metrics["optimizer"][i*4]
            split = model_metrics["split"][i*4]
            if "crossvalidation" in eval_type.lower() and split != "mean":
                plot_name = f"{eval_type} with {optimizer} optimizer"
                plot_type = "CV"
                metric_plot_index_type_dict[plot_name] = plot_type
            else:
                plot_name = eval_type
                plot_type = "Standard"
                metric_plot_index_type_dict[plot_name] = plot_type
            if calculate_roc:
                plot_type += " ROC"
                roc_plot_index_type_dict[plot_name] = plot_type


        optimizer_palette = sns.color_palette("husl", max(n_optimizers, n_max_splits))
        return  n_rows, n_optimizers, metric_plot_index_type_dict, roc_plot_index_type_dict, \
            optimizer_palette,model_metrics, bar_chart_metric_list, eval_types, fig_title, calculate_roc

    def _create_visualisation_charts(self, data: pd.DataFrame, n_rows: int, plot_name_type_dict: dict, optimizer_palette,
                                     fig_title: str, n_optimizers: int, chart_columns: list, chart_type: str):

        fig, axs = plt.subplots(n_rows, 2, figsize=(23.4, 33.1))

        plot_names = list(plot_name_type_dict.keys())

        for i, ax in enumerate(axs.flatten()):
            plot_name = plot_names[i]
            plot_type = plot_name_type_dict[plot_name]
            data_start = i * n_optimizers
            data_end = (i + 1) * n_optimizers

            if "CV" in plot_type:
                cv_columns = chart_columns + ["split"]
                chart_data = data[data_start:data_end]
                chart_eval_type = chart_data["eval_type"].unique()[0]
                chart_optimizer = chart_data["optimizer"].unique()[0]
                mean_row = data[(data['eval_type'] == chart_eval_type) & (data['optimizer'] == chart_optimizer) & (data['split'] == "mean")]
                chart_data = pd.concat([chart_data, mean_row], ignore_index=True)[cv_columns]
                if chart_type == "ROC":

                    ax = self._plot_roc_curve_cv(ax, chart_data, optimizer_palette, plot_name)
                elif chart_type == "Metric":
                    ax = self._plot_metric_bar_chart_cv(ax, chart_data, optimizer_palette, plot_name)

            else:
                standard_columns = chart_columns +  ["optimizer"]
                chart_data = data[data_start:data_end][standard_columns]
                if chart_type == "ROC":
                    sub_figtitle = f"ROC Curve by Threshold Optimizer for {plot_name}"
                    ax = self._plot_roc_curve(ax, chart_data, optimizer_palette,  "optimizer", sub_figtitle)
                elif chart_type == "Metric":
                    sub_figtitle = f"Metrics by Threshold Optimizer for {plot_name}"
                    ax = self._plot_metric_bar_chart(ax, chart_data, optimizer_palette, sub_figtitle, "optimizer")







        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.set_facecolor('w')
        fig.suptitle(fig_title, fontsize=16)
        fig.subplots_adjust(top=0.95)
        #plt.savefig(f"{self.result_path}_{chart_type}.svg")
        plt.savefig(f"{self.result_path}_{chart_type}.pdf")

    def _create_comparison_charts(self, data: pd.DataFrame, comparison_name: str, needed_metrics: list):
        n_compared_models = len(data["name"].unique())
        n_metrics = len(needed_metrics)
        comparison_color_palette = sns.color_palette("husl", n_compared_models)
        n_rows = math.ceil(n_metrics / 2)
        fig_title = self.result_name + "\n Comparison with setting \n" + comparison_name
        fig, axs = plt.subplots(2, 1, figsize=(23.4, 33.1))
        plot_name = "Comparison of bar chart metrics"
        bar_chart_metric_list = []
        roc_present = False
        if "ROC" in needed_metrics:
            bar_chart_metric_list = [metric for metric in needed_metrics if metric != "ROC"]
            roc_present = True
        else:
            bar_chart_metric_list = needed_metrics
        if len(bar_chart_metric_list) > 0:
            axs[0] = self._plot_metric_bar_chart(axs[0], data[["name"] + bar_chart_metric_list], comparison_color_palette, plot_name, "name" )
        if roc_present:
            axs[1] = self._plot_roc_curve(axs[1], data, comparison_color_palette,  "name", comparison_name)

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.set_facecolor('w')
        fig.suptitle(fig_title, fontsize=16)
        fig.subplots_adjust(top=0.95)
        # plt.savefig(f"{self.result_path}svg")
        plt.savefig(f"{self.result_path}.pdf")











