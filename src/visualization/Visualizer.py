import math
from copy import deepcopy
from re import split

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import SubplotSpec
import seaborn as sns

from metrics import ExperimentResult


class ResultVisualizer:

    def __init__(self, results: ExperimentResult, config):
        self.full_results = results
        self.result_name = results.result_name
        self.prepared_data = {}
        self.visualisation_settings = config["model_result_settings"]
        self.comparison_settings = config["model_comparison_settings"]
        self.active_setting = None


    def visualize_results(self):

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


                n_rows,  n_optimizers, plot_index_type_dict, roc_plot_index_type_dict, optimizer_palette, model_metrics, \
                 bar_chart_metric_list, eval_types, fig_title, calculate_roc = self._setup_model_plot(setting_name, model_metrics)
                self._visualize_model(n_rows,  n_optimizers, plot_index_type_dict, roc_plot_index_type_dict,optimizer_palette,model_metrics, \
                 bar_chart_metric_list, fig_title, calculate_roc)





    def _visualize_model(self, n_rows: int, n_optimizers: int, metric_plot_index_type_dict: dict, roc_plot_index_type_dict: dict, optimizer_palette,
                         model_metrics: pd.DataFrame, bar_chart_metric_list: list,  fig_title: str, calculate_roc: bool):

        columns_metric = bar_chart_metric_list
        columns_roc = ["ROC"]

        self._create_charts(model_metrics, self._plot_metric_bar_chart, self._plot_metric_bar_chart_cv, n_rows, metric_plot_index_type_dict, optimizer_palette, fig_title, n_optimizers, columns_metric)


        if calculate_roc:
            self._create_charts(model_metrics, self._plot_roc_curve, self._plot_roc_curve_cv, n_rows, roc_plot_index_type_dict, optimizer_palette, fig_title, n_optimizers, columns_roc)

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

    def visualize_comparison(self):
        pass


    @staticmethod
    def _plot_metric_bar_chart(ax, data, palette, plot_name: str):
        df_melted = data.melt(id_vars="optimizer", var_name="Metric", value_name="Value")
        ax.set_title(f"Metrics by Threshold Optimizer for {plot_name}")
        sns.barplot(x="Metric", y="Value", hue="optimizer", data=df_melted, ax=ax, palette=palette)
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
    def _plot_roc_curve(ax, data: pd.DataFrame, color_mapping, plot_name: str):
        if not "crossvalidation" in plot_name.lower():
            data["FPR"] = data["ROC"].apply(lambda x: x[0])
            data["TPR"] = data["ROC"].apply(lambda x: x[1])
            data.drop(columns=["ROC"], inplace=True)
            data.reset_index(drop=True, inplace=True)
            optimizer_list = []
            fpr_list = []
            tpr_list = []
            for index in range(len(data.index)):
                fpr = data["FPR"][index]
                tpr = data["TPR"][index]
                optimizer_list += [data["optimizer"][index]] * len(fpr)
                fpr_list += fpr
                tpr_list += tpr

            roc_data = pd.DataFrame({"FPR": fpr_list, "TPR": tpr_list, "optimizer": optimizer_list})
            sns.lineplot(x="FPR", y="TPR", hue="optimizer", data=roc_data, ax=ax, palette=color_mapping)
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Coin Toss')

        else:
            ax.text(0.5, 0.5, "Not applicable", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve by Threshold Optimizer for " + plot_name)
        return ax

    @staticmethod
    def _plot_roc_curve_cv(ax, data: pd.DataFrame, color_mapping, plot_name: str):
        data["FPR"] = data["ROC"].apply(lambda x: x[0])
        data["TPR"] = data["ROC"].apply(lambda x: x[1])
        data.drop(columns=["ROC"], inplace=True)
        data.reset_index(drop=True, inplace=True)
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
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title("ROC Curve by Threshold Optimizer for " + plot_name)
        pass


    def _setup_model_plot(self, setting, model_metrics: pd.DataFrame):
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


        optimizer_palette = sns.color_palette("husl", n_optimizers)
        return  n_rows, n_optimizers, metric_plot_index_type_dict, roc_plot_index_type_dict, \
            optimizer_palette,model_metrics, bar_chart_metric_list, eval_types, fig_title, calculate_roc

    @staticmethod
    def _create_charts( data: pd.DataFrame, std_func, cv_func, n_rows:int, plot_name_type_dict: dict,
                       optimizer_palette, fig_title: str, n_optimizers: int, chart_columns: list):

        fig, axs = plt.subplots(n_rows, 2, figsize=(23.4, 33.1))

        plot_names = list(plot_name_type_dict.keys())

        for i, ax in enumerate(axs.flatten()):
            plot_name = plot_names[i]
            plot_type = plot_name_type_dict[plot_name]
            data_start = i * n_optimizers
            data_end = (i + 1) * n_optimizers

            if "CV" in plot_type:
                cv_columns = chart_columns + ["split"]
                chart_data = data[data_start:data_end][cv_columns]
                ax = cv_func(ax, chart_data, optimizer_palette, plot_name)
                pass
            else:
                standard_columns = chart_columns +  ["optimizer"]
                chart_data = data[data_start:data_end][standard_columns]

                ax = std_func(ax, chart_data, optimizer_palette, plot_name)


        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.set_facecolor('w')
        fig.suptitle(fig_title, fontsize=16)
        fig.subplots_adjust(top=0.95)
        plt.show()






