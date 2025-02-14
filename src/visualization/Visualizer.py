from copy import deepcopy

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



                self._visualize_model(setting_name, model_metrics)





    def _visualize_model(self, setting, model_metrics: pd.DataFrame):
        model_name = model_metrics["model"].unique()[0]
        fig_title = self.result_name + "\n Evaluation for " + model_name + "\n with " + "Setting " + setting


        calculate_roc = "ROC" in model_metrics.columns
        if calculate_roc:
            num_cols = 2
            bar_chart_metric_list = deepcopy(self.active_setting["active_metrics"])
            bar_chart_metric_list.remove("ROC")
        else:
            bar_chart_metric_list = self.active_setting["active_metrics"]
            num_cols = 1

        eval_types = model_metrics["eval_type"].unique()
        n_eval_types = len(eval_types)
        optimizers = model_metrics["optimizer"].unique()
        n_optimizers = len(optimizers)

        compare_cv = self.active_setting["plot_cv_comparison"]
        crossvalidation_count = 0
        num_rows = n_eval_types
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
            num_rows += crossvalidation_count*n_optimizers

        num_plots = num_rows * num_cols
        plot_index_type_dict = {}
        for i in range(num_plots):
            plot_type = None
            if calculate_roc and i % 2 == 1:
                plot_type = "ROC"
            elif i< n_eval_types:
                plot_type = "metric"

            if compare_cv and i >= n_eval_types and i % 2 == 1 and calculate_roc:
               plot_type = "ROC CV"
            elif compare_cv and i >= n_eval_types:
                plot_type = "metric CV"

            plot_index_type_dict[i] = plot_type

        fig, axs = plt.subplots(num_rows, num_cols, figsize=(100, 100) )
        optimizer_palette = sns.color_palette("husl", n_optimizers)
        matplotlib_optimizer_colors = [color for color in optimizer_palette.as_hex()]
        optimizer_color_mapping = {optimizer: color for optimizer, color in zip(optimizers, matplotlib_optimizer_colors)}
        for i, ax in enumerate(axs.flatten()):
            plot_type = plot_index_type_dict[i]
            if "CV" in plot_type:
                #TODO
                pass
            else:
                if plot_type == "ROC":
                    roc_data = model_metrics[i*n_optimizers:(i+1)*n_optimizers][["ROC", "optimizer"]]
                    ax = self._plot_roc_curve(ax, roc_data, optimizer_palette)
                    pass
                elif plot_type == "metric":
                    chart_data = model_metrics[i*n_optimizers:(i+1)*n_optimizers][bar_chart_metric_list + ["optimizer"]]

                    ax = self._plot_metric_bar_chart(ax, chart_data, optimizer_palette)
        grid = plt.GridSpec(num_rows, num_cols)
        for idx, subtitle in enumerate(eval_types):
            self._create_subtitle(fig, grid[idx, ::], subtitle)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.set_facecolor('w')
        fig.suptitle(fig_title, fontsize=16)
        fig.subplots_adjust(top=0.8)
        plt.show()
















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





    def _visualize_cross_validation(self):
        pass


    def visualize_comparison(self):
        pass

    def _visualize_cv_ROC(self):
        pass

    def _visualize_multiple_ROC(self):
        pass

    def _create_subtitle(self, fig: plt.Figure, grid: SubplotSpec, title: str):
        "Sign sets of subplots with title"
        row = fig.add_subplot(grid)
        # the '\n' is important
        row.set_title(f'{title}\n', fontweight='semibold')
        # hide subplot
        row.set_frame_on(False)
        row.axis('off')

    def _plot_metric_bar_chart(self, ax, data, palette):
        df_melted = data.melt(id_vars="optimizer", var_name="Metric", value_name="Value")
        ax.set_title("Metrics by Threshold Optimizer")
        sns.barplot(x="Metric", y="Value", hue="optimizer", data=df_melted, ax=ax, palette=palette)
        return ax

    def _plot_roc_curve(self, ax, data, color_mapping):
        data["FPR"] = data["ROC"].apply(lambda x: x[0])
        data["TPR"] = data["ROC"].apply(lambda x: x[1])
        data.drop(columns=["ROC"], inplace=True)
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



        ax.plot([0,1], [0,1], linestyle='--', color='gray', label='Coin Toss')
        ax.set_title("ROC Curve by Threshold Optimizer")
        return ax






