from typing import Union

import pandas as pd
from sympy import false

from metrics import ExperimentResult


class Visualisizer:

    def __init__(self, results: ExperimentResult, config):
        self.full_results = results
        self.result_name = results["result_name"]
        self.prepared_data = {}
        self.visualisation_settings = config["model_result_settings"]
        self.comparison_settings = config["model_comparison_settings"]
        self.active_setting = None


    def visualize_results(self):

        contained_models = self.full_results.contained_model_results.keys()
        for setting in self.visualisation_settings.keys():
            needed_models = self.visualisation_settings[setting]["active_models"]
            present_models = []

            if "all" in needed_models:
                present_models = contained_models

            else:
                for model in needed_models:
                    if model in contained_models:
                        present_models.append(model)
                    else:
                        print(f"No information for {model} available. Skipping...")


    def _visualize_model(self, setting, present_models: list):

        pass

    def _prepare_visualize_model(self, setting,  model: str):
        needed_evals = []
        if "all" in setting["active_evals"]:
            needed_evals = self.full_results.contained_model_results[model].contained_evals.keys()
        else:
            for exp_eval in setting["active_evals"]:
                if exp_eval in self.full_results.contained_model_results[model].contained_evals.keys():
                    needed_evals.append(exp_eval)
                else:
                    print(f"No information for {exp_eval} available. Skipping...")



    def _prepare_visualize_optimizer(self, setting, model: str, eval_type: str, split: Union[str, list[str]]):
        needed_optimizers = []
        if "all" in setting["active_optimizers"]:
            needed_optimizers = self.full_results.contained_model_results[model].contained_evals[eval_type].contained_optimizers.keys()
        else:
            for optimizer in setting["active_optimizers"]:
                if optimizer in self.full_results.contained_model_results[model].contained_evals[eval_type].contained_optimizers.keys():
                    needed_optimizers.append(optimizer)
                else:
                    print(f"No information for {optimizer} available. Skipping...")

        for optimizer in needed_optimizers:
            if type(split) == list:
                # TODO
                pass
            elif type(split) == str:
                self._prepare_visualize_single_split(model, eval_type, optimizer, split)
            else:
                print(f"Split type {type(split)} not supported. Skipping...")





    def _prepare_visualize_single_split(self, model: str, eval_type: str, optimizer: str, split: str, metric_dict: dict = None):
        active_metrics = self.active_setting["active_metrics"]
        metric_data = self.full_results.contained_model_results[model].contained_evals[eval_type].contained_optimizers[
            optimizer].contained_splits[split].contained_metrics
        available_metrics = list(metric_data.keys())
        metrics_to_visualize = []
        if "all" in active_metrics:
            metrics_to_visualize = available_metrics
        else:
            for metric in active_metrics:
                if metric in available_metrics:
                    metrics_to_visualize.append(metric)

        if not metric_dict:
            metric_dict = {model: [model], eval_type: [eval_type], optimizer: [optimizer], split: [split]}
            for metric in metrics_to_visualize:
                if metric == "ROC":
                    metric_dict[metric] = [(metric_data["FPR"], metric_data[metric]["TPR"])]
                metric_dict[metric] = [metric_data[metric]]
        else:
            metric_dict["model"].append(model)
            metric_dict["optimizer"].append(optimizer)
            metric_dict["split"].append(split)
            for metric in metrics_to_visualize:
                if metric == "ROC":
                    metric_dict[metric].append((metric_data["FPR"], metric_data[metric]["TPR"]))
                metric_dict[metric].append(metric_data[metric])

        return metric_dict





    def _visualize_cross_validation(self):
        pass


    def visualize_comparison(self):
        pass

    def _visualize_cv_ROC(self):
        pass

    def _visualize_multiple_ROC(self):
        pass


