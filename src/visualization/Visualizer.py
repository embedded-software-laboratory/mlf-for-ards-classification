from sympy import false

from metrics import ExperimentResult


class Visualisizer:

    def __init__(self, results: ExperimentResult, config):
        self.full_results = results
        self.result_name = results["result_name"]
        self.evaluation_results = None
        self.training_results = None
        self.cv_evaluation_results = None
        self.cv_training_results = None
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
                    if model in  contained_models:
                        present_models.append(model)
                    else:
                        print(f"No information for {model} available. Skipping...")


    def _visualize_model(self, setting, present_models: list):
        pass

    def _visualize_single_split(self, model: str, eval_type: str, optimizer: str, split: str):
        active_metrics = self.active_setting["active_metrics"]
        available_metrics = list(self.full_results.contained_model_results[model].contained_evals[eval_type].contained_optimizers[optimizer].contained_splits[split].contained_metrics.keys())
        metrics_to_visualize = []
        if "all" in active_metrics:
            metrics_to_visualize = available_metrics
        else:
            for metric in active_metrics:
                if metric in available_metrics:
                    metrics_to_visualize.append(metric)






        pass

    def _visualize_cross_validation(self):
        pass


    def visualize_comparison(self):
        pass

    def _visualize_cv_ROC(self):
        pass

    def _visualize_multiple_ROC(self):
        pass


