

from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, roc_curve, f1_score, auc, \
    precision_recall_curve

from metrics.Models import *





class Accuracy(FloatMetricSpec):

    def calculate_metric(self, metric_params: dict, stage:str) -> GenericMetric:
        predicted_class = metric_params['predicted_label']
        true_class = metric_params['true_labels']
        return GenericMetric(metric_name="Accuracy" + " " + stage,
                             metric_value=FloatValue(metric_value=accuracy_score(true_class, predicted_class)),
                             metric_spec=Accuracy())

    def needs_probabilities(self) -> bool:
        return False

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=Accuracy())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=Accuracy())


class AUC(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        true_label = metric_parameters['true_labels']
        predicition_probs = metric_parameters['prediction_probs']
        fpr, tpr, _ = roc_curve(true_label, predicition_probs)
        auc_score = auc(fpr, tpr)
        return GenericMetric(metric_name="AUC" + " " + stage,
                                            metric_value=FloatValue(metric_value=auc_score), metric_spec=AUC())

    def needs_probabilities(self) -> bool:
        return True

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=AUC())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=AUC())


class F1Score(FloatMetricSpec):

    def calculate_metric(self, metric_params: dict, stage:str) -> GenericMetric:
        predicted_class = metric_params['predicted_label']
        true_class = metric_params['true_labels']
        return GenericMetric(metric_name="F1-Score" + " " + stage,
                             metric_value=FloatValue(metric_value=f1_score(predicted_class, true_class)),
                             metric_spec=F1Score())

    def needs_probabilities(self) -> bool:
        return False

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=F1Score())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=F1Score())


class FPR(ListMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        prediction_probs = metric_parameters['prediction_probs']
        true_labels = metric_parameters['true_labels']
        fpr, _, _ = roc_curve(true_labels, prediction_probs)
        return GenericMetric(metric_name="FPR" + " " + stage, metric_value=ListValue(metric_value=fpr),
                                            metric_spec=FPR())


    def needs_probabilities(self) -> bool:
        return True

    def create_from_value(self, metric_value: ListValue, metric_name: str) -> GenericMetric:
        return GenericMetric(metric_name=metric_name,
                             metric_value=StringValue(metric_value="Mean calculation makes no sense for FPR"),
                             metric_spec=FPR())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=ListValue(metric_value=metric_dict['metric_value']),
                             metric_spec=FPR())


class MCC(FloatMetricSpec):

    def calculate_metric(self, metric_params: dict, stage:str) -> GenericMetric:
        predicted_class = metric_params['predicted_label']
        true_class = metric_params['true_labels']
        return GenericMetric(metric_name="MCC" + " " + stage,
                             metric_value=FloatValue(metric_value=matthews_corrcoef(true_class, predicted_class)),
                             metric_spec=MCC())

    def needs_probabilities(self) -> bool:
        return False

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=MCC())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=MCC())


class OptimalProbability(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        fpr = metric_parameters['fpr']
        tpr = metric_parameters['tpr']
        threshold = metric_parameters['thresholds']
        calc_func = metric_parameters['calc_func']
        optimal_prob = calc_func(tpr, fpr, threshold)
        return GenericMetric(metric_name="Optimal Decision Threshold" + " " + stage,
                             metric_value=FloatValue(metric_value=optimal_prob),
                             metric_spec=OptimalProbability())

    def needs_probabilities(self) -> bool:
        return True

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=OptimalProbability())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=OptimalProbability())


class Sensitivity(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        prediction_labels = metric_parameters['predicted_label']
        true_labels = metric_parameters['true_labels']
        tn, fp, fn, tp = confusion_matrix(true_labels, prediction_labels).ravel()

        return GenericMetric(metric_name="Sensitivity" + " " + stage,
                             metric_value=FloatValue(metric_value=(tp / (tp + fn))),
                             metric_spec=Sensitivity())

    def needs_probabilities(self) -> bool:
        return False

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=Sensitivity())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=Sensitivity())


class Specificity(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        prediction_labels = metric_parameters['predicted_label']
        true_labels = metric_parameters['true_labels']
        tn, fp, fn, tp = confusion_matrix(true_labels, prediction_labels).ravel()

        return GenericMetric(metric_name="Specificity" + " " + stage,
                             metric_value=FloatValue(metric_value=(tn / (tn + fp))),
                             metric_spec=Specificity())

    def needs_probabilities(self) -> bool:
        return False

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=Specificity())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=Specificity())


class TPR(ListMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        prediction_probs = metric_parameters['prediction_probs']
        true_labels = metric_parameters['true_labels']
        _, tpr, _ = roc_curve(true_labels, prediction_probs)
        return GenericMetric(metric_name="TPR" + " " + stage, metric_value=ListValue(metric_value=tpr), metric_spec=TPR())

    def needs_probabilities(self) -> bool:
        return True

    def create_from_value(self, metric_value: ListValue, metric_name: str) -> GenericMetric:
        return GenericMetric(metric_name=metric_name, metric_value=StringValue(metric_value="Mean calculation makes no sense for TPR"), metric_spec=TPR())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=ListValue(metric_value=metric_dict['metric_value']),
                             metric_spec=TPR())

class PPV(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        prediction_labels = metric_parameters['predicted_label']
        true_labels = metric_parameters['true_labels']
        tn, fp, fn, tp = confusion_matrix(true_labels, prediction_labels).ravel()

        return GenericMetric(metric_name="PPV" + " " + stage,
                             metric_value=FloatValue(metric_value=(tp / (tp + fp))),
                             metric_spec=PPV())

    def needs_probabilities(self) -> bool:
        return False

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=PPV())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=PPV())




class NPV(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        prediction_labels = metric_parameters['predicted_label']
        true_labels = metric_parameters['true_labels']
        tn, fp, fn, tp = confusion_matrix(true_labels, prediction_labels).ravel()

        return GenericMetric(metric_name="NPV" + " " + stage,
                             metric_value=FloatValue(metric_value=(tn / (tn + fn))),
                             metric_spec=NPV())

    def needs_probabilities(self) -> bool:
        return False

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=NPV())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=NPV())


class Precisions(ListMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        prediction_probs = metric_parameters['prediction_probs']
        true_labels = metric_parameters['true_labels']
        precision, _, _ = precision_recall_curve(true_labels, prediction_probs)
        return GenericMetric(metric_name="Precision" + " " + stage, metric_value=ListValue(metric_value=precision), metric_spec=Precisions())

    def needs_probabilities(self) -> bool:
        return True

    def create_from_value(self, metric_value: ListValue, metric_name: str) -> GenericMetric:
        return GenericMetric(metric_name=metric_name, metric_value=StringValue(metric_value="Mean calculation makes no sense for Precision"), metric_spec=Precisions())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=ListValue(metric_value=metric_dict['metric_value']),
                             metric_spec=Precisions())

class Recalls(ListMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        prediction_probs = metric_parameters['prediction_probs']
        true_labels = metric_parameters['true_labels']
        _, recall, _ = precision_recall_curve(true_labels, prediction_probs)
        return GenericMetric(metric_name="Recall" + " " + stage, metric_value=ListValue(metric_value=recall), metric_spec=Recalls())

    def needs_probabilities(self) -> bool:
        return True

    def create_from_value(self, metric_value: ListValue, metric_name: str) -> GenericMetric:
        return GenericMetric(metric_name=metric_name, metric_value=StringValue(metric_value="Mean calculation makes no sense for Recall"), metric_spec=Recalls())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=ListValue(metric_value=metric_dict['metric_value']),
                             metric_spec=Recalls())