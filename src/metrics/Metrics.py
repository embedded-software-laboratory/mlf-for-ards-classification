import logging
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix, roc_curve, f1_score, auc, \
    precision_recall_curve

from metrics.Models import *

logger = logging.getLogger(__name__)


class Accuracy(FloatMetricSpec):

    def calculate_metric(self, metric_params: dict, stage:str) -> GenericMetric:
        logger.debug(f"Calculating Accuracy for stage='{stage}'")
        predicted_class = metric_params['predicted_label']
        true_class = metric_params['true_labels']
        value = accuracy_score(true_class, predicted_class)
        logger.info(f"Accuracy {stage}: {value:.4f}")
        return GenericMetric(metric_name="Accuracy" + " " + stage,
                             metric_value=FloatValue(metric_value=value),
                             metric_spec=Accuracy())

    def needs_probabilities(self) -> bool:
        return False

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        logger.debug(f"Creating Accuracy metric from value: {metric_value.metric_value}")
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=Accuracy())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        logger.debug(f"Reconstructing Accuracy metric from dict: {metric_dict.get('metric_name')}")
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=Accuracy())


class AUC(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        logger.debug(f"Calculating AUC for stage='{stage}'")
        true_label = metric_parameters['true_labels']
        predicition_probs = metric_parameters['prediction_probs']
        fpr, tpr, _ = roc_curve(true_label, predicition_probs)
        auc_score = auc(fpr, tpr)
        logger.info(f"AUC {stage}: {auc_score:.4f}")
        return GenericMetric(metric_name="AUC" + " " + stage,
                                            metric_value=FloatValue(metric_value=auc_score), metric_spec=AUC())

    def needs_probabilities(self) -> bool:
        return True

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        logger.debug(f"Creating AUC metric from value: {metric_value.metric_value}")
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=AUC())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        logger.debug(f"Reconstructing AUC metric from dict: {metric_dict.get('metric_name')}")
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=AUC())


class F1Score(FloatMetricSpec):

    def calculate_metric(self, metric_params: dict, stage:str) -> GenericMetric:
        logger.debug(f"Calculating F1-Score for stage='{stage}'")
        predicted_class = metric_params['predicted_label']
        true_class = metric_params['true_labels']
        value = f1_score(predicted_class, true_class)
        logger.info(f"F1-Score {stage}: {value:.4f}")
        return GenericMetric(metric_name="F1-Score" + " " + stage,
                             metric_value=FloatValue(metric_value=value),
                             metric_spec=F1Score())

    def needs_probabilities(self) -> bool:
        return False

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        logger.debug(f"Creating F1Score metric from value: {metric_value.metric_value}")
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=F1Score())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        logger.debug(f"Reconstructing F1Score metric from dict: {metric_dict.get('metric_name')}")
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=F1Score())


class FPR(ListMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        logger.debug(f"Calculating FPR for stage='{stage}'")
        prediction_probs = metric_parameters['prediction_probs']
        true_labels = metric_parameters['true_labels']
        fpr, _, _ = roc_curve(true_labels, prediction_probs)
        logger.info(f"FPR {stage}: array length {len(fpr)}")
        return GenericMetric(metric_name="FPR" + " " + stage, metric_value=ListValue(metric_value=fpr),
                                            metric_spec=FPR())


    def needs_probabilities(self) -> bool:
        return True

    def create_from_value(self, metric_value: ListValue, metric_name: str) -> GenericMetric:
        logger.debug("Creating FPR metric from value (mean not applicable)")
        return GenericMetric(metric_name=metric_name,
                             metric_value=StringValue(metric_value="Mean calculation makes no sense for FPR"),
                             metric_spec=FPR())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        logger.debug(f"Reconstructing FPR metric from dict: {metric_dict.get('metric_name')}")
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=ListValue(metric_value=metric_dict['metric_value']),
                             metric_spec=FPR())


class MCC(FloatMetricSpec):

    def calculate_metric(self, metric_params: dict, stage:str) -> GenericMetric:
        logger.debug(f"Calculating MCC for stage='{stage}'")
        predicted_class = metric_params['predicted_label']
        true_class = metric_params['true_labels']
        value = matthews_corrcoef(true_class, predicted_class)
        logger.info(f"MCC {stage}: {value:.4f}")
        return GenericMetric(metric_name="MCC" + " " + stage,
                             metric_value=FloatValue(metric_value=value),
                             metric_spec=MCC())

    def needs_probabilities(self) -> bool:
        return False

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        logger.debug(f"Creating MCC metric from value: {metric_value.metric_value}")
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=MCC())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        logger.debug(f"Reconstructing MCC metric from dict: {metric_dict.get('metric_name')}")
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=MCC())


class OptimalProbability(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        logger.debug(f"Calculating OptimalProbability for stage='{stage}'")
        fpr = metric_parameters['fpr']
        tpr = metric_parameters['tpr']
        threshold = metric_parameters['thresholds']
        calc_func = metric_parameters['calc_func']
        optimal_prob = calc_func(tpr, fpr, threshold)
        logger.info(f"OptimalProbability {stage}: {optimal_prob:.4f}")
        return GenericMetric(metric_name="Optimal Decision Threshold" + " " + stage,
                             metric_value=FloatValue(metric_value=optimal_prob),
                             metric_spec=OptimalProbability())

    def needs_probabilities(self) -> bool:
        return True

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        logger.debug(f"Creating OptimalProbability metric from value: {metric_value.metric_value}")
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=OptimalProbability())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        logger.debug(f"Reconstructing OptimalProbability metric from dict: {metric_dict.get('metric_name')}")
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=OptimalProbability())


class Sensitivity(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        logger.debug(f"Calculating Sensitivity for stage='{stage}'")
        prediction_labels = metric_parameters['predicted_label']
        true_labels = metric_parameters['true_labels']
        tn, fp, fn, tp = confusion_matrix(true_labels, prediction_labels).ravel()

        value = (tp / (tp + fn))
        logger.info(f"Sensitivity {stage}: {value:.4f}")
        return GenericMetric(metric_name="Sensitivity" + " " + stage,
                             metric_value=FloatValue(metric_value=value),
                             metric_spec=Sensitivity())

    def needs_probabilities(self) -> bool:
        return False

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        logger.debug(f"Creating Sensitivity metric from value: {metric_value.metric_value}")
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=Sensitivity())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        logger.debug(f"Reconstructing Sensitivity metric from dict: {metric_dict.get('metric_name')}")
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=Sensitivity())


class Specificity(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        logger.debug(f"Calculating Specificity for stage='{stage}'")
        prediction_labels = metric_parameters['predicted_label']
        true_labels = metric_parameters['true_labels']
        tn, fp, fn, tp = confusion_matrix(true_labels, prediction_labels).ravel()

        value = (tn / (tn + fp))
        logger.info(f"Specificity {stage}: {value:.4f}")
        return GenericMetric(metric_name="Specificity" + " " + stage,
                             metric_value=FloatValue(metric_value=value),
                             metric_spec=Specificity())

    def needs_probabilities(self) -> bool:
        return False

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        logger.debug(f"Creating Specificity metric from value: {metric_value.metric_value}")
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=Specificity())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        logger.debug(f"Reconstructing Specificity metric from dict: {metric_dict.get('metric_name')}")
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=Specificity())


class TPR(ListMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        logger.debug(f"Calculating TPR for stage='{stage}'")
        prediction_probs = metric_parameters['prediction_probs']
        true_labels = metric_parameters['true_labels']
        _, tpr, _ = roc_curve(true_labels, prediction_probs)
        logger.info(f"TPR {stage}: array length {len(tpr)}")
        return GenericMetric(metric_name="TPR" + " " + stage, metric_value=ListValue(metric_value=tpr), metric_spec=TPR())

    def needs_probabilities(self) -> bool:
        return True

    def create_from_value(self, metric_value: ListValue, metric_name: str) -> GenericMetric:
        logger.debug("Creating TPR metric from value (mean not applicable)")
        return GenericMetric(metric_name=metric_name, metric_value=StringValue(metric_value="Mean calculation makes no sense for TPR"), metric_spec=TPR())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        logger.debug(f"Reconstructing TPR metric from dict: {metric_dict.get('metric_name')}")
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=ListValue(metric_value=metric_dict['metric_value']),
                             metric_spec=TPR())

class PPV(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        logger.debug(f"Calculating PPV for stage='{stage}'")
        prediction_labels = metric_parameters['predicted_label']
        true_labels = metric_parameters['true_labels']
        tn, fp, fn, tp = confusion_matrix(true_labels, prediction_labels).ravel()

        value = (tp / (tp + fp))
        logger.info(f"PPV {stage}: {value:.4f}")
        return GenericMetric(metric_name="PPV" + " " + stage,
                             metric_value=FloatValue(metric_value=value),
                             metric_spec=PPV())

    def needs_probabilities(self) -> bool:
        return False

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        logger.debug(f"Creating PPV metric from value: {metric_value.metric_value}")
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=PPV())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        logger.debug(f"Reconstructing PPV metric from dict: {metric_dict.get('metric_name')}")
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=PPV())




class NPV(FloatMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        logger.debug(f"Calculating NPV for stage='{stage}'")
        prediction_labels = metric_parameters['predicted_label']
        true_labels = metric_parameters['true_labels']
        tn, fp, fn, tp = confusion_matrix(true_labels, prediction_labels).ravel()

        value = (tn / (tn + fn))
        logger.info(f"NPV {stage}: {value:.4f}")
        return GenericMetric(metric_name="NPV" + " " + stage,
                             metric_value=FloatValue(metric_value=value),
                             metric_spec=NPV())

    def needs_probabilities(self) -> bool:
        return False

    def create_from_value(self, metric_value: FloatValue, metric_name: str) -> GenericMetric:
        logger.debug(f"Creating NPV metric from value: {metric_value.metric_value}")
        return GenericMetric(metric_name=metric_name, metric_value=metric_value, metric_spec=NPV())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        logger.debug(f"Reconstructing NPV metric from dict: {metric_dict.get('metric_name')}")
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=FloatValue(metric_value=metric_dict['metric_value']),
                             metric_spec=NPV())


class Precisions(ListMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        logger.debug(f"Calculating Precision curve for stage='{stage}'")
        prediction_probs = metric_parameters['prediction_probs']
        true_labels = metric_parameters['true_labels']
        precision, _, _ = precision_recall_curve(true_labels, prediction_probs)
        logger.info(f"Precision curve {stage}: array length {len(precision)}")
        return GenericMetric(metric_name="Precision" + " " + stage, metric_value=ListValue(metric_value=precision), metric_spec=Precisions())

    def needs_probabilities(self) -> bool:
        return True

    def create_from_value(self, metric_value: ListValue, metric_name: str) -> GenericMetric:
        logger.debug("Creating Precision metric from value (mean not applicable)")
        return GenericMetric(metric_name=metric_name, metric_value=StringValue(metric_value="Mean calculation makes no sense for Precision"), metric_spec=Precisions())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        logger.debug(f"Reconstructing Precision metric from dict: {metric_dict.get('metric_name')}")
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=ListValue(metric_value=metric_dict['metric_value']),
                             metric_spec=Precisions())

class Recalls(ListMetricSpec):
    def calculate_metric(self, metric_parameters: dict, stage:str) -> GenericMetric:
        logger.debug(f"Calculating Recall curve for stage='{stage}'")
        prediction_probs = metric_parameters['prediction_probs']
        true_labels = metric_parameters['true_labels']
        _, recall, _ = precision_recall_curve(true_labels, prediction_probs)
        logger.info(f"Recall curve {stage}: array length {len(recall)}")
        return GenericMetric(metric_name="Recall" + " " + stage, metric_value=ListValue(metric_value=recall), metric_spec=Recalls())

    def needs_probabilities(self) -> bool:
        return True

    def create_from_value(self, metric_value: ListValue, metric_name: str) -> GenericMetric:
        logger.debug("Creating Recall metric from value (mean not applicable)")
        return GenericMetric(metric_name=metric_name, metric_value=StringValue(metric_value="Mean calculation makes no sense for Recall"), metric_spec=Recalls())

    def create_from_dict(self, metric_dict: dict) -> GenericMetric:
        logger.debug(f"Reconstructing Recall metric from dict: {metric_dict.get('metric_name')}")
        return GenericMetric(metric_name=metric_dict['metric_name'],
                             metric_value=ListValue(metric_value=metric_dict['metric_value']),
                             metric_spec=Recalls())