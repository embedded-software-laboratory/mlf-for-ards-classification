from ml_models.model_interface import Model


class EvaluationInformation:
    def __init__(self, config, model: Model, dataset_training=None, dataset_test=None):

        self.model = model
        self.eval_storage_location = config['storage_path']  # TODO check what is correct
        self.dataset_training = dataset_training  # TODO replace by representation of dataset
        self.dataset_test = dataset_test  # TODO replace by representation of dataset
        self.model_name = model.name
        self.eval_name = config['evaluation']['evaluation_name']
        self.cross_validation_performed = config["process"]["perform_cross_validation"]
        self.evaluation_performed = config["process"]["calculate_evaluation_metrics"]
        if self.cross_validation_performed:

            self.n_splits = config["evaluation"]["cross_validation"]["n_splits"]
            self.shuffle = config["evaluation"]["cross_validation"]["shuffle"]
            self.random_state = config["evaluation"]["cross_validation"]["random_state"]
        else:
            self.n_splits = 0

        self.model_has_proba = model.has_predict_proba()
        self.requested_metrics = config["evaluation"]["evaluation_metrics"]
        self.contained_metrics = []  # TODO check how to communicate incompatible metrics
        for metric in self.requested_metrics:
            if not eval(metric + "().needs_probabilities()"):
                self.contained_metrics.append(metric)

            if self.model_has_proba and eval(metric + "().needs_probabilities()"):
                self.contained_metrics.append(metric)

        if self.model_has_proba:
            self.threshold_optimizers = config["evaluation"]["threshold_optimization_algorithms"]
        else:
            self.threshold_optimizers = ["Standard"]

        self.predicted_labels = None
        self.predicted_probas = None
        self.true_labels = None
