from lightgbm import LGBMClassifier
import pickle
from ml_models.model_interface import Model
from ml_models.timeseries_model import TimeSeriesModel

class LightGBMModel(TimeSeriesModel):
    def __init__(self):
        super().__init__()
        self.name = "LightGBMModel"
        self.algorithm = "LightGBM"
        self.boosting_type = 'gbdt'
        self.objective = 'binary'
        # nächsten vier atribute verstehe ich noch nicht
        self.metric = 'binary_error'
        self.is_unbalance=True
        self.feature_fraction = 0.9
        self.bagging_fraction = 0.8
        self.bagging_freq = 5
        self.colsample_bytree = 0.8
        self.verbose = 0
        self.num_leaves = 31
        self.max_depth = -1
        self.learning_rate = 0.1
        self.n_estimators = 100
        # Number of boosted trees to fit.
        self.subsample_for_bin = 200000
        # was ist das?
        self.class_weight = None
        self.min_split_gain = 0.0
        # Minimum loss reduction required to make a further partition on a leaf node of the tree
        self.min_child_weight = 0.001
        self.min_child_samples = 20
        self.subsample = 1.0
        # Subsample ratio of the training instance.
        self.subsample_freq = 0
        colsample_bytree = 1.0
        self.reg_alpha = 0.0
        # L1 regularization term on weights.
        self.reg_lambda = 0.0
        # L2 regularization term on weights.
        self.random_state = None
        self.n_jobs = None
        self.importance_type='split'
        #What about early stopping round als attribut wenn das Model eskaliert?
        self.model = self._init_gbm()

        #klappt das?

    def train_model(self, training_data):
        # Daten und Labels extrahieren
        label = training_data["ards"]
        predictors = training_data.loc[:, training_data.columns != 'ards']

        self.model = self.model.fit(predictors, label)
        self.trained = True

    def predict(self, data):
        # Vorhersage für einen einzelnen Patienten machen
        return self.model.predict(data)

    def predict_proba(self, data):
        # Vorhersage der Wahrscheinlichkeiten für die Klassenzuordnung machen
        return self.model.predict_proba(data)

    def _init_gbm(self) -> LGBMClassifier:
        lightGBM = LGBMClassifier(
                #Attribute
                boosting_type = self.boosting_type,
                objective = self.objective,
                #nächsten vier atribute verstehe ich noch nicht
                metric = self.metric,
                is_unbalance=self.is_unbalance,
                feature_fraction=self.feature_fraction,
                bagging_fraction=self.bagging_fraction,
                bagging_freq=self.bagging_freq,
                verbose=self.verbose,
                num_leaves=self.num_leaves,
                max_depth = self.max_depth,
                learning_rate = self.learning_rate,
                n_estimators = self.n_estimators,
                #Number of boosted trees to fit.
                subsample_for_bin = self.subsample_for_bin,
                    #was ist das?
                class_weight = self.class_weight,
                min_split_gain = self.min_split_gain,
                #Minimum loss reduction required to make a further partition on a leaf node of the tree
                min_child_weight = self.min_child_weight,
                min_child_samples = self.min_child_samples,
                subsample = self.subsample,
                #Subsample ratio of the training instance.
                subsample_freq = self.subsample_freq,
                colsample_bytree = self.colsample_bytree,
                reg_alpha = self.reg_alpha,
                    #L1 regularization term on weights.
                reg_lambda = self.reg_lambda,
                    #L2 regularization term on weights.
                random_state = self.random_state,
                n_jobs = self.n_jobs,
                importance_type = self.importance_type
        )
        return lightGBM

    def get_params(self):
        return self.model.get_params(True)

    def save_model(self, filepath):
        file = open(filepath + ".txt", "wb")
        pickle.dump(self.model, file)

    def load_model(self, filepath):
        file = open(filepath + ".txt", "rb")
        self.model = pickle.load(file)

    def has_predict_proba(self):
        return True


#LightGBMModel().save("./Save/LightGBMModel")

# Laden
#LightGBMModel().load("./Save/LightGBMModel")
