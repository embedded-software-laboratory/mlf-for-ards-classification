from models.model_interface import Model
from sklearn.ensemble import RandomForestClassifier
import pickle


class Random_forest(Model):

    def __init__(self):
        super().__init__()
        self.name = "Random Forest"
        self.n_estimators = 700
        self.criterion = "gini"
        self.max_depth = 200
        self.min_samples_split = 10
        self.min_samples_leaf = 1
        self.min_weight_fraction_leaf = 0.0
        self.max_features = "sqrt"
        self.max_leaf_nodes = None
        self.min_impurity_decrease = 0.0
        self.bootstrap = True
        self.oob_score = False
        self.n_jobs = -1
        self.verbose = 0
        self.warm_start = False
        self.class_weight = None
        self.ccp_alpha = 0.0
        self.max_samples = None
        self.model = self._init_forest()

    def train_model(self, training_data):
        """Function that starts the learning process of the RF and stores the resulting model after completion"""

        # Init forest and read training data
        label = training_data["ards"]
        predictors = training_data.loc[:, training_data.columns != 'ards']

        # Learn and store resulting model
        self.model = self.model.fit(predictors, label)

    def predict(self, data):
        return self.model.predict(data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)

    def _init_forest(self) -> RandomForestClassifier:
        """Function that intializes the Random Forest"""

        # Init RF
        random_forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=3308,
            verbose=self.verbose,
            warm_start=self.warm_start,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
        )
        return random_forest

    def save(self, filepath):
        file = open(filepath + ".txt", "wb")
        pickle.dump(self.model, file)

    def load(self, filepath):
        file = open(filepath + ".txt", "rb")
        self.model = pickle.load(file)

    def has_predict_proba(self):
        return True
