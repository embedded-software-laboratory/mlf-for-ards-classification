from models.model_interface import Model
from sklearn.linear_model import LogisticRegression
import pickle
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd


class Logistic_regression(Model):

    def __init__(self):
            super().__init__()
            self.name = "Logistic_regression"
            self.penalty='l2'
            self.solver='saga'
            self.tol=0.0001
            self.C=1.0
            self.fit_intercept=True
            self.intercept_scaling=1
            self.class_weight=None
            self.random_state=None
            self.max_iter=100000
            self.multi_class='auto'
            self.verbose=0
            self.warm_start=False
            self.n_jobs=None
            self.l1_ratio=None
            self.model=self._init_lr()

    def train_model(self, training_data):
            label = training_data["ards"]
            predictors = training_data.loc[:, training_data.columns != 'ards']


            # Modell trainieren
            self.model=self.model.fit(predictors, label)
            self.calculate_vif(predictors)

    def predict(self, data):
        return self.model.predict(data)

    def predict_proba(self, data):
        return self.model.predict_proba(data)
    #Die Methode muss ein Array zurückgeben, welches für jede Zeile des Dataframes ein Array mit zwei Werten enthält. Der erste Wert gibt die Wahrscheinlichkeit an, dass es sich bei diesem Datensatz nicht um ARDS handelt, und der zweite gibt die Wahrscheinlichkeit an, dass es sich um ARDS handelt.

    def calculate_vif(self, data):
        print("Calculating VIF...")
        vif_data = pd.DataFrame()
        #create empty pandas dataframe
        #vif_data["Feature"] = data.predictors
        vif_data["Feature"] = data.columns
        #assign columns
        vif_data["VIF"] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
        #For each column index i, the VIF is calculated using the variance_inflation_factor function from the statsmodels.stats.outliers_influence module, applied to the entire input data (data.values) and the index i.
        vif_data.round(1)
        print(vif_data)

    def _init_lr(self) -> LogisticRegression:
        """Function that intializes the Random Forest"""

        # Init LR
        logistic_regression = LogisticRegression(
                        penalty = self.penalty,
                        tol = self.tol,
                        C = self.C,
                        fit_intercept = self.fit_intercept,
                        intercept_scaling = self.intercept_scaling,
                        class_weight = self.class_weight,
                        random_state = 3308,
                        solver = self.solver,
                        max_iter = self.max_iter,
                        multi_class = self.multi_class,
                        verbose = self.verbose,
                        warm_start = self.warm_start,
                        n_jobs = self.n_jobs,
                        l1_ratio = self.l1_ratio,
                    )
        return logistic_regression


    def save(self, filepath):
            file = open(filepath + ".txt", "wb")
            pickle.dump(self.model, file)

    def load(self, filepath):
            file = open(filepath + ".txt", "rb")
            self.model = pickle.load(file)

Logistic_regression().save("../Save/Logistic_regression")

    # Laden
Logistic_regression().load("../Save/Logistic_regression")

#Grid Search: In grid search, you define a grid of hyperparameter values that you want to search over. The algorithm then evaluates the model performance for each combination of hyperparameters and returns the combination that performs the best.

#Continuous parameters or discretisiert in clusters like BN?

# Durchführung der Grid-Suche
#grid_search.fit(X_train, y_train)

# Ausgabe der besten Hyperparameter-Kombination
#print("Beste Hyperparameter-Kombination:", grid_search.best_params_)
