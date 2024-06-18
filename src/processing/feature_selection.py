from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE, RFECV, SelectFromModel, SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

class Feature_selection:
    def __init__(self, config) -> None:
        self.feature_selection_method = "low_variance"
        self.available_methods = ["low_variance", "univariate", "recursive", "recursive_with_cv", "L1", "tree", "sequential"]
        self.set_selection_method(config["method"])
        self.variance = config["variance"]
        self.k = config["k"]

    def perform_feature_selection(self, dataframe):
        print("Start feature selection...")
        if self.feature_selection_method == "low_variance":
            print("perform low variance selection...")
            dataframe = self.low_variance_selection(dataframe, self.variance)
            print("Done!")
        if self.feature_selection_method == "univariate":
            print("perform univariate selection...")
            dataframe = self.univariate_selection(dataframe, self.k)
        if self.feature_selection_method == "recursive":
            dataframe = self.recursive_selection(dataframe, self.k)
        if self.feature_selection_method == "recursive_with_cv":
            dataframe = self.recursive_selection_with_crossvalidation(dataframe)
        if self.feature_selection_method == "L1":
            dataframe = self.l1_selection(dataframe)
        if self.feature_selection_method == "tree":
            dataframe = self.tree_selection(dataframe)
        if self.feature_selection_method == "sequential":
            dataframe = self.sequential_selection(dataframe)
        return dataframe
        

    def low_variance_selection(self, dataframe, variance):
        selection = VarianceThreshold(threshold=variance)
        temp, data = self.split_dataframe(dataframe)
        result = selection.fit_transform(data)
        result_dataframe_temp = pd.DataFrame(result, columns=selection.get_feature_names_out(data.columns), index=data.index)
        result_dataframe = pd.concat([temp, result_dataframe_temp], axis=1)
        return result_dataframe
    
    def univariate_selection(self, dataframe, k_value):
        temp, data = self.split_dataframe(dataframe)
        selection = SelectKBest(f_classif, k=k_value)
        result = selection.fit_transform(data, temp["ards"])
        result_dataframe_temp = pd.DataFrame(result, columns=selection.get_feature_names_out(data.columns), index=data.index)
        result_dataframe = pd.concat([temp, result_dataframe_temp], axis=1)
        return result_dataframe
    
    def recursive_selection(self, dataframe, k=None):
        temp, data = self.split_dataframe(dataframe)
        svc = SVC(kernel="linear", C=1)
        rfe = RFE(estimator=svc, n_features_to_select=5, step=1)
        rfe.fit(data, temp["ards"])
        ranking = rfe.ranking_
        if k == None:
            k = rfe.n_features_
        result_dataframe_temp = pd.DataFrame(data.iloc[:, ranking[0:k]], columns=data.iloc[:, ranking[0:k]].columns)
        result_dataframe = pd.concat([temp, result_dataframe_temp], axis=1)
        return result_dataframe
    
    def recursive_selection_with_crossvalidation(self, dataframe, k=None):
        temp, data = self.split_dataframe(dataframe)
        svc = SVC(kernel="linear", C=1)
        cv = StratifiedKFold(5)
        rfecv = RFECV(estimator=svc, min_features_to_select=5, step=1, cv=cv, scoring="accuracy", n_jobs=2)
        rfecv.fit(data, temp["ards"])
        ranking = rfecv.ranking_
        if k == None:
            k = rfecv.n_features_
        result_dataframe_temp = pd.DataFrame(data.iloc[:, ranking[0:k]], columns=data.iloc[:, ranking[0:k]].columns)
        result_dataframe = pd.concat([temp, result_dataframe_temp], axis=1)
        return result_dataframe
    
    def l1_selection(self, dataframe):
        temp, data = self.split_dataframe(dataframe)
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(data, temp["ards"])
        model = SelectFromModel(lsvc, prefit=True)
        result = model.fit_transform(data)
        result_dataframe_temp = pd.DataFrame(result, columns=model.get_feature_names_out(data.columns), index=data.index)
        result_dataframe = pd.concat([temp, result_dataframe_temp], axis=1)
        return result_dataframe
    
    def tree_selection(self, dataframe):
        temp, data = self.split_dataframe(dataframe)
        clf = ExtraTreesClassifier(n_estimators=50)
        clf = clf.fit(data, temp["ards"])
        model = SelectFromModel(clf, prefit=True)
        result = model.fit_transform(data)
        result_dataframe_temp = pd.DataFrame(result, columns=model.get_feature_names_out(data.columns), index=data.index)
        result_dataframe = pd.concat([temp, result_dataframe_temp], axis=1)
        return result_dataframe
    
    def sequential_selection(self, dataframe):
        temp, data = self.split_dataframe(dataframe)
        knn = KNeighborsClassifier(n_neighbors=3)
        sfs = SequentialFeatureSelector(knn, n_features_to_select=3)
        sfs = sfs.fit(data, temp["ards"])
        result = sfs.fit_transform(data, temp["ards"])
        result_dataframe_temp = pd.DataFrame(result, columns=sfs.get_feature_names_out(data.columns), index=data.index)
        result_dataframe = pd.concat([temp, result_dataframe_temp], axis=1)
        return result_dataframe

    #splits the given dataframe in two dataframes; the first consists of the columns patient_id, ARDS and time (which should not be removed in the feature selection process) and the other consists of the other columns
    def split_dataframe(self, dataframe):
        df1 = dataframe[["ards", "patient_id", "time"]]
        df2 = dataframe.drop(columns=["patient_id", "ards", "time"])
        return df1, df2
    
    def set_selection_method(self, method):
        if method not in self.available_methods:
            raise RuntimeError("Feature selection method " + method + " not available. Available methods are " + str(self.available_methods))
        self.feature_selection_method = method