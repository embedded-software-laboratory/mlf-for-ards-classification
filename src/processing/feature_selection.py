from processing.datasets_metadata import FeatureSelectionMetaData

import pandas as pd
import logging

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE, RFECV, SelectFromModel, SequentialFeatureSelector
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

logger = logging.getLogger(__name__)


class FeatureSelector:
    """
    Implements various feature selection methods to reduce dimensionality and improve model performance.
    Supports multiple selection algorithms including variance-based, univariate, recursive, L1, tree-based,
    and sequential feature selection approaches.
    """
    
    def __init__(self, config) -> None:
        """
        Initializes the FeatureSelection with configuration parameters.
        
        Args:
            config: Configuration dictionary containing feature selection method, variance threshold, and k value
        """
        logger.info("Initializing FeatureSelection...")
        self.feature_selection_method = "low_variance"
        self.available_methods = ["low_variance", "univariate", "recursive", "recursive_with_cv", "L1", "tree", "sequential"]
        self.set_selection_method(config["method"])
        self.variance = config["variance"]
        self.k = config["k"]
        self.meta_data = None
        logger.info(f"Variance threshold: {self.variance}, K value: {self.k}")
        logger.info("FeatureSelection initialized successfully.")

    def create_meta_data(self):
        """
        Creates metadata object describing the feature selection configuration and results.
        Stores information about the algorithm used, variance threshold, and number of features selected.
        """
        logger.info("Creating feature selection metadata...")
        min_variance = None
        num_features = None
        if self.feature_selection_method == "low_variance":
            min_variance = self.variance
        if self.feature_selection_method in ["univariate", "recursive", "recursive_with_cv"]:
            num_features = self.k
        self.meta_data = FeatureSelectionMetaData(
            feature_selection_algorithm=self.feature_selection_method,
            min_required_variance=min_variance,
            num_features_to_select=num_features,
            first_selection=True
        )
        logger.info(f"Metadata created for method: {self.feature_selection_method}")

    def perform_feature_selection(self, dataframe):
        """
        Main entry point for feature selection. Routes to appropriate selection method
        based on configured algorithm.
        
        Args:
            dataframe: Input DataFrame with all features including metadata columns (ards, patient_id, time)
            
        Returns:
            DataFrame with selected features and metadata columns preserved
        """
        initial_feature_count = dataframe.shape[1]
        logger.info("=" * 80)
        logger.info(f"Starting feature selection using method: {self.feature_selection_method}")
        logger.info(f"Initial data shape: {dataframe.shape}")
        logger.info("=" * 80)
        
        if self.feature_selection_method == "low_variance":
            logger.info(f"Applying Low Variance selection with threshold: {self.variance}")
            dataframe = self.low_variance_selection(dataframe, self.variance)
        elif self.feature_selection_method == "univariate":
            logger.info(f"Applying Univariate selection with k: {self.k}")
            dataframe = self.univariate_selection(dataframe, self.k)
        elif self.feature_selection_method == "recursive":
            logger.info(f"Applying Recursive Feature Elimination with k: {self.k}")
            dataframe = self.recursive_selection(dataframe, self.k)
        elif self.feature_selection_method == "recursive_with_cv":
            logger.info("Applying Recursive Feature Elimination with Cross-Validation")
            dataframe = self.recursive_selection_with_crossvalidation(dataframe)
        elif self.feature_selection_method == "L1":
            logger.info("Applying L1-based feature selection (Linear SVC)")
            dataframe = self.l1_selection(dataframe)
        elif self.feature_selection_method == "tree":
            logger.info("Applying Tree-based feature selection (Extra Trees)")
            dataframe = self.tree_selection(dataframe)
        elif self.feature_selection_method == "sequential":
            logger.info("Applying Sequential Feature Selection (KNN-based)")
            dataframe = self.sequential_selection(dataframe)
        
        final_feature_count = dataframe.shape[1]
        features_removed = initial_feature_count - final_feature_count
        logger.info("=" * 80)
        logger.info(f"Feature selection completed successfully")
        logger.info(f"Features removed: {features_removed} ({initial_feature_count} -> {final_feature_count})")
        logger.info(f"Final data shape: {dataframe.shape}")
        logger.info("=" * 80)
        return dataframe

    def low_variance_selection(self, dataframe, variance):
        """
        Removes features with variance below the specified threshold.
        Low variance features contain little information for distinguishing classes.
        
        Args:
            dataframe: Input DataFrame
            variance: Variance threshold below which features are removed
            
        Returns:
            DataFrame with only features having variance >= threshold
        """
        logger.info(f"Applying variance threshold: {variance}")
        selection = VarianceThreshold(threshold=variance)
        temp, data = self.split_dataframe(dataframe)
        result = selection.fit_transform(data)
        selected_features = selection.get_feature_names_out(data.columns)
        logger.info(f"Selected {len(selected_features)} features out of {data.shape[1]}")
        result_dataframe_temp = pd.DataFrame(result, columns=selected_features, index=data.index)
        result_dataframe = pd.concat([temp, result_dataframe_temp], axis=1)
        return result_dataframe
    
    def univariate_selection(self, dataframe, k_value):
        """
        Selects k best features based on univariate statistical tests (f_classif).
        Ranks features by their statistical significance with respect to the target variable.
        
        Args:
            dataframe: Input DataFrame
            k_value: Number of top features to select
            
        Returns:
            DataFrame containing the k best features plus metadata columns
        """
        logger.info(f"Selecting top {k_value} features using univariate statistical test")
        temp, data = self.split_dataframe(dataframe)
        selection = SelectKBest(f_classif, k=k_value)
        result = selection.fit_transform(data, temp["ards"])
        selected_features = selection.get_feature_names_out(data.columns)
        logger.info(f"Selected features: {list(selected_features)}")
        result_dataframe_temp = pd.DataFrame(result, columns=selected_features, index=data.index)
        result_dataframe = pd.concat([temp, result_dataframe_temp], axis=1)
        return result_dataframe
    
    def recursive_selection(self, dataframe, k=None):
        """
        Applies Recursive Feature Elimination (RFE) using SVM to select features.
        Iteratively removes features and retrains the model until desired number is reached.
        
        Args:
            dataframe: Input DataFrame
            k: Number of features to select (uses RFE default if None)
            
        Returns:
            DataFrame with selected features from RFE
        """
        logger.info("Starting Recursive Feature Elimination (RFE) with SVM...")
        temp, data = self.split_dataframe(dataframe)
        svc = SVC(kernel="linear", C=1)
        rfe = RFE(estimator=svc, n_features_to_select=5, step=1)
        logger.info("Fitting RFE model...")
        rfe.fit(data, temp["ards"])
        ranking = rfe.ranking_
        if k is None:
            k = rfe.n_features_
        logger.info(f"RFE selected {k} features")
        result_dataframe_temp = pd.DataFrame(data.iloc[:, ranking[0:k]], columns=data.iloc[:, ranking[0:k]].columns)
        result_dataframe = pd.concat([temp, result_dataframe_temp], axis=1)
        return result_dataframe
    
    def recursive_selection_with_crossvalidation(self, dataframe, k=None):
        """
        Applies Recursive Feature Elimination with Cross-Validation (RFECV).
        Automatically determines optimal number of features using stratified k-fold cross-validation.
        
        Args:
            dataframe: Input DataFrame
            k: Optional override for number of features (uses RFECV result if None)
            
        Returns:
            DataFrame with features selected by RFECV
        """
        logger.info("Starting Recursive Feature Elimination with Cross-Validation (RFECV)...")
        temp, data = self.split_dataframe(dataframe)
        svc = SVC(kernel="linear", C=1)
        cv = StratifiedKFold(5)
        logger.info("Fitting RFECV model with 5-fold cross-validation...")
        rfecv = RFECV(estimator=svc, min_features_to_select=5, step=1, cv=cv, scoring="accuracy", n_jobs=2)
        rfecv.fit(data, temp["ards"])
        optimal_features = rfecv.n_features_
        logger.info(f"RFECV determined optimal number of features: {optimal_features}")
        ranking = rfecv.ranking_
        if k is None:
            k = optimal_features
        result_dataframe_temp = pd.DataFrame(data.iloc[:, ranking[0:k]], columns=data.iloc[:, ranking[0:k]].columns)
        result_dataframe = pd.concat([temp, result_dataframe_temp], axis=1)
        return result_dataframe
    
    def l1_selection(self, dataframe):
        """
        Applies L1-based feature selection using Linear SVC.
        Features with non-zero coefficients are considered important.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            DataFrame with features selected by L1 regularization
        """
        logger.info("Starting L1-based feature selection using LinearSVC...")
        temp, data = self.split_dataframe(dataframe)
        logger.info("Training LinearSVC with L1 penalty...")
        lsvc = LinearSVC(C=0.01, penalty="l1", dual=False, max_iter=2000)
        lsvc.fit(data, temp["ards"])
        model = SelectFromModel(lsvc, prefit=True)
        result = model.fit_transform(data)
        selected_features = model.get_feature_names_out(data.columns)
        logger.info(f"L1 selection retained {len(selected_features)} features")
        result_dataframe_temp = pd.DataFrame(result, columns=selected_features, index=data.index)
        result_dataframe = pd.concat([temp, result_dataframe_temp], axis=1)
        return result_dataframe
    
    def tree_selection(self, dataframe):
        """
        Applies tree-based feature selection using Extra Trees classifier.
        Features are ranked by their importance scores from the ensemble model.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            DataFrame with features selected by Extra Trees importance
        """
        logger.info("Starting Tree-based feature selection using Extra Trees...")
        temp, data = self.split_dataframe(dataframe)
        logger.info("Training Extra Trees classifier with 50 estimators...")
        clf = ExtraTreesClassifier(n_estimators=50, random_state=42)
        clf.fit(data, temp["ards"])
        model = SelectFromModel(clf, prefit=True)
        result = model.fit_transform(data)
        selected_features = model.get_feature_names_out(data.columns)
        logger.info(f"Tree-based selection retained {len(selected_features)} features")
        result_dataframe_temp = pd.DataFrame(result, columns=selected_features, index=data.index)
        result_dataframe = pd.concat([temp, result_dataframe_temp], axis=1)
        return result_dataframe
    
    def sequential_selection(self, dataframe):
        """
        Applies Sequential Feature Selection (Forward/Backward) using KNN classifier.
        Iteratively adds or removes features to optimize model performance.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            DataFrame with features selected by sequential algorithm
        """
        logger.info("Starting Sequential Feature Selection using KNN (k=3)...")
        temp, data = self.split_dataframe(dataframe)
        knn = KNeighborsClassifier(n_neighbors=3)
        logger.info("Fitting sequential feature selector (selecting 3 features)...")
        sfs = SequentialFeatureSelector(knn, n_features_to_select=3)
        sfs.fit(data, temp["ards"])
        result = sfs.fit_transform(data, temp["ards"])
        selected_features = sfs.get_feature_names_out(data.columns)
        logger.info(f"Sequential selection selected features: {list(selected_features)}")
        result_dataframe_temp = pd.DataFrame(result, columns=selected_features, index=data.index)
        result_dataframe = pd.concat([temp, result_dataframe_temp], axis=1)
        return result_dataframe

    def split_dataframe(self, dataframe):
        """
        Splits the dataframe into metadata and feature columns.
        Metadata columns (ards, patient_id, time) are preserved and not subject to feature selection.
        
        Args:
            dataframe: Input DataFrame
            
        Returns:
            Tuple of (metadata_df, features_df) where metadata_df contains ards/patient_id/time
            and features_df contains all other numeric features
        """
        df1 = dataframe[["ards", "patient_id", "time"]]
        df2 = dataframe.drop(columns=["patient_id", "ards", "time"])
        logger.debug(f"DataFrame split: {len(df1.columns)} metadata columns, {len(df2.columns)} feature columns")
        return df1, df2
    
    def set_selection_method(self, method):
        """
        Sets the feature selection method to use. Validates that the method is available.
        
        Args:
            method: Name of the feature selection method
            
        Raises:
            RuntimeError: If the specified method is not in the available methods list
        """
        if method not in self.available_methods:
            error_msg = f"Feature selection method '{method}' not available. Available methods are: {self.available_methods}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        self.feature_selection_method = method
        logger.info(f"Feature selection method set to: {method}")