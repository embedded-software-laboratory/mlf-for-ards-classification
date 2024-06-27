import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

class Feature_analysis:

    def __init__(self, timeseries_classes, config):
        #cross validation params
        pass

    def calculate_vif(self, dataframe):
        print("Calculating VIF...")
        vif_data = pd.DataFrame()
        # create empty pandas dataframe
        vif_data["Feature"] = dataframe.predictors
        # assign columns
        vif_data["VIF"] = [variance_inflation_factor(dataframe.values, i) for i in range(dataframe.shape[1])]
        # For each column index i, the VIF is calculated using the variance_inflation_factor function from the statsmodels.stats.outliers_influence module, applied to the entire input data (data.values) and the index i.
        return vif_data
        print(vif_data)