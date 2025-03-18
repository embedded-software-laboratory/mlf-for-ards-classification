import pandas as pd
import numpy as np

import tensorflow.keras as keras
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class WindowGenerator:

    def __init__(self, data: pd.DataFrame, **kwargs):
        self.input_width = int(kwargs.get('input_width', 10))
        self.output_width = int(kwargs.get('output_width', 1))
        self.batch_size = int(kwargs.get('batch_size', 32))
        self.shift = int(kwargs.get('shift', 1))
        self.label_columns = list(kwargs.get('label_columns', []))
        if self.label_columns:
            self.label_columns_indices = {name: i for i, name in enumerate(self.label_columns)}
        self.column_indices = {name: i for i, name in enumerate(data.columns)}
        self.total_window_size = self.input_width + self.shift


