import pandas as pd
import numpy as np

import tensorflow.keras as keras
import tensorflow as tf



class WindowGenerator:

    def __init__(self, data: pd.DataFrame, **kwargs):
        self.data = data
        self.input_width = int(kwargs.get('input_width', 10))
        self.output_width = int(kwargs.get('output_width', 1))
        self.batch_size = int(kwargs.get('batch_size', 32))
        self.shift = int(kwargs.get('shift', 1))
        self.label_columns = list(kwargs.get('label_columns', []))
        if self.label_columns:
            self.label_columns_indices = {name: i for i, name in enumerate(self.label_columns)}
        self.column_indices = {name: i for i, name in enumerate(data.columns)}
        self.total_window_size = self.input_width + self.shift
        assert self.shift > 0
        assert self.shift < self.total_window_size
        assert self.shift >= self.output_width

        self.input_slice = slice(0, self.input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.output_start = self.total_window_size - self.output_width
        self.output_slice = slice(self.output_start, None)
        self.output_indices = np.arange(self.total_window_size)[self.output_slice]

    def generate_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.output_slice, :]
        if self.label_columns:
            labels = tf.stack([labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.output_width, None])
        return inputs, labels

    def generate_dataset(self):
        data = np.array(self.data, dtype=np.float32)
        if data.shape[0] <= self.total_window_size:
            return None

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=self.batch_size,
        )
        ds = ds.map(self.generate_window)
        return ds


