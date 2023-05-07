import os
import pandas as pd
from config.file_config import file_config


class BatchGenerator:
    def __init__(self, dataset, kwargs):
        self.dataset = dataset
        self.kwargs = kwargs
        assert kwargs['gen_type'] in ['train', 'test', 'evaluate', 'negative']

    def __iter__(self):
        for x in self.dataset.__iter__(**self.kwargs):
            yield x

    @property
    def batch_size(self):
        return self.kwargs['batch_size']

    @property
    def gen_type(self):
        return self.kwargs['gen_type']


class DatasetGenerator:
    def __init__(self, base_dir):
        self.feature_dir = os.path.join(base_dir, file_config['data_dir_name'], file_config['feature_dir_name'])
        self.input_feature_prefix = file_config['input_feature_prefix']
        self.output_feature_prefix = file_config['output_feature_prefix']

    def __iter__(self, gen_type='train', batch_size=None, neg_ratio=None, shuffle=True, squeeze_output=True):
        assert gen_type in ['train', 'test', 'evaluate', 'negative']
        if gen_type in ['test', 'evaluate', 'negative']:
            """for evaluation, user_id, item_id orders must be maintained"""
            shuffle = False

        def _iter_():
            for hdf_in, hdf_out in self._files_iter(gen_type=gen_type, shuffle=shuffle):
                with pd.HDFStore(hdf_in, mode='r') as hdf_in, pd.HDFStore(hdf_out, mode='r') as hdf_out:
                    X_all = pd.read_hdf(hdf_in, mode='r').values
                    Y_all = pd.read_hdf(hdf_out, mode='r').values
                    yield X_all, Y_all, hdf_in

        for X_all, Y_all, block in _iter_():
            if neg_ratio:
                pass
            else:
                gen = self.generator(X_all, Y_all, batch_size, shuffle)
                for X, y in gen:
                    if squeeze_output:
                        y = y.squeeze()
                    yield X, y

    @staticmethod
    def generator(X, y, batch_size, shuffle=True):
        import numpy as np
        if batch_size is None:
            yield X, y
        else:
            num_of_batches = int(np.ceil(X.shape[0] * 1.0 / batch_size))
            sample_index = np.arange(X.shape[0])
            if shuffle:
                np.random.shuffle(sample_index)
            assert X.shape[0] > 0
            for i in range(num_of_batches):
                batch_index = sample_index[batch_size * i: batch_size * (i+1)]
                X_batch = X[batch_index]
                y_batch = y[batch_index]
                yield X_batch, y_batch

    def _files_iter(self, gen_type='train', shuffle=False):
        files = [os.path.join(self.feature_dir, f) for f in os.listdir(self.feature_dir)
                 if f'{self.input_feature_prefix}_{gen_type}' in f]
        for f in files:
            yield f, f.replace(self.input_feature_prefix, self.output_feature_prefix)

    def batch_generator(self, kwargs):
        return BatchGenerator(self, kwargs)
