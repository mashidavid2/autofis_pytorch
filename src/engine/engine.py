import abc
import pandas as pd
import numpy as np
import os
from typing import Optional
from classes import ModelInfo, ColumnInfo, Evaluation, FeatureInfo
from utils import MetricAtK, EarlyStopping
from dataset import DatasetGenerator
from config.file_config import file_config
from utils import Timer


class Engine:
    def __init__(self, model_info: ModelInfo, column_info: ColumnInfo, base_dir: str, save_dir: str):
        self.model_info = model_info
        self.column_info = column_info
        self.base_dir = base_dir
        self.save_dir = save_dir
        self.feature_dir = os.path.join(base_dir, file_config['data_dir_name'], file_config['feature_dir_name'])
        self.save_postfix = file_config['check_point_postfix']
        self._metric = MetricAtK(base_dir=base_dir, top_k=10)
        self.feature_info: Optional[FeatureInfo] = None
        self.early_stopping: Optional[EarlyStopping] = None
        self.timer: Optional[Timer] = None

    def train(self, dataset_generator: DatasetGenerator) -> Evaluation:
        self._load_feature_info()
        evaluation = self._train(dataset_generator)
        if self.early_stopping is None:
            self._save()
        return evaluation

    def inference(self, feature: pd.DataFrame) -> np.ndarray:
        self._load()
        scores = self._inference(feature)
        return scores

    def _load_feature_info(self):
        """
        load feature_info generate from featurization for model parameter
        """
        self.feature_info = FeatureInfo.load(self.feature_dir, self.model_info.model_name.name)

    def _print_epoch_loss(self, epoch_id, test_loss):
        print('[Epoch {}], Loss {}'.format(epoch_id, test_loss))

    @abc.abstractmethod
    def _train(self, dataset_generator: DatasetGenerator) -> Evaluation:
        raise NotImplementedError('train must define in each engine')

    @abc.abstractmethod
    def _evaluate(self, test_gen: DatasetGenerator) -> Evaluation:
        raise NotImplementedError('evaluation must define in each engine')

    @abc.abstractmethod
    def _save(self):
        raise NotImplementedError('save must define in each engine')

    @abc.abstractmethod
    def _load(self):
        raise NotImplementedError('load must define in each engine')

    def _close(self):
        """implemented only for tensorflow v1 model that needs to close session"""
        pass

    @abc.abstractmethod
    def _inference(self, feature: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError('inference must define in each engine')






