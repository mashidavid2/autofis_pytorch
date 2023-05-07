from typing import Optional
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, sum_models
from sklearn.metrics import log_loss
from engine import Engine
from classes import CatBoostInfo, ColumnInfo, Evaluation
from config.file_config import file_config
from utils import Timer
import datetime


class CatBoostEngine(Engine):
    def __init__(self, model_info: CatBoostInfo, column_info: ColumnInfo, base_dir: str):
        super(CatBoostEngine, self).__init__(
            model_info, column_info, base_dir)
        self.model_info = model_info
        self.model: Optional[CatBoostClassifier] = None
        self.crit = log_loss

    def _train(self, dataset_generator) -> Evaluation:
        num_block = int(np.ceil(self.feature_info.train_size / file_config['block_size']))
        unit_block = 1
        self.timer = Timer(
            train_size=self.feature_info.train_size,
            batch_size=file_config['block_size'],
            epoch_size=1, # catboost sub model train whole iteration at once
            unit_batch=unit_block
        )
        self.timer.start()

        train_param = {
            'gen_type': 'train',
            # 'batch_size': self.model_info.batch_size,
            'batch_size': file_config['block_size'],
            'squeeze_output': False
        }
        train_gen = dataset_generator.batch_generator(train_param)

        test_param = {
            'gen_type': 'test',
            'batch_size': self.model_info.batch_size,
            'squeeze_output': False,
            'shuffle': False
        }
        test_gen = dataset_generator.batch_generator(test_param)

        cat_cols = self.column_info.get_categorical_columns()
        models = []
        block_id = 0

        for block_id, block_data in enumerate(train_gen):
            X, y = block_data
            X = pd.DataFrame(X, columns=self.column_info.get_feature_names())
            y = pd.DataFrame(y, columns=[self.column_info.get_rating_name()])

            self.model = CatBoostClassifier(
                iterations=self.model_info.epoch,
                learning_rate=self.model_info.learning_rate,
                depth=self.model_info.depth,
                allow_writing_files=False
            )

            self.model.fit(X, y, cat_features=cat_cols, verbose=False)
            block_loss = self.model.best_score_['learn']['Logloss']

            elapsed, eta = self.timer(stage='train')
            print(
                f'[Sub Model Train statistics] model: {block_id + 1}/{num_block}, '
                f'training_loss: {block_loss}, '
                f'elapsed: {str(datetime.timedelta(seconds=elapsed))}, ETA: {str(datetime.timedelta(seconds=eta))}'
            )

            if (block_id + 1) % unit_block == 0:
                evaluation = self._evaluate(test_gen)

                elapsed, eta = self.timer(stage='test')
                print(
                    f'[Sub Model Test statistics] model: {block_id + 1}/{num_block}, evaluation: {str(evaluation)}'
                    f'elapsed: {str(datetime.timedelta(seconds=elapsed))}, ETA: {str(datetime.timedelta(seconds=eta))}\n'
                )
            models.append(self.model)

        self.model = sum_models(models, weights=[1/len(models)] * len(models))

        evaluation = self._evaluate(test_gen)
        elapsed, eta = self.timer(stage='test')

        print(
            f'[Full Model Test statistics] model: {block_id + 1}/{num_block}, evaluation: {str(evaluation)}'
            f'elapsed: {str(datetime.timedelta(seconds=elapsed))}, ETA: {str(datetime.timedelta(seconds=eta))}\n'
        )
        return evaluation

    def _evaluate(self, test_gen) -> Evaluation:
        test_preds = []
        for batch_data in test_gen:
            X, y = batch_data
            X = pd.DataFrame(X, columns=self.column_info.get_feature_names())
            batch_pred = self.model.predict(X, prediction_type='Probability')
            test_preds.append(batch_pred[:, 1])
        test_preds = np.concatenate(test_preds)
        test_preds = np.float64(test_preds)
        test_preds = np.clip(test_preds, 1e-8, 1 - 1e-8)
        return self._metric.evaluate(test_preds)

    def _save(self):
        import os
        self.model.save_model(
            fname=os.path.join(self.base_dir, f'{self.model_info.model_name}.{self.save_postfix}'))

    def _load(self):
        import os
        self.model = CatBoostClassifier()
        self.model.load_model(
            fname=os.path.join(self.base_dir, f'{self.model_info.model_name}.{self.save_postfix}'))

    def _inference(self, feature: pd.DataFrame) -> np.ndarray:
        return self.model.predict(feature, prediction_type='Probability')[:, 1]

