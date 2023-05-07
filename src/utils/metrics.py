import pandas as pd
import math
import numpy as np
import os
from typing import Tuple
from classes.evaluation import Evaluation
from sklearn.metrics import roc_auc_score, log_loss
from config.file_config import file_config


class MetricAtK(object):
    def __init__(self, base_dir: str, top_k: int = 10):
        self._interaction_dir = os.path.join(base_dir, file_config['data_dir_name'], file_config['preprocessed_dir_name'])
        self._top_k = top_k

    def evaluate(self, test_pred: np.ndarray) -> Evaluation:
        full = self._full(test_pred)
        loss = log_loss(y_true=full['label'].values, y_pred=full['score'].values)
        auc = roc_auc_score(y_true=full['label'].values, y_score=full['score'].values)
        hit_ratio = self._calc_hit_ratio(full)
        map = self._calc_map(full)
        ndcg = self._calc_ndcg(full)
        mrr = self._calc_mrr(full)
        return Evaluation(loss, auc, hit_ratio, map, ndcg, mrr)

    def _full(self, test_pred: np.ndarray) -> pd.DataFrame:
        full = self._get_prediction_table(test_pred, 'test')
        pos = full[full['label'] == 1]
        pos = pos.copy()
        pos.rename(columns={'item_id': 'pos_item_id', 'score': 'pos_score'}, inplace=True)
        pos.drop(columns=['label'], inplace=True)
        full = pd.merge(full, pos, on='base_id', how='left')
        full['rank'] = full.groupby('base_id')['score'].rank(method='first', ascending=False)
        full.sort_values(['base_id', 'rank'], inplace=True)
        return full

    def _get_prediction_table(self, preds, pred_type='test') -> pd.DataFrame:
        """
        base_id can be user_id or item_id
        """
        # assert pred_type in ['test', evaluate', 'negative']
        base_ids, item_ids, labels = self._get_ids_and_labels(pred_type)
        return pd.DataFrame({'base_id': base_ids, 'item_id': item_ids, 'label': labels, 'score': preds})

    def _get_ids_and_labels(self, pred_type) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        files = [os.path.join(self._interaction_dir, f) for f
                 in os.listdir(self._interaction_dir) if pred_type in f]
        ids_and_labels = None
        for f in files:
            interaction_table = pd.read_hdf(f, mode='r')
            if ids_and_labels is None:
                ids_and_labels = interaction_table.values
            else:
                ids_and_labels = np.concatenate(ids_and_labels, interaction_table.values)
        return ids_and_labels[:, 0], ids_and_labels[:, 1], ids_and_labels[:, 2].astype('int32')

    def _calc_hit_ratio(self, full: pd.DataFrame) -> float:
        top_k = full[full['rank'] <= self._top_k]
        test_in_top_k = top_k[top_k['pos_item_id'] == top_k['item_id']]  # golden items hit in the top_K items
        return len(test_in_top_k) * 1.0 / full['base_id'].nunique()

    def _calc_ndcg(self, full: pd.DataFrame) -> float:
        top_k = full[full['rank'] <= self._top_k]
        test_in_top_k = top_k[top_k['pos_item_id'] == top_k['item_id']]
        test_in_top_k = test_in_top_k.copy()
        test_in_top_k['ndcg'] = test_in_top_k['rank'].apply(
            lambda x: math.log(2) / math.log(1 + x))  # the rank starts from 1
        return test_in_top_k['ndcg'].sum() * 1.0 / full['base_id'].nunique()

    def _calc_mrr(self, full: pd.DataFrame) -> float:
        top_k = full[full['rank'] <= self._top_k]
        test_in_top_k = top_k[top_k['label'] == 1]
        test_in_top_k = test_in_top_k.copy()
        test_in_top_k['mrr'] = test_in_top_k['rank'].apply(lambda x: 1 / x)
        return test_in_top_k['mrr'].sum() * 1.0 / full['base_id'].nunique()

    def _calc_map(self, full: pd.DataFrame) -> float:
        def calc_ap(min_rank):
            return sum([1/(i+1) for i in range(self._top_k) if (i+1) >= min_rank]) / self._top_k

        top_k = full[full['rank'] <= self._top_k]
        test_in_top_k = top_k[top_k['label'] == 1].groupby('base_id').min()
        # normalize by calc_ap(1), sample has only one positive sample (leave one out sampling)
        # return test_in_top_k['rank'].apply(calc_ap).sum() * 1.0 / full['base_id'].unique()
        norm_value = calc_ap(1)
        return test_in_top_k['rank'].apply(calc_ap).apply(
            lambda x: x / norm_value).sum() * 1.0 / full['base_id'].nunique()
