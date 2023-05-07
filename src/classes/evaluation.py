from enum import Enum


class EvaluationType(Enum):
    Loss = 'loss'
    AUC = 'auc'
    HitRatio = 'hit_ratio'
    MAP = 'map'
    NDCG = 'ndcg'
    MRR = 'mrr'


class Evaluation:
    def __init__(self, loss=0, auc=0, hit_ratio=0, map=0, ndcg=0, mrr=0):
        self.loss = loss
        self.auc = auc
        self.hit_ratio = hit_ratio
        self.map = map
        self.ndcg = ndcg
        self.mrr = mrr

    def get_metric_from_evaluation_type(self, evaluation_type: EvaluationType):
        return self.__getattribute__(evaluation_type.value)

    def __str__(self):
        return ''.join([f'{key} = {value}, 'for key, value in self.__dict__.items()])


