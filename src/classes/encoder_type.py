from enum import Enum


class EncoderType(str, Enum):
    OneHot = "one_hot"
    Label = "label"
    KBins = 'k_bins'
    Norm = "norm"
    Nothing = "nothing"
    CatBoost = "cat_boost"

