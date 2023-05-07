from enum import Enum


class FeatureType(str, Enum):
    USER = "user"
    ITEM = "item"
    ALL = "all"
    RAW = "raw"
