from dataclasses import dataclass
from enum import Enum


class RecommendationType(str, Enum):
    USER2ITEM = 'USER2ITEM'
    ITEM2ITEM = 'ITEM2ITEM'
