from enum import Enum


class ColumnType(str, Enum):
    Categorical = "Categorical"
    Numerical = "Numerical"
    Textual = "Textual"
    TimeStamp = "TimeStamp"
    Auto = "Auto"
    Empty = "Empty"
