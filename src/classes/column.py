from dataclasses import dataclass
from classes.column_type import ColumnType


@dataclass
class Column:
    name: str
    type: ColumnType
