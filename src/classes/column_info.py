from dataclasses import dataclass
from dataclasses_json import dataclass_json
from classes.column import Column
from classes.column_type import ColumnType
from typing import List


@dataclass_json
@dataclass
class ColumnInfo:
    rating_column: Column
    user_id_column: Column
    user_feature_columns: List[Column]
    item_id_column: Column
    item_feature_columns: List[Column]
    timestamp_column: Column

    def __post_init__(self):
        """
        id column들은 반드시 Categorical type이여야 하는데, user가 잘못 넣었을 경우 handling
        """
        self.user_id_column.type = ColumnType.Categorical
        self.item_id_column.type = ColumnType.Categorical

    def __str__(self):
        return ''.join([f'{key} = {value}, ' for key, value in self.__dict__.items()])

    # TODO Column에 USER, ITEM type 정보 넣을지 말지
    def get_categorical_columns(self) -> list:
        return [col.name for col in self.user_feature_columns + self.item_feature_columns if col.type == ColumnType.Categorical]

    def get_numeric_columns(self) -> list:
        return [col.name for col in self.user_feature_columns + self.item_feature_columns if col.type == ColumnType.Numerical]

    def get_user_categorical_columns(self) -> list:
        return [col.name for col in self.user_feature_columns if col.type == ColumnType.Categorical]

    def get_user_numerical_columns(self) -> list:
        return [col.name for col in self.user_feature_columns if col.type == ColumnType.Numerical]

    def get_item_categorical_columns(self) -> list:
        return [col.name for col in self.item_feature_columns if col.type == ColumnType.Categorical]

    def get_item_numerical_columns(self) -> list:
        return [col.name for col in self.item_feature_columns if col.type == ColumnType.Numerical]

    def get_user_name(self) -> str:
        return self.user_id_column.name

    def get_item_name(self) -> str:
        return self.item_id_column.name

    def get_rating_name(self) -> str:
        if self.rating_column.name == '':
            return 'VRating'
        return self.rating_column.name

    def get_temporal_name(self) -> str:
        # if self.timestamp_column.name == '':
        #     return 'timestamp'
        # return self.timestamp_column.name
        return 'timestamp'

    def exist_temporal(self) -> bool:
        # return self.timestamp_column.name != ''
        return False

    def exist_rating(self) -> bool:
        return self.rating_column.name != ''

    def get_user_feature_names(self) -> list:
        return self.get_user_numerical_columns() + self.get_user_categorical_columns()

    def get_item_feature_names(self) -> list:
        return self.get_item_numerical_columns() + self.get_item_categorical_columns()

    def get_user_col_names(self) -> list:
        return [self.get_user_name()] + self.get_user_feature_names()

    def get_item_col_names(self) -> list:
        return [self.get_item_name()] + self.get_item_feature_names()

    def get_interaction_col_names(self) -> list:
        return [self.get_user_name(), self.get_item_name(), self.get_rating_name()]

    def get_interaction_col_names_with_temp(self) -> list:
        return [self.get_user_name(), self.get_item_name(), self.get_rating_name(), self.get_temporal_name()]

    def get_feature_names(self) -> list:
        return self.get_user_feature_names() + self.get_item_feature_names()

    def get_feature_names_with_id_columns(self) -> list:
        return self.get_user_col_names() + self.get_item_col_names()

    def get_train_col_names(self) -> list:
        cols = list()
        cols.append(self.get_user_name())
        cols.extend(self.get_user_feature_names())
        cols.append(self.get_item_name())
        cols.extend(self.get_item_feature_names())
        cols.append(self.get_rating_name())
        return cols

    def get_inference_cols_names(self) -> list:
        cols = list()
        cols.append(self.get_user_name())
        cols.extend(self.get_user_feature_names())
        cols.append(self.get_item_name())
        cols.extend(self.get_item_feature_names())
        return cols
