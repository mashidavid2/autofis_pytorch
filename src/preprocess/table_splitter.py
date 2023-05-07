import os
import pandas as pd
import random
from copy import deepcopy
from typing import List, Optional
# from src.classes import Column, ColumnType, ColumnInfo
# from src.config.file_config import file_config
from classes import Column, ColumnType, ColumnInfo
from config.file_config import file_config


class TableSplitter(object):
    def __init__(self, base_dir: str, column_info: ColumnInfo, interaction_file_path: str,
                 user_file_path: str = None, item_file_path: str = None):
        self.preprocessed_dir = os.path.join(base_dir, file_config['data_dir_name'], file_config['preprocessed_dir_name'])
        self.column_info = column_info
        self.user_file_name = file_config['user_file_name']
        self.item_file_name = file_config['item_file_name']
        self.interaction_table_prefix = file_config['interaction_table_prefix']
        self.hdf5_postfix = file_config['hdf5_postfix']
        self.block_size = file_config['block_size']

        self.interaction_file_path = interaction_file_path
        self.user_file_path = user_file_path
        self.item_file_path = item_file_path

    def split_to_user_item_interaction_table(self):
        self._directory()
        case = self._check_file_configuration()

        if case == 1:
            user_table = pd.read_csv(self.user_file_path)
            item_table = pd.read_csv(self.item_file_path)

            pds = pd.read_csv(self.interaction_file_path, iterator=True, chunksize=self.block_size)
            for idx, interaction_table_chunk in enumerate(pds):
                interaction_table_chunk = self._get_cleaned_data(interaction_table_chunk)
                self._validate_and_save_interaction_table(
                    interaction_table_chunk.copy(), f'{self.interaction_table_prefix}_part_{idx}', user_table, item_table)
        else:
            pds = pd.read_csv(self.interaction_file_path, iterator=True, chunksize=self.block_size)
            user_table = None
            item_table = None
            for idx, chunk in enumerate(pds):
                chunk = self._get_cleaned_data(chunk)
                user_table = self._get_user_table(chunk[self.column_info.get_user_col_names()], user_table)
                item_table = self._get_item_table(chunk[self.column_info.get_item_col_names()], item_table)

                self._validate_and_save_interaction_table(
                    chunk[self.column_info.get_interaction_col_names_with_temp()].copy(),
                    f'{self.interaction_table_prefix}_part_{idx}', user_table, item_table)

        self._validate_and_save_feature_table(
            user_table, self.user_file_name, self.column_info.user_id_column, self.column_info.user_feature_columns)
        self._validate_and_save_feature_table(
            item_table, self.item_file_name, self.column_info.item_id_column, self.column_info.item_feature_columns)

    def _directory(self):
        os.makedirs(self.preprocessed_dir, exist_ok=True)
        file_list = os.listdir(self.preprocessed_dir)
        for f in file_list:
            os.remove(os.path.join(self.preprocessed_dir, f))

    def _check_file_configuration(self):
        case = 0
        """
        0: case for user, item, interaction data in single interaction(log) table 
        1: case for each user, item, interaction(log) table exist separately  
        """
        if (self.user_file_path is not None) and (self.item_file_path is not None):
            case = 1
        else:
            assert AttributeError('user, item table must exist at the same time')

        if self.interaction_file_path is None:
            assert AttributeError('interaction table must exist')
        return case

    def _get_cleaned_data(self, data: pd.DataFrame) -> pd.DataFrame:
        virtual_rating_df = self._get_virtual_rated_data(data)
        binarized_df = self._get_binarized_data(virtual_rating_df)
        temporal_df = self._get_temporal_processed_data(binarized_df)
        return temporal_df

    def _get_virtual_rated_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.column_info.exist_rating():
            col_rating = self.column_info.get_rating_name()
            virtual_rating_df = data.drop_duplicates()
            virtual_rating_df[col_rating] = virtual_rating_df.value_counts().values
            return virtual_rating_df
        return data

    def _get_binarized_data(self, data: pd.DataFrame) -> pd.DataFrame:
        col_rating_name = self.column_info.get_rating_name()
        data.loc[data[col_rating_name] > 0, col_rating_name] = 1
        return data

    def _get_temporal_processed_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.column_info.exist_temporal():
            tem_col = self.column_info.get_temporal_name()
            data[tem_col] = [random.random() for _ in range(len(data))]
            return data
        return data

    def _get_item_table(self, item_data: pd.DataFrame, other_table: Optional[pd.DataFrame] = None):
        item_table = deepcopy(item_data)
        item_table.drop_duplicates(inplace=True)
        if other_table is not None:
            item_table = item_table.append(other_table)
            item_table.drop_duplicates(subset=[self.column_info.get_item_name()], keep='first', inplace=True)
        item_table.sort_values(by=self.column_info.get_item_name(), inplace=True)
        item_table.reset_index(inplace=True, drop=True)
        return item_table

    def _get_user_table(self, user_data: pd.DataFrame, other_table: Optional[pd.DataFrame] = None):
        user_table = deepcopy(user_data)
        user_table.drop_duplicates(inplace=True)
        if other_table is not None:
            user_table = user_table.append(other_table)
            user_table.drop_duplicates(subset=[self.column_info.get_user_name()], keep='first', inplace=True)
        user_table.sort_values(by=self.column_info.get_user_name(), inplace=True)
        user_table.reset_index(inplace=True, drop=True)
        return user_table

    def _validate_and_save_interaction_table(
            self, interaction_table: pd.DataFrame, table_name: str, user_table: pd.DataFrame, item_table: pd.DataFrame):
        self._validate_interaction_table(interaction_table, user_table, item_table)
        self._convert_categorical_value_to_str(
            interaction_table, [self.column_info.user_id_column, self.column_info.item_id_column])
        self._save_table_to_hdf(interaction_table, table_name)

    def _validate_interaction_table(self, interaction_table: pd.DataFrame, user_table: pd.DataFrame, item_table: pd.DataFrame):
        user_set = set(user_table[self.column_info.get_user_name()].unique())
        item_set = set(item_table[self.column_info.get_item_name()].unique())
        user_set_in_interaction_table = set(interaction_table[self.column_info.get_user_name()].unique())
        item_set_in_interaction_table = set(interaction_table[self.column_info.get_item_name()].unique())

        not_exist_in_user_set = str(user_set_in_interaction_table - user_set)
        not_exist_in_item_set = str(item_set_in_interaction_table - item_set)

        assert user_set_in_interaction_table.issubset(user_set), \
                f'user_id data (column {self.column_info.get_user_name()}) must be in user_table\n ' \
                f'these user_id data that do not exist in user_table: ({not_exist_in_user_set})'

        assert item_set_in_interaction_table.issubset(item_set), \
                f'user_id data (column {self.column_info.get_item_name()}) must be in user_table\n' \
                f'these user_id data that do not exist in user_table: ({not_exist_in_item_set})'

    def _validate_and_save_feature_table(self, table: pd.DataFrame, table_name: str, id_column: Column,
                                         feature_columns: List[Column]):
        if table_name == self.user_file_name:
            pass
        self._validate_id_column(table, id_column)
        self._validate_feature_columns(table, feature_columns)
        self._handle_missing_value(table, feature_columns)
        self._convert_categorical_value_to_str(table, feature_columns + [id_column])
        self._save_table_to_hdf(table, table_name)

    def _validate_id_column(self, table: pd.DataFrame, id_column: Column):
        assert table[id_column.name].nunique() == len(table), \
                f'id column ({id_column.name} must be primary key'

        assert not table[id_column.name].isnull().values.any(), \
            f'id column ({id_column.name}) must do not have null value'

    def _validate_feature_columns(self, table: pd.DataFrame, feature_columns: List[Column]):
        for column in feature_columns:
            if column.type == ColumnType.Numerical:
                assert table[column.name].dtype == int or table[column.name].dtype == float, \
                    f'feature column ({column.name}) type is not {column.type}, but {table[column.name].dtype}'
            if column.type == ColumnType.Categorical:
                pass

    def _handle_missing_value(self, table: pd.DataFrame, feature_columns: Optional[List[Column]]):
        for column in feature_columns:
            if column.type == ColumnType.Numerical:
                table[column.name].fillna(table[column.name].mean(), inplace=True)
            elif column.type == ColumnType.Categorical:
                """categorical data can be handled in feature encoder (feature transformer step)"""
                pass

    def _convert_categorical_value_to_str(self, table: pd.DataFrame, columns: List[Column]):
        for column in columns:
            if column.type == ColumnType.Categorical and (table[column.name].dtype == int or table[column.name].dtype == float):
                table.loc[:, column.name] = table[column.name].apply(str).replace({'nan': None})

    def _save_table_to_hdf(self, table: pd.DataFrame, table_name: str):
        table.to_hdf(os.path.join(self.preprocessed_dir, f'{table_name}.{self.hdf5_postfix}'), key='fixed', index=False)

    def clean_files(self):
        file_list = os.listdir(self.preprocessed_dir)
        for f in file_list:
            if file_config['user_file_name'] in f:
                continue
            if file_config['item_file_name'] in f:
                continue
            os.remove(os.path.join(self.preprocessed_dir, f))
