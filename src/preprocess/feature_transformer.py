import os
import pandas as pd
from typing import Tuple
# from src.classes import ColumnInfo, ModelInfo, ModelName, FeatureType, FeatureInfo
# from src.config.file_config import file_config
# from src.preprocess import FeatureEncoder
from classes import ColumnInfo, ModelInfo, ModelName, FeatureType, FeatureInfo
from config.file_config import file_config
from preprocess import FeatureEncoder

class FeatureTransformer(object):
    def __init__(self, base_dir: str, column_info: ColumnInfo, model_info: ModelInfo):
        self.preprocess_dir = os.path.join(base_dir, file_config['data_dir_name'], file_config['preprocessed_dir_name'])
        self.feature_dir = os.path.join(base_dir, file_config['data_dir_name'], file_config['feature_dir_name'])
        self.user_file_name = file_config['user_file_name']
        self.item_file_name = file_config['item_file_name']
        self.interaction_table_prefix = file_config['interaction_table_prefix']
        self.hdf5_postfix = file_config['hdf5_postfix']
        self.column_info = column_info
        self.model_info = model_info
        self.feature_type = model_info.feature_type
        self.feature_encoder = FeatureEncoder(self.column_info, model_info, model_info.feature_type)
        self.feature_info = FeatureInfo()

    def _directory(self):
        os.makedirs(self.feature_dir, exist_ok=True)
        file_list = os.listdir(self.feature_dir)
        for f in file_list:
            os.remove(os.path.join(self.feature_dir, f))

    def transform_to_file_for_training(self):
        self._directory()
        self._fit()
        self._transform_to_file()
        self._save_feature_info()

    def get_raw_and_transformed_feature(self, base_id) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._fit()
        return self._get_raw_and_transformed_feature(base_id)

    def _fit(self):
        if self.feature_type == FeatureType.RAW:
            """
            CatBoost model itself transform data feature,
            so just generate negative sample data     
            """
            return
        user_table = None
        item_table = None

        if self.feature_type == FeatureType.ALL or self.feature_type == FeatureType.USER:
            user_table = pd.read_hdf(os.path.join(self.preprocess_dir, f'{self.user_file_name}.{self.hdf5_postfix}'))
        if self.feature_type == FeatureType.ALL or self.feature_type == FeatureType.ITEM:
            item_table = pd.read_hdf(os.path.join(self.preprocess_dir, f'{self.item_file_name}.{self.hdf5_postfix}'))
        self.feature_encoder.fit(user_table, item_table)

        self.feature_info.feature_size_dict = self.feature_encoder.feature_size_dict

    def _save_feature_info(self):
        self.feature_info.save_to_pickle(self.feature_dir, self.model_info.model_name.name)

    def _transform_to_file(self):

        def update_feature_info(feature: pd.DataFrame, file_name):
            if 'train' in file_name:
                self.feature_info.train_size += len(feature)
            elif 'test' in file_name:
                self.feature_info.test_size += len(feature)
            elif 'evaluate' in file_name:
                self.feature_info.evaluate_size += len(feature)
            elif 'negative' in file_name:
                self.feature_info.negative_size += len(feature)

        if self.feature_type == FeatureType.ALL:
            user_table = pd.read_hdf(os.path.join(self.preprocess_dir, f'{self.user_file_name}.{self.hdf5_postfix}'))
            item_table = pd.read_hdf(os.path.join(self.preprocess_dir, f'{self.item_file_name}.{self.hdf5_postfix}'))
            assert isinstance(user_table, pd.DataFrame) and isinstance(item_table, pd.DataFrame)

            interaction_table_files = [f_in for f_in in os.listdir(self.preprocess_dir)
                                       if self.interaction_table_prefix in f_in]

            for f in interaction_table_files:
                interaction_table = pd.read_hdf(os.path.join(self.preprocess_dir, f))
                assert isinstance(interaction_table, pd.DataFrame)

                data = pd.merge(interaction_table, user_table, on=self.column_info.get_user_name(), how='left')[
                    self.column_info.get_user_col_names()]
                user_feature = self.feature_encoder.transform(data) #merge user features and interaction features

                data = pd.merge(interaction_table, item_table, on=self.column_info.get_item_name(), how='left')[
                    self.column_info.get_item_col_names()]
                item_feature = self.feature_encoder.transform(data) #merge item features and interaction features

                feature = pd.concat((user_feature, item_feature), axis=1) #concat merged user features and merged item features

                update_feature_info(feature, f)

                if self.model_info.model_name is ModelName.AutoFis:
                    base_dims = [sum(self.feature_info.feature_dims[0:i]) for i
                                 in range(self.feature_info.feature_nums)]
                    feature = feature + base_dims

                feature.to_hdf(os.path.join(self.feature_dir,
                                            f.replace(file_config['interaction_table_prefix'],
                                                      file_config['input_feature_prefix'])),
                               key='fixed', index=False)
                interaction_table[self.column_info.get_rating_name()].to_hdf(
                    os.path.join(self.feature_dir, f.replace(
                        file_config['interaction_table_prefix'],
                        file_config['output_feature_prefix'])),
                    key='fixed', index=False)

        elif self.feature_type == FeatureType.RAW:
            user_table = pd.read_hdf(os.path.join(self.preprocess_dir, f'{self.user_file_name}.{self.hdf5_postfix}'))
            item_table = pd.read_hdf(os.path.join(self.preprocess_dir, f'{self.item_file_name}.{self.hdf5_postfix}'))
            assert isinstance(user_table, pd.DataFrame) and isinstance(item_table, pd.DataFrame)

            interaction_table_files = [f_in for f_in in os.listdir(self.preprocess_dir) if
                                       self.interaction_table_prefix in f_in]

            for f in interaction_table_files:
                interaction_table = pd.read_hdf(os.path.join(self.preprocess_dir, f))
                assert isinstance(interaction_table, pd.DataFrame)

                feature = pd.merge(interaction_table, user_table, on=self.column_info.get_user_name(), how='left')

                feature = pd.merge(feature, item_table, on=self.column_info.get_item_name(), how='left')

                feature = feature[self.column_info.get_feature_names()]

                update_feature_info(feature, f)

                feature.to_hdf(os.path.join(self.feature_dir,
                                            f.replace(file_config['interaction_table_prefix'],
                                                      file_config['input_feature_prefix'])),
                               key='fixed', index=False)
                interaction_table[self.column_info.get_rating_name()].to_hdf(
                    os.path.join(self.feature_dir, f.replace(
                        file_config['interaction_table_prefix'],
                        file_config['output_feature_prefix'])),
                    key='fixed', index=False)
        else:
            pass

    def _get_raw_and_transformed_feature(self, base_id) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raw_feature = None
        transformed_feature = None

        def get_raw_feature(base_id):
            user_table = pd.read_hdf(os.path.join(self.preprocess_dir, f'{self.user_file_name}.{self.hdf5_postfix}'))
            item_table = pd.read_hdf(os.path.join(self.preprocess_dir, f'{self.item_file_name}.{self.hdf5_postfix}'))
            assert isinstance(user_table, pd.DataFrame) and isinstance(item_table, pd.DataFrame)
            if user_table[self.column_info.get_user_name()].dtype == int:
                base_id = int(base_id)
            elif user_table[self.column_info.get_user_name()].dtype == pd.StringDtype:
                base_id = str(base_id)
            user = user_table[user_table[self.column_info.get_user_name()] == base_id][
                    self.column_info.get_user_col_names()]
            items = item_table[self.column_info.get_item_col_names()]
            return pd.merge(user, items, how='cross')

        if self.feature_type == FeatureType.ALL:
            raw_feature = get_raw_feature(base_id)
            transformed_feature = self.feature_encoder.transform(raw_feature)
        elif self.feature_type == FeatureType.RAW:
            raw_feature = get_raw_feature(base_id)
            transformed_feature = raw_feature[self.column_info.get_feature_names()]
        else:
            pass
        return raw_feature, transformed_feature

    def clean_files(self):
        file_list = os.listdir(self.feature_dir)
        for f in file_list:
            if file_config['feature_info_class_file_name'] in f:
                continue
            os.remove(os.path.join(self.feature_dir, f))
