import pandas as pd
import os
import json
# from src.config.file_config import file_config
# from src.classes import ColumnInfo
from config.file_config import file_config
from classes import ColumnInfo

class TableReader:
    def __init__(self, base_dir, column_info: ColumnInfo):
        self.preprocess_dir = os.path.join(base_dir, file_config['data_dir_name'], file_config['preprocessed_dir_name'])
        self.user_file_name = file_config['user_file_name']
        self.item_file_name = file_config['item_file_name']
        self.hdf5_postfix = file_config['hdf5_postfix']
        self.column_info = column_info

    def get_user_info_str(self):
        # TODO user info 저장 방법 변경
        user_table = pd.read_hdf(os.path.join(self.preprocess_dir, f'{self.user_file_name}.{self.hdf5_postfix}'))
        assert isinstance(user_table, pd.DataFrame)
        user_ids = user_table[self.column_info.get_user_name()].to_list()
        user_ids_dict = [{'userId': str(user_id)} for user_id in user_ids]
        return json.dumps(user_ids_dict)


