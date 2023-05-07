import os
import sys
import argparse
import json
import yaml
import os.path as path
module_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)
from classes import ColumnInfo, ModelInfoMapper
from executor import ExecutorFactory
from utils import from_dict, yaml_to_model_dict, yaml_to_config_dict

package_path = path.dirname(path.dirname(path.abspath(__file__)))
config_path = path.join(package_path,'config')

def train():
    #for local use
    model_name = ''
    with open(path.join(config_path,'autofis_model_config.yaml'),'r') as f:
        model_yaml = yaml.safe_load(f)
    model_name, model_info_dict = yaml_to_model_dict(model_yaml)
    # with open(path.join(config_path,'data_config.yaml'),'r') as f:
    #     column_yaml = yaml.safe_load(f)
    # column_dict = yaml_to_config_dict(column_yaml)
    column_dict = {
        'rating_column': {'name': 'Rating', 'type': 'Numerical'},
        'user_id_column': {'name': 'UserID', 'type': 'Categorical'},
        'user_feature_columns': [
            {'name': 'Gender', 'type': 'Categorical'},
            {'name': 'Age', 'type': 'Categorical'},
            {'name': 'Occupation', 'type': 'Categorical'}
        ],
        'item_id_column': {'name': 'MovieID', 'type': 'Categorical'},
        'item_feature_columns': [
            {'name': 'Genres', 'type': 'Categorical'},
            {'name': 'Year', 'type': 'Numerical'}
        ],
        'timestamp_column': {'name': '', 'type': 'TimeStamp'}
    }
    current_path = path.dirname(path.abspath(__file__))
    dataset = 'ml-1m'
    save_dir = path.join(current_path, 'save')
    train_dir = path.join(current_path, 'tmp/dataset_'+ dataset)
    interaction_path = path.join(train_dir, dataset +'_interaction_table.csv')
    user_path = path.join(train_dir, dataset +'_user_table.csv')
    item_path = path.join(train_dir, dataset +'_item_table.csv')
    model_info_mapper = ModelInfoMapper(model_name, model_info_dict)

    executor = ExecutorFactory.from_model_name(model_name)

    evaluation = executor.execute_train(
        models_info=model_info_mapper.model_infos,
        column_info=from_dict(ColumnInfo, column_dict),
        base_dir=train_dir,
        interaction_file_path=interaction_path,
        user_file_path=user_path,
        item_file_path=item_path,
        save_dir=save_dir,
    )

def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--column_info', type=str, default='')
    parser.add_argument('--recommendation_info', type=str, default='')
    parser.add_argument('--models_info', type=str, default='') 
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_train_args()
    train()