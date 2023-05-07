import argparse
import json
from mlplatform_lib.api_client import ApiClient
from mlplatform_lib.mlplatform import MlPlatformApi
from mlplatform_lib.predefinedai import PredefinedAIApi, PredefinedAIArgumentParser
from src.classes import ColumnInfo, ModelInfoMapper, RecommendationInfo
from mlplatform_lib.predefinedai.predefinedai_config_parser import PredefinedAIConfigParser
from src.executor import ExecutorFactory
from src.utils import from_dict

api_client = ApiClient(server_config_path="../config/server_config.yaml")
predefinedai_api = PredefinedAIApi(api_client=api_client)


def inference():
    parser = PredefinedAIConfigParser(
        data_config_path='../config/data_config_test.yaml',
        model_config_path='../config/autofis_model_config.yaml',
        stage_config_path='../config/stage_config.yaml'
    )

    column_dict = {}
    column_dict.update(parser.get_data_config_dict_from_data_key('user_feature_data'))
    column_dict.update(parser.get_data_config_dict_from_data_key('item_feature_data'))
    column_dict.update(parser.get_data_config_dict_from_data_key('user_item_interaction_data'))

    model_name = parser.get_model_name()
    model_info_dict = {}
    for sub_model_name in parser.get_sub_model_name_list():
        model_info_dict[sub_model_name] = parser.get_model_hyperparameters_dict(sub_model_name)
        model_info_dict[sub_model_name]['model_name'] = sub_model_name

    args = parser.get_inference_args_dict()
    print("inference start")
    print("user_id", args["user_id"])

    train_dir = predefinedai_api.get_train_path()
    output_csv_path = predefinedai_api.get_inference_csv_path()

    model_info_mapper = ModelInfoMapper(model_name, model_info_dict)

    executor = ExecutorFactory.from_model_name(model_name)

    executor.execute_inference(
        models_info=model_info_mapper.model_infos,
        column_info=from_dict(ColumnInfo, column_dict),
        train_dir=train_dir,
        target_id=args['user_id'],
        output_path=output_csv_path
    )

    predefinedai_api.upload_inference_csv(inference_csv_path=output_csv_path)


def parse_inference_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--column_info', type=str, default='')
    parser.add_argument('--recommendation_info', type=str, default='')
    parser.add_argument('--models_info', type=str, default='')  # 기존 version front 호환을 위한 임시 argument
    parser.add_argument('--id', type=str, default='')
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_inference_args()

    # column_info = {
    #     'col_rating_name': {'name': 'Rating', 'type': 'num'},
    #     'col_user_name': {'name': 'UserID', 'type': 'cat'},
    #     'col_user_features': [
    #         {'name': 'Gender', 'type': 'cat'},
    #         {'name': 'Age', 'type': 'num'},
    #         {'name': 'Occupation', 'type': 'cat'}
    #     ],
    #     'col_item_name': {'name': 'MovieID', 'type': 'cat'},
    #     'col_item_features': [
    #         {'name': 'Genre', 'type': 'cat'},
    #         {'name': 'Year', 'type': 'num'}
    #     ],
    #     'col_temporal_name': {'name': '', 'type': 'tem'}
    # }
    #
    # recommendation_info_cat_boost = {
    #     'recommendation_type': 'USER2ITEM',
    #     'model_name': 'CatBoost',
    #     'models_info': {
    #         'CatBoost': {
    #             'model_name': 'CatBoost',
    #             'epoch': 100,
    #             'batch_size': 2048,
    #             'num_negative': 4,
    #             'learning_rate': 0.01,
    #             'depth': 10
    #             # 'depth': 1
    #         },
    #     }
    # }
    #
    # recommendation_info_autofis = {
    #     'recommendation_type': 'USER2ITEM',
    #     'model_name': 'AutoFis',
    #     'models_info': {
    #         'AutoFis': {
    #             'model_name': 'AutoFis',
    #             'batch_size': 2048,
    #             'num_negative': 4,
    #             'learning_rate': 0.001,
    #             'latent_dim': 10,
    #             # 'epoch': 100,
    #             'epoch': 10
    #         },
    #     }
    # }
    #
    # args.recommendation_info = json.dumps(recommendation_info_autofis)
    # # args.recommendation_info = json.dumps(recommendation_info_cat_boost)
    # args.id = 1
    inference()
