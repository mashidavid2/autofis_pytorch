import os
import sys
import argparse
import json

module_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
print(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)
from src.classes import ColumnInfo, RecommendationInfo
from src.executor import ExecutorFactory
from src.utils import from_json

seeds = [0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC,
                 0x0123, 0x4567, 0x3210, 0x7654, 0x89AB, 0xCDEF, 0xBA98, 0xFEDC]


def train(args):
    base_dir = os.path.abspath(os.path.dirname('__file__'))
    csv_path = base_dir + '/tmp/dataset_ml-1m/ml_1m_full_table.csv'
    base_dir = base_dir + '/tmp'

    recommendation_info = RecommendationInfo(args.recommendation_info)

    executor = ExecutorFactory.from_model_name(recommendation_info.model_name)

    executor.execute_train(
        models_info=recommendation_info.models_info,
        column_info=from_json(ColumnInfo, args.column_info),
        base_dir=base_dir,
        interaction_file_path=csv_path
    )


def inference(args):
    base_dir = os.path.abspath(os.path.dirname('__file__'))
    base_dir = base_dir + '/tmp'

    recommendation_info = RecommendationInfo(args.recommendation_info)

    executor = ExecutorFactory.from_model_name(recommendation_info.model_name)

    executor.execute_inference(
        models_info=recommendation_info.models_info,
        column_info=from_json(ColumnInfo, args.column_info),
        train_dir=base_dir,
        target_id=args.id
    )


def parse_train_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--column_info", type=str, default="")
    parser.add_argument("--recommendation_info", type=str, default="")
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()
    return args


def parse_inference_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--column_info', type=str, default='')
    parser.add_argument('--recommendation_info', type=str, default='')
    parser.add_argument('--id', type=str, default='')
    parser.add_argument('--top_k', type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    column_info = {
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

    recommendation_info_cat_boost = {
        'recommendation_type': 'USER2ITEM',
        'model_name': 'CatBoost',
        'models_info': {
            'CatBoost': {
                # 'model_name': 'CatBoost',
                # 'epoch': 100,
                # # 'epoch': 10,
                # 'batch_size': 2048,
                # 'num_negative': 4,
                # 'learning_rate': 0.01,
                # 'depth': 10
                # # 'depth': 1
            },
        }
    }

    recommendation_info_autofis = {
        'recommendation_type': 'USER2ITEM',
        'model_name': 'AutoFis',
        'models_info': {
            'AutoFis': {
                'model_name': 'AutoFis',
                'batch_size': 2048,
                'num_negative': 4,
                'learning_rate': 0.001,
                'latent_dim': 10,
                # # 'epoch': 100,
                'epoch': 5
            },
        }
    }

    # recommendation_info = {
    #     'recommendation_type': 'USER2ITEM',
    #     'model_name': 'SPE',
    #     'models_info': {
    #         'NCF': {
    #             'model_name': 'NCF',
    #             'train_test_ratio': 0.7,
    #             'batch_size': 1024,
    #             'latent_dim_gmf': 8,
    #             'latent_dim_mlp': 8,
    #             'num_negative': 4,
    #             'learning_rate': 0.001,
    #             'regularization': 0.01,
    #             'layers': [16,64,32,16,8],
    #             'epoch': 50,
    #             # 'epoch': 1
    #         },
    #         'MLP': {
    #             'model_name': 'MLP',
    #             'train_test_ratio': 0.7,
    #             'batch_size': 1024,
    #             'latent_dim': 8,
    #             'num_negative': 4,
    #             'learning_rate': 0.001,
    #             'regularization': 0.01,
    #             'layers': [16,64,32,16,8],
    #             'epoch': 200,
    #             # 'epoch': 1
    #         },
    #         'GMF': {
    #             'model_name': 'GMF',
    #             'train_test_ratio': 0.7,
    #             'batch_size': 1024,
    #             'latent_dim': 8,
    #             'num_negative': 4,
    #             'learning_rate': 0.001,
    #             'regularization': 0,
    #             'epoch': 200,
    #             # 'epoch': 1,
    #         },
    #         'SDAE': {
    #             'model_name': 'SDAE',
    #             'train_test_ratio': 0.8,
    #             'batch_size': 128,
    #             'learning_rate': 0.001,
    #             'dropout': 0.1,
    #             'corruption': 0.3,
    #             'regularization': 0.001,
    #             'hidden_layers': [16],
    #             'latent_layer': 8,
    #             'epoch': 1
    #         },
    #         'SPE': {
    #             'model_name': 'SPE',
    #             'train_test_ratio': 0.7,
    #             'batch_size': 1024,
    #             'learning_rate': 0.001,
    #             'regularization_spe': 0.01,
    #             'regularization_ncf': 0.01,
    #             'regularization_sdae_encode': 0.001,
    #             'regularization_sdae_decode': 0.001,
    #             'corruption': 0.3,
    #             'epoch': 1
    #         }
    #     }
    # }

    # autofis_epoch = [1, 100]
    # autofis_latent_dim = [10, 20]
    #
    # for epoch in autofis_epoch:
    #     for dim in autofis_latent_dim:
    #         recommendation_info_autofis['models_info']['AutoFis']['epoch'] = epoch
    #         recommendation_info_autofis['models_info']['AutoFis']['latent_dim'] = dim
    #         args = parse_train_args()
    #         args.column_info = json.dumps(column_info)
    #         args.recommendation_info = json.dumps(recommendation_info_autofis)
    #         train(args)
    #
    # catboost_epoch = [1, 1000]
    # catboost_batch_size = [2048]
    # catboost_learning_rate = [0.01, 0.005]
    # catboost_depth = [4, 6, 8, 10]
    # for epoch in catboost_epoch:
    #     for batch_size in catboost_batch_size:
    #         for lr in catboost_learning_rate:
    #             for depth in catboost_depth:
    #                 recommendation_info_cat_boost['models_info']['CatBoost']['epoch'] = epoch
    #                 recommendation_info_cat_boost['models_info']['CatBoost']['batch_size'] = batch_size
    #                 recommendation_info_cat_boost['models_info']['CatBoost']['learning_rate'] = lr
    #                 recommendation_info_cat_boost['models_info']['CatBoost']['depth'] = depth
    #                 args = parse_train_args()
    #                 args.column_info = json.dumps(column_info)
    #                 args.recommendation_info = json.dumps(recommendation_info_cat_boost)
    #                 train(args)

    args = parse_train_args()
    args.column_info = json.dumps(column_info)
    args.recommendation_info = json.dumps(recommendation_info_autofis)
    train(args)

    # args = parse_train_args()
    # args.column_info = json.dumps(column_info)
    # args.recommendation_info = json.dumps(recommendation_info_cat_boost)
    # train(args)

    # args = parse_train_args()
    # args.column_info = json.dumps(column_info)
    # args.recommendation_info = json.dumps(recommendation_info)
    # train(args)

    # args = parse_inference_args()
    # args.column_info = json.dumps(column_info)
    # args.recommendation_info = json.dumps(recommendation_info_autofis)
    # args.id = 1
    # inference(args)

    # args = parse_inference_args()
    # args.column_info = json.dumps(column_info)
    # args.recommendation_info = json.dumps(recommendation_info_cat_boost)
    # args.id = 1
    # inference(args)

