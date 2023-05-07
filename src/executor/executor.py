from typing import Dict, Union
import pandas as pd
import time
import datetime
import os
import csv
# from src.engine import Engine, EngineFactory
# from src.classes.column_info import ColumnInfo
# from src.classes.model_name import ModelName
# from src.preprocess import TableSplitter, LeaveOneOutSplitter, FeatureTransformer
# from src.dataset import DatasetGenerator
# from src.classes.model_info import ModelInfo
# from src.classes.evaluation import Evaluation
from engine import Engine, EngineFactory
from classes.column_info import ColumnInfo
from classes.model_name import ModelName
from preprocess import TableSplitter, LeaveOneOutSplitter, FeatureTransformer
from dataset import DatasetGenerator
from classes.model_info import ModelInfo
from classes.evaluation import Evaluation
from observer import Observer, ObserverFactory


def check_min_requirements(model_info: ModelInfo, column_info: ColumnInfo):
    if len(column_info.get_feature_names()) < model_info.min_requirement_of_feature_num:
        raise AttributeError(
            f'{model_info.model_name.value} model must satisfy minimum requirement of feature numbers, '
            f'feature number: {len(column_info.get_feature_names())} (column_info: {column_info}')


class Executor:
    def __init__(self, model_name: ModelName):
        self.model_name = model_name

    def execute_train(self, models_info: Dict[ModelName, ModelInfo], column_info: ColumnInfo, base_dir: str,
                      interaction_file_path: str, user_file_path: str = None, item_file_path: str = None, save_dir: str = None) -> Evaluation:
        """if several model need to train, override this method"""
        start_time = time.time()
        model_info = models_info[self.model_name]
        check_min_requirements(model_info, column_info)
        print(f'[ModelInfo] {str(model_info)}\n')
        print(f'[ColumnInfo] {str(column_info)}\n')

        print('[Preprocessing Start]')
        table_splitter = self._get_table_splitter(
            column_info, base_dir, interaction_file_path, user_file_path, item_file_path)
        table_splitter.split_to_user_item_interaction_table() #validate data and save as h5 file

        leave_one_out_splitter = self._get_leave_one_out_splitter(column_info, base_dir)
        leave_one_out_splitter.leave_one_out_split(negative_sample_ratio=model_info.num_negative)
        #splits interaction data into train, test, eval

        feature_transformer = self._get_feature_transformer(model_info, column_info, base_dir)
        feature_transformer.transform_to_file_for_training()
        end_preprocess_time = time.time()
        print('[Preprocessing Done]\n')

        self.make_dir(save_dir)
        dataset_generator = self._get_dataset_generator(base_dir) #generate dataset

        observer = self._get_observer(model_info)
        engine = self._get_engine(model_info, column_info, base_dir, save_dir)
        print(f'[{model_info.model_name.value} Model Train Start]')
        evaluation = engine.train(dataset_generator)
        print(f'[{model_info.model_name.value} Model Train Done]\n')
        end_time = time.time()

        preprocessed_time = end_preprocess_time - start_time
        training_time = end_time - start_time
        total_time = end_time - start_time

        leave_one_out_splitter.clean_files()
        feature_transformer.clean_files()

        result = self._result_to_dict(model_info, evaluation, preprocessed_time, training_time, total_time)
        observer.report(result, save_dir)
        return evaluation
    
    def make_dir(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

    def _result_to_dict(self, model_info: ModelInfo, evaluation: Evaluation,
                     preprocessed_time, training_time, total_time):
        result = dict()
        result[self.model_name] = dict()
        result[self.model_name]['model_info'] = model_info.to_dict()
        result[self.model_name]['model_info']['evaluation_type'] = result['AutoFis']['model_info']['evaluation_type'].value
        # result[self.model_name]['model_name'] = self.model_name
        result[self.model_name]['results'] = dict()
        result[self.model_name]['results']['end_date'] = time.strftime('%Y-%m-%d %I:%M:%S', time.localtime(time.time()))
        result[self.model_name]['results']['preprocessed_time'] = str(datetime.timedelta(seconds=preprocessed_time))
        result[self.model_name]['results']['training_time'] = str(datetime.timedelta(seconds=training_time))
        result[self.model_name]['results']['total_time'] = str(datetime.timedelta(seconds=total_time))
        result[self.model_name]['results']['test_loss'] = evaluation.loss
        result[self.model_name]['results']['auc'] = evaluation.auc
        result[self.model_name]['results']['hit_ratio'] = evaluation.hit_ratio
        result[self.model_name]['results']['map'] = evaluation.map
        result[self.model_name]['results']['ndcg'] = evaluation.ndcg
        result[self.model_name]['results']['mrr'] = evaluation.mrr

        # print('[Result]')
        # print(''.join([f'{key} = {value}, ' for key, value in result.items()]))

        '''
        is_first = os.path.isfile(os.path.join(save_dir, f'{result["model_name"]}.csv'))

        with open(os.path.join(save_dir, f'{result["model_name"]}_result.csv'), 'a') as f:
            w = csv.writer(f)
            if not is_first:
                w.writerow(result.keys())
            w.writerow(result.values())
        '''    
        return result

    def execute_inference(self, models_info: Dict[ModelName, ModelInfo], column_info: ColumnInfo,
                          train_dir: str, target_id: Union[str, int], top_k=10, output_path=None) -> pd.DataFrame:
        model_info = models_info[self.model_name]
        feature_transformer = self._get_feature_transformer(model_info, column_info, train_dir)
        raw_feature, transformed_feature = feature_transformer.get_raw_and_transformed_feature(target_id)

        observer = self._get_observer(model_info)
        engine = self._get_engine(model_info, column_info, train_dir)
        scores = engine.inference(transformed_feature)

        col_score = 'Score'
        raw_feature[col_score] = scores
        inference_table = raw_feature.sort_values(by=col_score, ascending=False, ignore_index=True)[0:top_k]
        ordered_cols = column_info.get_inference_cols_names() + [col_score]
        inference_table = inference_table[ordered_cols]

        # mlplatform backend 서버에서 데이터에 (", ')문자열이 섞여 있으면 parsing 이 잘 안되 str으로 한번 더 감싸서 내보내줌
        for col in inference_table.columns:
            if inference_table[col].dtype == object:
                inference_table[col] = [str(value) for value in inference_table[col]]
        if output_path is not None:
            inference_table.to_csv(output_path, index=False)
        print(inference_table)
        return inference_table

    def _get_engine(self, model_info: ModelInfo, column_info: ColumnInfo, base_dir: str, save_dir: str) -> Engine:
        return EngineFactory.from_infos(model_info, column_info, base_dir, save_dir)
    
    def _get_observer(self, model_info: ModelInfo) -> Observer:
        return ObserverFactory.from_infos(model_info)

    def _get_table_splitter(self, column_info: ColumnInfo, base_dir: str, interaction_file_path: str,
                            user_file_path: str, item_file_path: str) -> TableSplitter:
        return TableSplitter(
            base_dir=base_dir,
            column_info=column_info,
            interaction_file_path=interaction_file_path,
            user_file_path=user_file_path,
            item_file_path=item_file_path
        )

    def _get_leave_one_out_splitter(self, column_info: ColumnInfo, base_dir: str) -> LeaveOneOutSplitter:
        return LeaveOneOutSplitter(
            base_dir=base_dir,
            column_info=column_info
        )

    def _get_feature_transformer(self, model_info: ModelInfo, column_info: ColumnInfo, base_dir: str) -> FeatureTransformer:
        return FeatureTransformer(
            base_dir=base_dir,
            model_info=model_info,
            column_info=column_info,
        )

    def _get_dataset_generator(self, base_dir: str) -> DatasetGenerator:
        return DatasetGenerator(
            base_dir=base_dir
        )


class CatBoostExecutor(Executor):
    def __init__(self, model_name: ModelName):
        super(CatBoostExecutor, self).__init__(model_name)


class AutoFisExecutor(Executor):
    def __init__(self, model_name: ModelName):
        super(AutoFisExecutor, self).__init__(model_name)