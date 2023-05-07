import os
import pandas as pd
import random
from copy import deepcopy
# from src.classes import ColumnInfo
# from src.config.file_config import file_config
from classes import ColumnInfo
from config.file_config import file_config


class LeaveOneOutSplitter(object):
    def __init__(self, base_dir: str, column_info: ColumnInfo):
        self.preprocessed_dir = os.path.join(base_dir, file_config['data_dir_name'], file_config['preprocessed_dir_name'])
        self.column_info = column_info
        self.user_file_name = file_config['user_file_name']
        self.item_file_name = file_config['item_file_name']
        self.interaction_table_prefix = file_config['interaction_table_prefix']
        self.hdf5_postfix = file_config['hdf5_postfix']
        self.block_size = file_config['block_size']
        self.negative_sample_ratio_for_test = 50

    def leave_one_out_split(self, negative_sample_ratio=4):
        file_list = os.listdir(self.preprocessed_dir)
        interaction_files = [
            os.path.join(self.preprocessed_dir, f) for f in file_list if f'{self.interaction_table_prefix}_part' in f
        ]
        interaction_set_table = None
        latest = None

        for f in interaction_files:
            interaction_table = pd.read_hdf(f)
            assert isinstance(interaction_table, pd.DataFrame)

            interaction_set_table = self._get_interaction_set_table(interaction_table, interaction_set_table)
            latest = self._get_latest_data(interaction_table, latest)
            os.remove(f) # remove raw_interaction_table

        test_interaction_set_table = self._get_interaction_set_table(latest)

        item_table = pd.read_hdf(os.path.join(self.preprocessed_dir, f'{self.item_file_name}.{self.hdf5_postfix}'))

        self._save_interaction_table('train', interaction_set_table, item_table, negative_sample_ratio)
        #makes interaction table (1 if positive 0 if negative) -> used for CTR

        self._save_interaction_table(
            'test', test_interaction_set_table, item_table, self.negative_sample_ratio_for_test)
        self._save_interaction_table(
            'evaluate', test_interaction_set_table, item_table, self.negative_sample_ratio_for_test)

    def _get_latest_data(self, data: pd.DataFrame, latest=None):
        """leave one out train/test split"""
        data = deepcopy(data)
        if latest is not None:
            data = pd.concat((data, latest), axis=0)

        data['rank_latest'] = data.groupby(self.column_info.get_user_name())[
            self.column_info.get_temporal_name()].rank(
            method='first', ascending=False)
        # train = data[data['rank_latest'] > 1]
        # train.drop(columns=['rank_latest'], inplace=True)
        # train.reset_index(inplace=True, drop=True)
        test = data.loc[data['rank_latest'] == 1]
        test = test.drop(columns=['rank_latest'])
        test.reset_index(inplace=True, drop=True)
        return test

    def _get_interaction_set_table(self, interaction_data: pd.DataFrame, other_set_table=None):
        col_user = self.column_info.get_user_name()
        col_item = self.column_info.get_item_name()
        interaction_set_table = interaction_data.groupby(col_user)[col_item].apply(set).reset_index() \
            .rename(columns={col_item: 'interacted_items'})

        def union_interaction(df):
            if pd.isna(df['interacted_items_x']):
                return df['interacted_items_y']
            elif pd.isna(df['interacted_items_y']):
                return df['interacted_items_x']
            else:
                return df['interacted_items_x'] | df['interacted_items_y']

        if other_set_table is not None:
            interaction_set_table = pd.merge(interaction_set_table, other_set_table, on=col_user, how='outer')
            interaction_set_table['interacted_items'] = \
                interaction_set_table[['interacted_items_x', 'interacted_items_y']].apply(union_interaction, axis=1)
            interaction_set_table = interaction_set_table[[col_user, 'interacted_items']]
            interaction_set_table.sort_values(by=col_user, ignore_index=True)
        return interaction_set_table

    def _save_interaction_table(self, gen_type, interaction_set_table, item_table, negative_sample_ratio):
        assert gen_type in ['train', 'test', 'evaluate']
        negative_set_table = self._get_negative_set_table(gen_type, interaction_set_table, item_table, negative_sample_ratio)
        interaction_set_table = deepcopy(interaction_set_table)
        interaction_set_table['interacted_items'] = interaction_set_table['interacted_items'].apply(list)
        users, items, ratings = [], [], []
        for row in interaction_set_table.itertuples():
            for i in range(len(row.__getattribute__('interacted_items'))):
                users.append(row.__getattribute__(self.column_info.get_user_name()))
                items.append(row.__getattribute__('interacted_items')[i])
                ratings.append(1)

        n_users, n_items, n_ratings = [], [], []
        for row in negative_set_table.itertuples():
            for i in range(len(row.__getattribute__('negative_items'))):
                n_users.append(row.__getattribute__(self.column_info.get_user_name()))
                n_items.append(row.__getattribute__('negative_items')[i])
                n_ratings.append(0)

        if gen_type in ['train', 'test']:
            users.extend(n_users)
            items.extend(n_items)
            ratings.extend(n_ratings)

            if gen_type == 'train':
                interaction_samples = list(zip(users, items, ratings))
                random.shuffle(interaction_samples)
                users, items, ratings = zip(*interaction_samples)

        num_chunk = int(len(users) / self.block_size) + 1
        for idx in range(num_chunk):
            interaction_table = pd.DataFrame({
                self.column_info.get_user_name(): users[idx * self.block_size:(idx + 1) * self.block_size],
                self.column_info.get_item_name(): items[idx * self.block_size:(idx + 1) * self.block_size],
                self.column_info.get_rating_name(): ratings[idx * self.block_size:(idx + 1) * self.block_size]
            })
            self._save_table_to_hdf(interaction_table, f'{self.interaction_table_prefix}_{gen_type}_part_{idx}')

        if gen_type == 'evaluate':
            """evaluation data for at k_metrics"""
            num_chunk = int(len(n_users) / self.block_size) + 1
            for idx in range(num_chunk):
                negative_interaction_table = pd.DataFrame({
                    self.column_info.get_user_name(): n_users[idx * self.block_size:(idx + 1) * self.block_size],
                    self.column_info.get_item_name(): n_items[idx * self.block_size:(idx + 1) * self.block_size],
                    self.column_info.get_rating_name(): n_ratings[idx * self.block_size:(idx + 1) * self.block_size]
                })
                self._save_table_to_hdf(negative_interaction_table, f'{self.interaction_table_prefix}_negative_part_{idx}')

    def _get_negative_set_table(self, gen_type, interaction_set_table: pd.DataFrame, item_table: pd.DataFrame, negative_sample_ratio):
        item_pool = set(item_table[self.column_info.get_item_name()])
        negative_set_table = deepcopy(interaction_set_table)
        if gen_type == 'train':
            negative_set_table['count'] = negative_set_table['interacted_items'].apply(lambda x: len(x))

            negative_set_table['all_negative_items'] = negative_set_table['interacted_items'].apply(lambda x: item_pool - x)
            negative_set_table['negative_items'] = negative_set_table[['all_negative_items', 'count']].apply(
                lambda x: random.sample(x['all_negative_items'],
                                        min(len(x['all_negative_items']), x['count'] * negative_sample_ratio)), axis=1)
        if gen_type in ['test', 'evaluate']:
            negative_set_table['negative_items'] = negative_set_table['interacted_items']\
                .apply(lambda x: item_pool - x)\
                .apply(lambda x: random.sample(x, min(len(x), negative_sample_ratio)))
        return negative_set_table[[self.column_info.get_user_name(), 'negative_items']]

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
