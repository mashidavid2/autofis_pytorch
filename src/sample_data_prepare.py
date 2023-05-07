import os
import re
import pandas as pd


def sample_data_prepare():
    """
    ml-1m data preprocessing
    ml-1m public dataset (users.dat, movies.dat, ratings.dat)을 test를 위한 csv data format으로 변환
    src/tmp/dataset_ml-1m 에 해당 public dataset을 넣어 주어야 함
    """
    base_dir = os.path.abspath(os.path.dirname('__file__'))
    data_dir = os.path.join(base_dir, 'src', 'tmp', 'dataset_ml-1m')

    # UserID::Gender::Age::Occupation::Zip-code
    user_table = pd.read_table(os.path.join(data_dir, 'users.dat'), sep='::', encoding='utf-8', header=None)
    user_table.rename(columns={0: 'UserID', 1: 'Gender', 2: 'Age', 3: 'Occupation', 4: 'ZipCode'}, inplace=True)
    # MovieID::Title::Genres
    item_table = pd.read_table(os.path.join(data_dir, 'movies.dat'), sep='::', encoding='ISO-8859-1', header=None)
    item_table.rename(columns={0: 'MovieID', 1: 'Title', 2: 'Genres'}, inplace=True)
    item_table['Genres'] = item_table['Genres'].apply(lambda x: x.split('|')[0])
    regex = r'\s{0,}\((.*?)\)'
    item_table['Year'] = item_table['Title'].apply(lambda x: re.findall(regex, x)).apply(lambda x: int(x[len(x)-1]))
    item_table['Title'] = item_table['Title'].apply(lambda x: ''.join(re.split(regex, x)[0:-2]))
    # UserID::MovieID::Rating::Timestamp
    interaction_table = pd.read_table(os.path.join(data_dir, 'ratings.dat'), sep='::', encoding='utf-8', header=None)
    interaction_table.rename(columns={0: 'UserID', 1: 'MovieID', 2: 'Rating', 3: 'Timestamp'}, inplace=True)

    full_table = interaction_table.merge(user_table, how='left', on='UserID')
    full_table = full_table.merge(item_table, how='left', on='MovieID')

    user_table.to_csv(os.path.join(data_dir, 'ml_1m_user_table.csv'), index=False, encoding='utf-8')
    item_table.to_csv(os.path.join(data_dir, 'ml_1m_item_table.csv'), index=False, encoding='utf-8')
    interaction_table.to_csv(os.path.join(data_dir, 'ml_1m_interaction_table.csv'), index=False, encoding='utf-8')
    full_table.to_csv(os.path.join(data_dir, 'ml_1m_full_table.csv'), index=False, encoding='utf-8')


if __name__ == '__main__':
    sample_data_prepare()
