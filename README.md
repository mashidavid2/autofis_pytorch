## Personalized Item Recommendation model

### Model Test
#### requirements
- venv 구성하여 requirements.txt에 있는 dependency python package 추가 필요
```
$ python -m venv $(venv_name)
$ source $(venv_path)/bin/activate
$ pip install -r $(requirements.txt path)
```

#### sample data prepare
- ml-1m public dataset을 이용하여 test
- ml-1m public dataset (users.dat, movies.dat, ratings.dat)을 test를 위한 csv data format으로 변환
- src/tmp/dataset_ml-1m 에 해당 public dataset을 넣어 주어야 함
- dataset 및 학습 결과 들은 tmp 내에서 관리하여 gitignore처리 

#### model test
- args로 train 돌릴 model 및 dataset 정보를 넣어서 test
- model hyperparameter를 입력하지 않을 경우, 각 ModelInfo dataclass에서 정의한 default 값 사용
- 예시
```
## model_test.py 

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
    
args = parse_train_args()
args.column_info = json.dumps(column_info)
args.recommendation_info = json.dumps(recommendation_info_autofis)
train(args)
```