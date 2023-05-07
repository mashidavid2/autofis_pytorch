## AutoFIS Recommendation model with MovieLen 1M dataset

#### requirements
- I recommend you to create virtual environment for running the code 
```
$ conda create -n $(the name of your env) python=3.10
$ pip install -r $(requirements.txt path)
```
some errors could come up while installing the requirements. If so, you should have to install them one by one.

#### sample raw data preprocessing
- for my code, I used ml-1m benchmark dataset
- sample_data_prepare.py converts ml-1m public dataset (users.dat, movies.dat, ratings.dat) into csv data format
- you should put the benchmark dataset at src/tmp/dataset_ml-1m

- To use other datasets, you have to modify the preprocess part of the code by yourselves as of right now. Could be updated later on.

#### model train
- I run the model with the information in the args or configurations
- If you don't insert model hyperparameters, it will use default values in the ModelInfo dataclass
- Example
```
## train.py 

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