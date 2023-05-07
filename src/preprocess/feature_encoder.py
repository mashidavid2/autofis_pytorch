import numpy as np
import pandas as pd
from collections import OrderedDict
# from src.classes import ColumnInfo, ModelInfo, EncoderType, FeatureType
from classes import ColumnInfo, ModelInfo, EncoderType, FeatureType
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler


class FeatureEncoder:
    def __init__(self, column_info: ColumnInfo, model_info: ModelInfo, feature_type: FeatureType):
        # assert feature_type in ['all', 'user', 'item', 'cat']
        # self.feature_type = feature_type
        self.model_name = model_info.model_name
        self.num_encoder_type = model_info.num_encoder_type
        self.cat_encoder_type = model_info.cat_encoder_type
        self.encoders = self._get_encoders_dict(column_info, feature_type)

    def _get_encoders_dict(self, column_info, feature_type: FeatureType):
        """
        order by [user_id, user_num, user_cat, item_id, item_num, item_cat]
        """
        encoders = OrderedDict()
        if feature_type is FeatureType.ALL or feature_type is FeatureType.USER:
            for col in column_info.get_user_numerical_columns():
                encoders[col] = Encoder(self.num_encoder_type)
            for col in column_info.get_user_categorical_columns():
                encoders[col] = Encoder(self.cat_encoder_type)
        if feature_type is FeatureType.ALL or feature_type is FeatureType.ITEM:
            for col in column_info.get_item_numerical_columns():
                encoders[col] = Encoder(self.num_encoder_type)
            for col in column_info.get_item_categorical_columns():
                encoders[col] = Encoder(self.cat_encoder_type)
        return encoders

    def fit(self, user_data: pd.DataFrame, item_data: pd.DataFrame):
        for col, encoder in self.encoders.items():
            if user_data is not None and col in user_data.columns:
                encoder.fit(user_data[col])
            if item_data is not None and col in item_data.columns:
                encoder.fit(item_data[col])

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        result = None
        el = [(col, encoder) for col, encoder in self.encoders.items() if col in data.columns]
        for col, encoder in el:
            if result is None:
                result = encoder.transform(data[col])
                if encoder.encoder_type in [EncoderType.KBins, EncoderType.Norm]:
                    result = pd.DataFrame(result, columns=[col])
            else:
                temp = encoder.transform(data[col])
                if encoder.encoder_type in [EncoderType.KBins, EncoderType.Norm]:
                    temp = pd.DataFrame(temp, columns=[col])
                result = pd.concat((result, temp), axis=1)
        return result

    @property
    def feature_size_dict(self):
        feature_size_dict = OrderedDict()
        for col, encoder in self.encoders.items():
            feature_size_dict[col] = encoder.feature_size
        return feature_size_dict


class Encoder:
    def __init__(self, encoder_type: EncoderType):
        self.encoder_type = encoder_type
        self.encoder = self._get_encoder(encoder_type)

    def _get_encoder(self, encoder_type):
        if encoder_type == EncoderType.OneHot:
            return OneHotEncoder()
        elif encoder_type == EncoderType.Label:
            return OrdinalEncoder()
            # return OrdinalEncoderWrapper()
        elif encoder_type == EncoderType.KBins:
            return KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
        elif encoder_type == EncoderType.Norm:
            return MinMaxScaler(clip=True)

    def fit(self, data: pd.DataFrame):
        if self.encoder_type in [EncoderType.KBins, EncoderType.Norm]:
            data = np.array(data).reshape(-1, 1)
        self.encoder.fit(data)

    @property
    def feature_size(self):
        if self.encoder_type == EncoderType.OneHot:
            return len(self.encoder.feature_names)
        elif self.encoder_type == EncoderType.Label:
            return len(self.encoder.mapping[0]['mapping'])
        elif self.encoder_type == EncoderType.KBins:
            return self.encoder.n_bins
        elif self.encoder_type == EncoderType.Norm:
            return 1
        # TODO handle unique data feature size

    def transform(self, data: pd.DataFrame):
        if self.encoder_type in [EncoderType.KBins, EncoderType.Norm]:
            data = np.array(data).reshape(-1, 1)
        elif self.encoder_type == EncoderType.Label:
            unknown_value = len(self.encoder.mapping[0]['mapping'])
            res = self.encoder.transform(data)
            col = res.columns[0]
            # change unknown value, cause default ordinal encoder set unknown value to -1, or -2
            res.loc[res[col] < 0, col] = unknown_value
            return res
        return self.encoder.transform(data)
