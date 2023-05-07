from classes.recommendation_type import RecommendationType
from classes.model_name import ModelName
from classes.model_info import ModelInfo, CatBoostInfo, AutoFisInfo, NCFInfo, GMFInfo, MLPInfo, SDAEInfo, SPEInfo
from typing import Dict, Union
from utils import from_dict
from inflection import underscore
import json

# model_config = {}
# model_config[ModelName.AutoFis] = [AutoFisInfo]
# model_config[ModelName.CatBoost] = [CatBoostInfo]
# model_config[ModelName.SPE] = [NCFInfo, GMFInfo]


class ModelInfoMapper:
    def __init__(self, model_name, model_info_dict):
        self.model_infos = {}
        if model_name == ModelName.AutoFis:
            self.model_infos['AutoFis'] = from_dict(AutoFisInfo, model_info_dict['AutoFis'])
        elif model_name == ModelName.CatBoost:
            self.model_infos['CatBoost'] = from_dict(CatBoostInfo, model_info_dict['CatBoost'])

# @dataclass_json
# @dataclass
class RecommendationInfo:
    recommendation_type: RecommendationType
    model_name: ModelName
    models_info: Dict[ModelName, ModelInfo]

    def __init__(self, recommendation_info: Union[Dict, str]):
        _recommendation_info = recommendation_info
        try:
            recommendation_info = json.loads(recommendation_info)
        except (TypeError, ValueError, json.JSONDecodeError, json.decoder.JSONDecodeError) as e:
            raise Exception(f'recommendation_info({_recommendation_info}) is not json or dictionary')

        assert isinstance(recommendation_info, Dict)

        temp_dict = {}
        for key, value in recommendation_info.items():
            key = underscore(key)
            temp_dict[key] = value

        self.recommendation_type = RecommendationType(temp_dict['recommendation_type'])
        self.model_name = ModelName(temp_dict['model_name'])
        self.models_info = {}
        # TODO model_config 추가해서 적용?

        if self.model_name == ModelName.CatBoost:
            self.models_info[ModelName.CatBoost] = \
                from_dict(CatBoostInfo, temp_dict['models_info'][ModelName.CatBoost.value])

        elif self.model_name == ModelName.AutoFis:
            self.models_info[ModelName.AutoFis] = \
                from_dict(AutoFisInfo, temp_dict['models_info'][ModelName.AutoFis.value])

        elif self.model_name == ModelName.SPE:
            self.models_info[ModelName.NCF] = \
                from_dict(NCFInfo, temp_dict['models_info'][ModelName.NCF.value])
            self.models_info[ModelName.GMF] = \
                from_dict(GMFInfo, temp_dict['models_info'][ModelName.GMF.value])
            self.models_info[ModelName.MLP] = \
                from_dict(MLPInfo, temp_dict['models_info'][ModelName.GMF.value])
            self.models_info[ModelName.SDAE] = \
                from_dict(SDAEInfo, temp_dict['models_info'][ModelName.SDAE.value])
            self.models_info[ModelName.SPE] = \
                from_dict(SPEInfo, temp_dict['models_info'][ModelName.SPE.value])
        else:
            assert ValueError(f'{self.model_name.value} model does not exist')



