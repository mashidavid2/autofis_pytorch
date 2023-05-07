import json
import os
from classes.model_name import ModelName
from utils.torch_utils import string_to_title, NpEncoder, get_dict_with_str_key

from abc import ABC, abstractmethod


class Observer(ABC):
    @abstractmethod
    def report(self, **kwargs):
        pass

class AutoFisObserver(Observer):
    def __init__(self, model_name: ModelName):
        self.model_name = model_name

    def report(self, result, save_dir:str = None):
        with open(os.path.join(save_dir, self.model_name + '.json'), 'w') as f:
            json.dump(result, f, indent=4)
            
        # for model in result:
            # for report_info in result[model]:
                # dirname = os.path.join(save_dir, "learning_info", model,)
                # os.makedirs(dirname, exist_ok=True)
                # with open(os.path.join(dirname, report_info + ".json"), "w") as json_file:
                    # json.dump(
                        # get_dict_with_str_key({report_info: result[model][report_info]}),
                        # json_file,
                        # cls=NpEncoder,
                        # indent=4,
                        # ensure_ascii=False,
                    # )

class CatBoostObserver(Observer):
    def __init__(self, model_name: ModelName):
        self.model_name = model_name

    def report(self, **kwargs):
        """
        result contains learning info. dict for each model
        kwargs = {
            "result": {
                model1: learning info1,
                model2: learning info2,
                ...
            }
        }
        learning_info = {
            "feature_importance": effect of each column (dict)
            "index": index of the pipeline (dict)
            "loss": evaluation result for each evaluation metric (dict)
            "partial_dependence": values for partial dependence plot (dict)
            "prediction": predicted target values (dict)
        }
        """
        result = kwargs["result"]
        for model in result:
            for report_info in result[model]:
                dirname = os.path.join(self.file_path, "learning_info", model,)
                os.makedirs(dirname, exist_ok=True)
                with open(os.path.join(dirname, report_info + ".json"), "w") as json_file:
                    json.dump(
                        get_dict_with_str_key({report_info: result[model][report_info]}),
                        json_file,
                        cls=NpEncoder,
                        indent=4,
                        ensure_ascii=False,
                    )
