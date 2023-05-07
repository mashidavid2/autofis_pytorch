from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import OrderedDict, List
from config.file_config import file_config


@dataclass
@dataclass_json
class FeatureInfo:
    train_size: int = 0
    test_size: int = 0
    evaluate_size: int = 0
    negative_size: int = 0
    num_users: int = 0
    num_items: int = 0
    feature_size_dict: OrderedDict[str, int] = field(default_factory=OrderedDict)

    @property
    def feature_dims(self) -> List[int]:
        return list(self.feature_size_dict.values())

    @property
    def feature_nums(self) -> int:
        return len(self.feature_size_dict)

    @property
    def feature_total_size(self) -> int:
        return sum(self.feature_size_dict.values())

    def save_to_pickle(self, feature_dir, model_name):
        import pickle
        import os
        with open(os.path.join(feature_dir, f'{model_name}_{file_config["feature_info_class_file_name"]}'), 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, feature_dir, model_name) -> 'FeatureInfo':
        import pickle
        import os
        with open(os.path.join(feature_dir, f'{model_name}_{file_config["feature_info_class_file_name"]}'), 'rb') as f:
            return pickle.load(f)






