from classes import ModelInfo, ColumnInfo, ModelName
from engine import CatBoostEngine
from engine import AutoFisEngine


class EngineFactory:
    @classmethod
    def from_infos(cls, model_info: ModelInfo, column_info: ColumnInfo, base_dir, save_dir):
        if model_info.model_name == ModelName.CatBoost:
            return CatBoostEngine(model_info, column_info, base_dir, save_dir)
        elif model_info.model_name == ModelName.AutoFis:
            return AutoFisEngine(model_info, column_info, base_dir, save_dir)
        else:
            raise ValueError(f'{model_info.model_name.name} model does not exist')
