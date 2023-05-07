from classes import ModelInfo, ModelName
from observer.observers import Observer, AutoFisObserver, CatBoostObserver

class ObserverFactory:
    """
    Module for activating specific Observer
    """
    @classmethod
    def from_infos(cls, model_info: ModelInfo) -> Observer:
        model_name = model_info.model_name
        if model_name == ModelName.CatBoost:
            return CatBoostObserver(model_name)
        elif model_name == ModelName.AutoFis:
            return AutoFisObserver(model_name)
        else:
            raise ValueError(f'{model_name} model does not exist')
