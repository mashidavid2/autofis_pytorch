from classes.model_name import ModelName
from executor import Executor, CatBoostExecutor, AutoFisExecutor


class ExecutorFactory:
    """
    Module for activating specific Executor 
    """
    @classmethod
    def from_model_name(cls, model_name: ModelName) -> Executor:
        if model_name == ModelName.CatBoost:
            return CatBoostExecutor(model_name)
        elif model_name == ModelName.AutoFis:
            return AutoFisExecutor(model_name)
        # elif model_name == ModelName.SDAE:
        #     return SPEExecutor(model_name)
        else:
            raise ValueError(f'{model_name} model does not exist')
