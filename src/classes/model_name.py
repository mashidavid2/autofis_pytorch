from enum import Enum


class StrEnum(str, Enum):
    def __repr__(self):
        return str(self.name)

    def __str__(self):
        return str(self.name)


class ModelName(StrEnum):
    CatBoost = 'CatBoost'
    NCF = 'NCF'
    GMF = 'GMF'
    MLP = 'MLP'
    SDAE = 'SDAE'
    SPE = 'SPE'
    AutoFis = 'AutoFis'
