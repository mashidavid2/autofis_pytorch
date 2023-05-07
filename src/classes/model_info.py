from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List
from classes import ModelName, FeatureType, EncoderType
from classes import EvaluationType


@dataclass_json
@dataclass
class ModelInfo:
    model_name: ModelName = field(init=False)
    # algorithm parameter
    epoch: int = field(init=False)
    batch_size: int = field(init=False)
    learning_rate: float = field(init=False)
    num_negative: int = field(init=False)
    # preprocessing parameter
    feature_type: FeatureType = field(init=False)
    num_encoder_type: EncoderType = field(init=False)
    cat_encoder_type: EncoderType = field(init=False)
    evaluation_type: EvaluationType = field(init=False)
    min_requirement_of_feature_num: int = field(init=False)

    def __str__(self):
        return ''.join([f'{key} = {value}, 'for key, value in self.__dict__.items()])


@dataclass_json
@dataclass
class CatBoostInfo(ModelInfo):
    model_name: ModelName = ModelName.CatBoost
    # algorithm parameter
    epoch: int = 1000
    batch_size: int = 2048
    learning_rate: float = 0.01
    num_negative: int = 4
    # model parameter
    depth: int = 10
    # ----- below fixed parameter -----
    # preprocessing parameter
    feature_type: FeatureType = FeatureType.RAW
    num_encoder_type: EncoderType = EncoderType.CatBoost
    cat_encoder_type: EncoderType = EncoderType.CatBoost
    evaluation_type: EvaluationType = EvaluationType.HitRatio
    min_requirement_of_feature_num: int = 1


@dataclass_json
@dataclass
class AutoFisInfo(ModelInfo):
    model_name: ModelName = ModelName.AutoFis
    # algorithm parameter
    epoch: int = 100
    batch_size: int = 2048
    num_negative: int = 4
    learning_rate: float = 0.001
    # model parameter
    latent_dim: int = 20
    # ----- below fixed parameter -----
    depth: int = 5
    width: int = 700
    # second-order parameter
    weight_base: float = 0.6
    # grda parameter
    grda_c: float = 0.0005
    grda_mu: float = 0.8
    decay_rate: float = 0.7
    learning_rate2: float = 1.0
    # decay_rate2: float = 1.0
    # learning_rate2: float = 0.1
    decay_rate2: float = 0.999
    # preprocessing parameter
    feature_type: FeatureType = FeatureType.ALL
    num_encoder_type: EncoderType = EncoderType.KBins
    cat_encoder_type: EncoderType = EncoderType.Label
    evaluation_type: EvaluationType = EvaluationType.AUC
    min_requirement_of_feature_num: int = 2


@dataclass_json
@dataclass
class GMFInfo(ModelInfo):
    epoch: int = 200
    batch_size: int = 1024
    learning_rate: float = 0.001
    latent_dim: int = 8
    num_negative: int = 4
    regularization: float = 0
    num_users: int = field(init=False)
    num_items: int = field(init=False)
    # regularization: float


@dataclass_json
@dataclass
class MLPInfo(ModelInfo):
    epoch: int = 200
    batch_size: int = 1024
    learning_rate: float = 0.001
    latent_dim: int = 8
    num_negative: int = 4
    regularization: float = 0.0000001
    layers: List[int] = field(default_factory=[16, 64, 32, 16, 8])
    num_users: int = field(init=False)
    num_items: int = field(init=False)


@dataclass_json
@dataclass
class NCFInfo(ModelInfo):
    epoch: int = 50
    batch_size: int = 1024
    learning_rate: float = 0.001
    num_negative: int = 4
    regularization: float = 0.01
    layers: List[int] = field(default_factory=[16, 64, 32, 16, 8])
    latent_dim_gmf: int = 8
    latent_dim_mlp: int = 8
    num_users: int = field(init=False)
    num_items: int = field(init=False)
    # layers: List[int] = field(init=False)
    # latent_dim_gmf: int = field(init=False)
    # latent_dim_mlp: int = field(init=False)


@dataclass_json
@dataclass
class SDAEInfo(ModelInfo):
    epoch: int = 100
    batch_size: int = 1024
    train_test_ratio: float = 0.8
    learning_rate: int = 0.001
    dropout: float = 0.1
    corruption: float = 0.3
    regularization: float = 0.001
    hidden_layers: List[int] = field(default_factory=[32, 16])
    latent_layer: int = 8
    input_layer: int = field(init=False)


@dataclass_json
@dataclass
class SPEInfo(ModelInfo):
    epoch: int = 100
    batch_size: int = 1024
    train_test_ratio: float = 0.7
    learning_rate: float = 0.001
    regularization_spe: float = 0.01
    regularization_ncf: float = 0.01
    regularization_sdae_encode: float = 0.001
    regularization_sdae_decode: float = 0.001
    corruption: float = 0.3

