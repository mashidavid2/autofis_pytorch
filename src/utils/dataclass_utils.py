from inspect import signature
from inflection import underscore, camelize
from dataclasses import is_dataclass
from enum import EnumMeta
from classes.column import Column
from typing import List
import json


def from_dict(dataclass, dictionary):
    temp_dict = {}
    sig = signature(dataclass)
    for key, value in dictionary.items():
        if not is_first_uppercase(key):
            key = underscore(key)
        assert key in sig.parameters.keys(), \
            f'key ({key}) dose not exist in {dataclass.__name__} keys ({dataclass.__annotations__.keys()}), ' \
            f'({dataclass.__name__}: {dictionary})'
        key_class = sig.parameters[key].annotation
        if is_dataclass(key_class):
            temp_dict[key] = from_dict(key_class, value)
        elif (
            hasattr(key_class, '__origin__') and
            key_class.__origin__ == list
        ):
            if is_dataclass(key_class.__args__[0]):
                temp_dict[key] = [
                    from_dict(key_class.__args__[0], v) for v in value
                ]
        elif isinstance(key_class, EnumMeta):
            temp_dict[key] = key_class(value)
        else:
            assert isinstance(value, dataclass.__annotations__[key]), \
                    f'{key} type must be {dataclass.__annotations__[key]} (type: {type(value)}, value: {value}))'
            temp_dict[key] = value
    return dataclass(**temp_dict)


def is_first_uppercase(s):
    return s[0].upper() == s[0]


def from_json(dataclass, json_str):
    return from_dict(dataclass, json.loads(json_str))

def yaml_to_model_dict(yaml):
    yaml = yaml['modelConfigs'][0]
    model_name = yaml['key']
    model_dict = dict()
    model_dict[model_name] = dict()
    model_dict[model_name]['model_name'] = model_name
    #model_dict['model_name'] = yaml['key']
    for hyperparameter in yaml['hyperparameters']:
        model_dict[model_name][hyperparameter['key']] = hyperparameter['value']
    return model_name, model_dict

def yaml_to_config_dict(yaml):
    configs = yaml['dataConfigs']
    config_dict = dict()
    for config in configs:
        for con in config['configurations']:
            if con['type'] == 'Column':
                config_dict[con['key']] = Column
            elif con['type'] == 'Columns':
                config_dict[con['key']] = List[Column]
            else:
                raise Exception('Invalid configuration')
    return config_dict
