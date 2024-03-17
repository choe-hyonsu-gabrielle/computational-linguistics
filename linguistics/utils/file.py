import os
import json
import glob
import pickle
from os.path import exists
from typing import Any, Union
from tqdm import tqdm


def load_json(filename: str, encoding: str = 'utf-8'):
    """ json 파일 하나를 불러 오고 데이터를 반환
    :param filename: filename
    :param encoding: encoding option for open()
    :return: json data
    """
    with open(filename, encoding=encoding) as fp:
        return json.load(fp)


def load_jsons(filepaths: list[str], encoding: str = 'utf-8', flatten: bool = False) -> list:
    """ 여러 개의 json 파일을 불러 오고 데이터를 list 형태로 반환
    :param filepaths: filepath list (ex. ["./1.json", "./2.json", ...]
    :param encoding: encoding option for open()
    :param flatten: 불러온 모든 데이터를 단일 list로 반환
    :return: list of json data
    """
    results = []
    for filepath in tqdm(filepaths, desc=f'- loading {len(filepaths)} files'):
        assert filepath.endswith('.json'), filepath
        if not flatten:
            results.append(load_json(filename=filepath, encoding=encoding))
        else:
            results.extend(load_json(filename=filepath, encoding=encoding))
    return results


def get_absolute_root_path(root_name='synthetics', suffix='\\'):
    """ get absolute root path
    :param: root_name: dir name for project root.
    :param: suffix: path delimiter
    :return: If the call location is "C:/Users/.../Projects/synthetics/dir1/demo",
             then return will be "C:/Users/.../Projects/synthetics"
    """
    current_location = os.getcwd()
    cut = current_location.rfind(root_name)
    return ''.join([current_location[:cut], root_name]) + suffix


def save_pickle(filename: str, instance: Any):
    with open(filename, 'wb') as fp:
        pickle.dump(instance, fp, pickle.HIGHEST_PROTOCOL)


def load_pickle(filename: str):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)