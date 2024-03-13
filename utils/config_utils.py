import json
import os
from bunch import Bunch
from utils.utils import mkdir_if_not_exist


def get_config_from_json(json_file):
    """
    :param json_file
    :return: config class
    """
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    config = Bunch(config_dict)
    config.tb_dir = os.path.join("experiments", config.exp_name, "logs/")
    config.cp_dir = os.path.join("experiments", config.exp_name, "checkpoints/")

    mkdir_if_not_exist(config.tb_dir)
    mkdir_if_not_exist(config.cp_dir)

    return config
