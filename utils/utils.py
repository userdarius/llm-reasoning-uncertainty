# utils/utils.py
import yaml
import logging


def load_config(config_path):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def setup_logging():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger


# Example usage:
# config = load_config('config/config.yaml')
# logger = setup_logging()
