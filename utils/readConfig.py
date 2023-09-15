import yaml
from pathlib import Path

def read_config():
    with open(Path(__file__).parent / './config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config