import yaml
import os 
def load_config():
    #looks for congi.yaml in root dir
    config_path=os.path.join(os.path.dirname(os.path.dirname(__file__)),"config.yaml")

    with open(config_path,"r") as f:
        return yaml.safe_load(f)

SETTINGS=load_config()# global config

