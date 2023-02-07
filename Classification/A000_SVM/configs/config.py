from pydantic import BaseModel
import yaml
import os


class SVM(BaseModel):
    feature_used: list
    n_features: int = 2
    kernel_svm: str = 'rbf'
    C_svm: list
    step: int = 10
    model_name: str
    plot_title: str


    
class Dataset(BaseModel):
    dataset_type: str = "list"
    fs: int = 40000
    resample: bool = True
    resample_rate: int = 32768
    valid_ratio: float = 0.1
    

class Train(BaseModel):
    increment: int = 10

    

class MainConfig(BaseModel):
    model: SVM = None
    train: Train = None
    dataset: Dataset = None


def load_config(yaml_filepath="config.yaml"):
    with open(yaml_filepath, "r") as config_f:
        try:
            config_dict = yaml.safe_load(config_f)
            model_dict = {
                "model": SVM(**config_dict["model"]),
                "train": Train(**config_dict["train"]),
                "dataset": Dataset(**config_dict["dataset"])

            }
            main_config = MainConfig(**model_dict)
            return main_config
        
        except yaml.YAMLError as e:
            print(e)


def save_config(main_config, config_name="train_test_config.yaml"):
    with open(os.path.join(config_name), 'w') as s:
        yaml.safe_dump(main_config.dict(), stream=s)