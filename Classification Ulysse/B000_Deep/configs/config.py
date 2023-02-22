from pydantic import BaseModel
import yaml
import os


class CNN_MLP(BaseModel):
    output_dim: int
    ratios_CNN: list
    channel_size: list
    size_MLP: list
    model_name: str = "CNN_MLP"


    
class Dataset(BaseModel):
    batch_size: int = 32
    valid_ratio: float = 0.2
    num_thread: int = 0
    dataset_type: str = "list"
    fs: int = 40000
    resample: bool = True
    resample_rate: int = 32768
    

class Train(BaseModel):
    lr: float
    epochs: int 
    save_ckpt: int = 5
    add_fig: int = 5
    loss: str = "MSE"

    

class MainConfig(BaseModel):
    model: CNN_MLP = None
    train: Train = None
    dataset: Dataset = None


def load_config(yaml_filepath="config.yaml"):
    with open(yaml_filepath, "r") as config_f:
        try:
            config_dict = yaml.safe_load(config_f)
            model_dict = {
                "model": CNN_MLP(**config_dict["model"]),
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