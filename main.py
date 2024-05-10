import argparse
from utils.datasets import get_dataset
from train import train_eval
from utils import tools
import time
from pprint import pprint
import random
from model import MVANA
import os
def get_parser():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--ds_name', default='NCI1',choices=['NCI1', 'NCI109', 'REDDIT-MULTI-12K',"PROTEINS", "DD", "Mutagenicity"])
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    ds_name = args.ds_name
    print(f"************** dataset is {ds_name}********************")
    device = tools.get_best_free_gpu()
    seed = 8971
    tools.set_seed(seed)

    # *******************************load config******************************
    config = tools.load_json("config/CommonConfig.json")
    ds_config_loc = f"config/{ds_name}.json"
    print("")
    if not os.path.exists(ds_config_loc):
        ds_config = {}
    else:
        ds_config = tools.load_json(f"config/{ds_name}.json")
    train_config = config["train_config"]
    net_config = config["net_params"]
    config["dataset"] = ds_name
    # if ds_name not in ["Mutagenicity","NCI1", "NCI109"]:
    #     train_config["batch_size"] = 64
    # else:
    #     train_config["batch_size"] = 512

    for key,value in ds_config.items():
        if key in train_config.keys():
            train_config[key] = value
        else:
            net_config[key] = value

    dataset = get_dataset(ds_name, config["data_dir"])
    num_feature, num_classes = dataset.num_features, dataset.num_classes
    net_config["device"] = device
    net_config["in_channels"] = num_feature
    net_config["out_channels"] = num_classes
    pprint(net_config)
    time_str = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y_') + str(random.randint(0, 100))
    acc, std, duration_mean = train_eval.cross_validation(ds_name,"MVANA",dataset,seed,MVANA,device,config,time_str)
    print(f"dataset name is {ds_name}, test acc is {acc}, test std is {std}, duration mean is {duration_mean}")
