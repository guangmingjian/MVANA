from model import MVANA
import torch
import os
from utils import tools
from utils.datasets import get_dataset
from train.train_eval import k_fold,evaluate_network
from torch_geometric.data import DataLoader
import numpy as np
from pprint import pprint

#  NCI1,NCI109,REDDIT-MULTI-12K
ds_name = "NCI1"
gpu_id = 0

ds_config = tools.load_json(f"config/{ds_name}.json")
config = tools.load_json("config/CommonConfig.json")
# updata config
net_config = config["net_params"]
train_config = config["train_config"]
for key, value in ds_config.items():
    if key in net_config:
        net_config[key] = value
    else:
        train_config[key] = value
device = "cuda:" + str(gpu_id)
# device = "cpu"
pprint(net_config)
pprint(train_config)

seed = 8971
tools.set_seed(seed)
dataset = get_dataset(ds_name, config["data_dir"])
num_feature, num_classes = dataset.num_features, dataset.num_classes
net_config["device"] = device
net_config["in_channels"] = num_feature
net_config["out_channels"] = num_classes
model = MVANA(**net_config)
model.to(device)
print(model)
all_accs = []
for fold, (train_idx, test_idx,
           val_idx) in enumerate(zip(*k_fold(dataset, 10, seed))):

    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]
    val_dataset = dataset[val_idx]
    train_loader = DataLoader(train_dataset, train_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, train_config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, train_config["batch_size"], shuffle=False)
    model.load_state_dict(torch.load(os.path.join("records",ds_name,"models",
                                                  f"fold_{fold}.pth"), map_location=f'cuda:{gpu_id}'))
    _,acc = evaluate_network(model,device,test_loader)
    all_accs.append(acc)
    print(f"fold {fold + 1}, test acc is {acc}")
acc_mean = np.mean(np.array(all_accs))
acc_std = np.std(np.array(all_accs))
print("Test Accuracy: {:.4f} ± {:.4f}".format(acc_mean * 100 ,acc_std * 100))