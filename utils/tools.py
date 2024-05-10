import json
import os

import numpy as np
import torch
import torch_geometric as tgnn
from torch import nn
import random

def get_best_free_gpu(gpu_list: list = [0, 1]):
    # print(gpu_list)
    gpu_free = []
    for gpu in gpu_list:
        gpu_free.append(tgnn.profile.get_gpu_memory_from_nvidia_smi(gpu)[0])
    gpu_free = np.argsort(gpu_free)[::-1]
    return "cuda:%s" % gpu_free[0]


def load_json(json_loc: str) -> dict:
    with open(json_loc) as f:
        config = json.load(f)
    return config


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("dir is created successful")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()


def gpu_setup(use_gpu, gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda:" + str(gpu_id))
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


def save_data_to_json(outfile, data):
    with open(outfile, 'w') as f:
        json.dump(data, f)


def write_dirs(model_name, ex_out_dir, ds_name, out_dir, time_str):
    com_loc = os.path.join(model_name, ds_name, ex_out_dir)
    root_log_dir = out_dir + 'logs/' + com_loc
    root_ckpt_dir = out_dir + 'checkpoints/' + com_loc
    write_file_name = out_dir + 'results/' + com_loc
    write_config_file = out_dir + 'configs/' + com_loc

    if not os.path.exists(write_file_name):
        os.makedirs(write_file_name)
    if not os.path.exists(write_config_file):
        os.makedirs(write_config_file)
    dirs = root_log_dir + time_str, root_ckpt_dir + time_str, write_file_name + time_str, write_config_file + time_str
    return dirs
