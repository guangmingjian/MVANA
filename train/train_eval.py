import shutil
import optuna
import os
import time
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from sklearn.model_selection import KFold
import numpy as np
from utils import tools
from functools import reduce
from torch import nn


# def weight_reset(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         m.reset_parameters()


def cross_validation(ds_name, model_name, dataset, seed, Net,
                     device, config, time_str):
    accs, durations = [], []
    net_config = config["net_params"]
    train_paras = config["train_config"]
    folds = train_paras["kf"]
    batch_size = train_paras["batch_size"]
    lr = train_paras["lr"]
    weight_decay = train_paras["weight_decay"]
    patience = train_paras["patience"]
    epochs = train_paras["epochs"]
    result_content = "Dataset: {},\nModel: {}\n\nparams={}\n\nnet_params={}\n\n" \
        .format(ds_name, model_name, train_paras, net_config)
    write_re_root = config["out_dir"] + "result/%s/" % ds_name
    model_loc = write_re_root + "%s/models/" % time_str
    txt_loc = write_re_root + "%s/" % time_str
    tools.mkdir(model_loc)

    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds, seed))):

        train_dataset = dataset[train_idx]
        test_dataset = dataset[test_idx]
        val_dataset = dataset[val_idx]
        # print(train_idx)
        # print(test_idx)
        # print(val_idx)

        if 'adj' in train_dataset[0]:
            train_loader = DenseLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DenseLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DenseLoader(test_dataset, batch_size, shuffle=False)
        else:

            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        model = Net(**net_config)
        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        if fold == 0:
            print(model)
            result_content = result_content + "model={}\n\n".format(model)

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        t_start = time.perf_counter()

        best_acc = 0
        max_patience = 0
        # fold_test_acc = 0
        #
        # all_epoch_info = {
        #     "epoch": [],
        #     "train_loss": [],
        #     "train_acc": [],
        #     "val_loss": [],
        #     "val_acc": [],
        #     "test_loss": [],
        #     "test_acc": [],
        #     "epoch_time": []
        # }

        for epoch in range(1, epochs + 1):
            start = time.time()
            epoch_train_loss, epoch_train_acc, optimizer = train_epoch(model, optimizer, device, train_loader)
            epoch_time = time.time() - start
            epoch_val_loss, epoch_val_acc = evaluate_network(model, device, val_loader)
            # epoch_test_loss, epoch_test_acc = evaluate_network(model, device, test_loader)

            # for key, value in zip(list(all_epoch_info.keys()),
            #                       [epoch, epoch_train_loss, epoch_train_acc, epoch_val_loss, epoch_val_acc,
            #                        epoch_test_loss, epoch_test_acc, epoch_time]):
            #     all_epoch_info[key].append(value)

            # epoch_test_loss, epoch_test_acc = evaluate_network(model, device, test_loader)
            if epoch % 10 == 0:
                # epoch_test_loss, epoch_test_acc = evaluate_network(model, device, test_loader)
                eval_info = {
                    'fold': fold + 1,
                    'epoch': epoch,
                    'train_loss': np.round(epoch_train_loss, 3),
                    # 'val_loss': np.round(epoch_val_loss, 3),
                    'train_acc': np.round(epoch_train_acc, 3),
                    'val_acc': np.round(epoch_val_acc, 3),
                    # 'test_acc': np.round(epoch_test_acc, 3),
                    'mean_acc': np.round(np.mean(accs), 3)
                }
                result_content = result_content + "{}\n".format(eval_info)

                print(eval_info)

            if epoch_val_acc > best_acc or epoch == 1:
                torch.save(model.state_dict(), os.path.join(model_loc, f"{fold + 1}.pth"))
                print(
                    "Model saved at epoch {}, val_acc is {}, epoch_time is {}".format(
                        epoch,
                        epoch_val_acc, epoch_time))
                result_content = result_content + "Model saved at epoch {}, val_acc is {} \n".format(
                    epoch,
                    epoch_val_acc
                    )
                best_acc = epoch_val_acc
                max_patience = 0

            else:
                max_patience += 1
            if max_patience > patience:
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        t_end = time.perf_counter()
        durations.append(t_end - t_start)

        model.load_state_dict(torch.load(os.path.join(model_loc,
                                                      f"{fold + 1}.pth"), map_location=device))
        _,fold_test_acc = evaluate_network(model, device, test_loader)
        accs.append(fold_test_acc)

        print("For fold {}, test acc: {:.6f}".format(fold + 1, fold_test_acc))
        result_content = result_content + "For fold {}, test acc: {:.6f}\n\n".format(fold + 1, fold_test_acc)

    acc_mean = np.mean(np.array(accs))
    acc_std = np.std(np.array(accs))
    duration_mean = np.mean(np.array(durations))
    print('Test Accuracy: {:.4f} ± {:.4f}, Duration: {:.4f}'.
          format(acc_mean * 100, acc_std * 100, duration_mean))
    result_content = result_content + "\nTest Accuracy: {:.4f} ± {:.4f}, Duration: {:.4f}\n\n\nAll Splits Test Accuracies: {}\n\n".format(
        acc_mean * 100, acc_std * 100, duration_mean, accs)

    with open(os.path.join(txt_loc, "result.txt"), 'w') as f:
        f.write(result_content)
    torch.cuda.empty_cache()
    return acc_mean, acc_std, duration_mean




def train_epoch(model, optimizer, device, data_loader):
    model.train()
    epoch_loss = 0
    nb_data = 0
    total_curr = 0
    for iter, data in enumerate(data_loader):
        optimizer.zero_grad()
        data = data.to(device)
        batch_targets = data.y
        # num_nodes = data.num_nodes
        data, edge_index, batch = data.x, data.edge_index, data.batch
        # if data == None:
        #     data = torch.randn([num_nodes, 2]).float().to(device)
        batch_pres = model.forward(data, edge_index, batch)

        cur_pre = cal_curr_num(batch_pres, batch_targets)
        loss = model.loss(batch_pres, batch_targets.view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        nb_data += batch_targets.size(0)
        total_curr += cur_pre
    epoch_loss /= (iter + 1)
    acc = total_curr / nb_data

    return epoch_loss, acc, optimizer


def cal_curr_num(scores, targets):
    pred = scores.detach().max(dim=1)[1]  # 返回每一行最大值对应的索引
    acc = pred.eq(targets.view(-1)).sum().item()
    return acc


def evaluate_network(model, device, data_loader):
    model.eval()
    epoch_test_loss = 0
    nb_data = 0
    total_curr = 0
    with torch.no_grad():
        for iter, data in enumerate(data_loader):
            data = data.to(device)
            batch_targets = data.y
            data, edge_index, batch = data.x, data.edge_index, data.batch
            batch_pres = model.forward(data, edge_index, batch)
            cur_pre = cal_curr_num(batch_pres, batch_targets)
            loss = F.nll_loss(batch_pres, batch_targets)
            epoch_test_loss += loss.detach().item()
            total_curr += cur_pre
            # nb_data += batch_targets.size(0)
        epoch_test_loss /= (iter + 1)
        acc = total_curr / len(data_loader.dataset)
    return epoch_test_loss, acc


def k_fold(dataset, folds, seed):
    skf = KFold(folds, shuffle=True, random_state=seed)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(torch.from_numpy(idx).to(torch.long))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.bool)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))

    return train_indices, test_indices, val_indices


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device):
    model.train()

    total_loss = 0
    for data in loader:
        optimizer.zero_grad()
        data = data.to(device)
        out = model(data)
        loss = F.nll_loss(out, data.y.view(-1))
        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
    return total_loss / len(loader.dataset)


def eval_acc(model, loader, device):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_loss(model, loader, device):
    model.eval()

    loss = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            out = model(data)
        loss += F.nll_loss(out, data.y.view(-1), reduction='sum').item()
    return loss / len(loader.dataset)
