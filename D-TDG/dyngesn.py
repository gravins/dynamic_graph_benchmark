import os
import torch
import ray

import numpy as np
import argparse
import datetime
import random
import pandas
import tqdm
import time

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, balanced_accuracy_score

from graphesn import DynamicGraphReservoir, initializer, Readout   
from graphesn.util import compute_dynamic_graph_alpha

# Best hyper-parameters:
# elliptic:      
#   batch = 8
#   units = 16     # embedding dim
#   sigma = 0.9    # random weight init. value
#   leakage = 0.9
#   lr = 0.01
#   wd = 0.0001
# as_733:        
#   batch = 32
#   units = 32
#   sigma = 0.9
#   leakage = 0.1
#   lr = 0.01
#   wd = 0.0001
# bitcoin_alpha: 
#   batch = 32
#   units = 32
#   sigma = 0.9
#   leakage = 0.1
#   lr = 0.01
#   wd = 0.0001
# twitter:       
#   batch = 32
#   units = 32
#   sigma = 0.9
#   leakage = 0.5
#   lr = 0.01
#   wd = 0.001
#
# How to run this experiments:
# python3 -u dyngesn.py --dataset $dataset --path $path --units 32 16 8 --sigma 0.9 0.5 0.1 --leakage 0.9 0.5 0.1 --lr 0.01 0.001 0.0001 --wd 0.001 0.0001 --batch $batch


#ray.init() # local ray initialization
ray.init(address=os.environ.get("ip_head"), _redis_password=os.environ.get("redis_password"))  # clustering ray initialization


elliptic = 'elliptic'
twitter = 'twitter'
as_733 = 'as_733'
bitcoin_alpha = 'bitcoin_alpha'

def prepare_data(path, compute_alpha=True):
    datalist = torch.load(path)
    edge_index_list = [d.edge_index for d in datalist]
    x_list = [d.x for d in datalist]
    alpha = compute_dynamic_graph_alpha(edge_index_list) if compute_alpha else None
    return datalist, x_list, edge_index_list, alpha


def compute_metrics(dataset_name, y_true, y_pred, y_conf=None):
    if dataset_name == twitter:
        score = {
            'mae': torch.nn.functional.l1_loss(y_pred, y_true).cpu().item(),
            'mse': torch.nn.functional.mse_loss(y_pred, y_true).cpu().item()
        }
    else:
        score = {
            'auroc': roc_auc_score(y_true, y_conf),
            'f1': f1_score(y_true, y_pred),
            'acc': accuracy_score(y_true, y_pred),
            'balanced_acc': balanced_accuracy_score(y_true, y_pred)
        }
    return score


def eval(readout, X, y, datalist, T_range, criterion, device, dataset_name):
    readout.eval()
    y_true, y_conf = [], []
    with torch.no_grad():
        for t in T_range:
            snapshot = datalist[t]
            y_conf.append(readout(X[t].to(device), snapshot.to(device)).cpu().detach())
            y_true.append(y[t].cpu().detach())

    y_conf = torch.cat(y_conf)
    y_true = torch.cat(y_true)

    # Compute loss
    loss = criterion(y_conf, y_true).cpu().item()

    # Compute metrics
    if dataset_name != twitter:
        y_conf = torch.sigmoid(y_conf)
        y_pred = (y_conf > 0.5).float()
    score = compute_metrics(dataset_name, y_true, y_pred, y_conf)

    return loss, score


class LinearReadout(torch.nn.Module):
    def __init__(self, num_features, num_targets, link_prediction=False) -> None:
        super().__init__()
        self.readout = torch.nn.Linear(num_features * 2 if link_prediction else num_features,
                                       num_targets)
    
    def forward(self, X, snapshot):
        if 'link_pred_ids' in snapshot:
            source, target = snapshot.link_pred_ids
            x = torch.cat((X[source], X[target]),dim=-1)
        else:
            x = X
            if hasattr(snapshot, 'node_mask'):
                x = x[snapshot.node_mask]
        return self.readout(x)


@ray.remote(num_cpus=2)
def train_eval_ridge(path, alpha, units, sigma, leakage, ld, dataset_name, num_trials, device):
    datalist, x_list, edge_index_list, _ = prepare_data(path, False)
    T_train, T_valid = int(len(edge_index_list)*0.70), int(len(edge_index_list)*0.85)
    Y = [d.y for d in datalist]

    results = {
        'alpha': alpha,
        'units': units,
        'sigma': sigma,
        'leakage': leakage,
        'ld': ld
    }
    if dataset_name == twitter:
        train_mae, val_mae, test_mae, train_mse, val_mse, test_mse = [], [], [], [], [], []
    else:
        train_auroc, val_auroc, test_auroc = [], [], []
        train_f1, val_f1, test_f1 = [], [], []
        train_acc, val_acc, test_acc = [], [], []
        train_balanced_acc, val_balanced_acc, test_balanced_acc = [], [], []

    for trial_index in range(num_trials):
        # Set the seed for the new trial
        torch.manual_seed(trial_index)
        torch.cuda.manual_seed_all(trial_index)
        random.seed(trial_index)
        np.random.seed(trial_index)

        reservoir = DynamicGraphReservoir(num_layers=1, in_features=datalist[0].x.shape[-1], hidden_features=units, return_sequences=True)
        reservoir.initialize_parameters(recurrent=initializer('uniform', sigma=sigma / alpha),
                                        input=initializer('uniform', scale=1),
                                        leakage=leakage)
        reservoir.to(device)
        X = reservoir(edge_index=edge_index_list, input=x_list)
        
        readout = Readout(num_features=units, num_targets=datalist[0].y.shape[-1])
        if dataset_name != twitter:
            m, M = Y[0].min(), Y[0].max() 
            for y in Y:
                el_min = y.min()
                el_max = y.max()
                m = el_min if el_min < m else m
                M = el_max if el_max > M else M
            for i in range(len(Y)):
                Y[i][Y[i]==m] = -1
                Y[i][Y[i]==M] = 1
            
        if dataset_name in [as_733, bitcoin_alpha]:
            XX = []
            for x, snapshot in zip(X, datalist):
                source, target = snapshot.link_pred_ids
                XX.append(torch.cat((x[source], x[target]),dim=-1))
            X_tr = torch.concat(XX[:T_train])
            X_vl = torch.concat(XX[T_train:T_valid])
            X_ts = torch.concat(XX[T_valid:])
            X = torch.concat(XX)
        elif hasattr(datalist[0], 'node_mask'):
            XX = []
            for x, snapshot in zip(X, datalist):
                node_mask = snapshot.node_mask
                XX.append(x[node_mask])
            X_tr = torch.concat(XX[:T_train])
            X_vl = torch.concat(XX[T_train:T_valid])
            X_ts = torch.concat(XX[T_valid:])
            X = torch.concat(XX)
        else:
            X_tr = X[:T_train].view(-1, X.shape[-1])
            X_vl = X[T_train:T_valid].view(-1, X.shape[-1])
            X_ts = X[T_valid:].view(-1, X.shape[-1])

        readout.fit(data=(
                    X_tr, torch.concat(Y[:T_train])),
                    regularization=ld)

        y_pred_tr = readout(X_tr)
        y_pred_vl = readout(X_vl)
        y_pred_ts = readout(X_ts)

        y_true_tr = torch.concat(Y[:T_train])
        y_true_vl = torch.concat(Y[T_train:T_valid])
        y_true_ts = torch.concat(Y[T_valid:])

        if dataset_name != twitter:
            y_pred_tr = (y_pred_tr > 0).float()
            y_pred_vl = (y_pred_vl > 0).float()
            y_pred_ts = (y_pred_ts > 0).float()
            y_true_tr = (y_true_tr > 0).float()
            y_true_vl = (y_true_vl > 0).float()
            y_true_ts = (y_true_ts > 0).float()

        score_tr = compute_metrics(dataset_name, y_true_tr, y_pred_tr)
        score_vl = compute_metrics(dataset_name, y_true_vl, y_pred_vl)
        score_ts = compute_metrics(dataset_name, y_true_ts, y_pred_ts)

        if dataset_name == twitter:
            train_mae.append(score_tr['mae'])
            val_mae.append(score_vl['mae'])
            test_mae.append(score_ts['mae'])
            train_mse.append(score_tr['mse'])
            val_mse.append(score_vl['mse'])
            test_mse.append(score_ts['mse'])
        else:
            train_auroc.append(score_tr['auroc'])
            val_auroc.append(score_vl['auroc'])
            test_auroc.append(score_ts['auroc'])

            train_f1.append(score_tr['f1'])
            val_f1.append(score_vl['f1'])
            test_f1.append(score_ts['f1'])
            
            train_acc.append(score_tr['acc'])
            val_acc.append(score_vl['acc'])
            test_acc.append(score_ts['acc'])

            train_balanced_acc.append(score_tr['balanced_acc'])
            val_balanced_acc.append(score_vl['balanced_acc'])
            test_balanced_acc.append(score_ts['balanced_acc'])

    if dataset_name == twitter:
        results.update({
            'mean_train_mae': np.mean(train_mae), 
            'mean_val_mae': np.mean(val_mae), 
            'mean_test_mae': np.mean(test_mae), 
            'mean_train_mse': np.mean(train_mse), 
            'mean_val_mse': np.mean(val_mse), 
            'mean_test_mse': np.mean(test_mse),
            'train_mae': train_mae, 
            'val_mae': val_mae, 
            'test_mae': test_mae, 
            'train_mse': train_mse, 
            'val_mse': val_mse, 
            'test_mse': test_mse
        })
    else:
        results.update({
            'mean_train_auroc': np.mean(train_auroc),
            'mean_val_auroc': np.mean(val_auroc),
            'mean_test_auroc': np.mean(test_auroc),
            'mean_train_f1': np.mean(train_f1),
            'mean_val_f1': np.mean(val_f1),
            'mean_test_f1': np.mean(test_f1),
            'mean_train_acc': np.mean(train_acc),
            'mean_val_acc': np.mean(val_acc),
            'mean_test_acc': np.mean(test_acc), 
            'train_auroc': train_auroc,
            'val_auroc': val_auroc,
            'test_auroc': test_auroc,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_balanced_acc': train_balanced_acc,
            'val_balanced_acc': val_balanced_acc,
            'test_balanced_acc': test_balanced_acc
        })
    return results


@ray.remote(num_cpus=2)
def train_eval(batch_size, path, alpha, units, sigma, leakage, lr, wd, dataset_name, num_trials, device):
    datalist, x_list, edge_index_list, _ = prepare_data(path, False)
    T_train, T_valid = int(len(edge_index_list)*0.70), int(len(edge_index_list)*0.85)
    y = [d.y for d in datalist]

    collate_fn = lambda samples_list: samples_list
    tr_loader = DataLoader(datalist[:T_train], batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    results = {
        'alpha': alpha,
        'units': units,
        'sigma': sigma,
        'leakage': leakage,
        #'ld': ld,
        'lr': lr,
        'wd': wd
    }
    if dataset_name == twitter:
        train_mae, val_mae, test_mae, train_mse, val_mse, test_mse = [], [], [], [], [], []
    else:
        train_auroc, val_auroc, test_auroc = [], [], []
        train_f1, val_f1, test_f1 = [], [], []
        train_acc, val_acc, test_acc = [], [], []
        train_balanced_acc, val_balanced_acc, test_balanced_acc = [], [], []

        train_loss, val_loss, test_loss = [], [], []

    for trial_index in range(num_trials):
        # Set the seed for the new trial
        torch.manual_seed(trial_index)
        torch.cuda.manual_seed_all(trial_index)
        random.seed(trial_index)
        np.random.seed(trial_index)

        reservoir = DynamicGraphReservoir(num_layers=1, in_features=datalist[0].x.shape[-1], hidden_features=units, return_sequences=True)
        reservoir.initialize_parameters(recurrent=initializer('uniform', sigma=sigma / alpha),
                                        input=initializer('uniform', scale=1),
                                        leakage=leakage)
        reservoir.to(device)
        X = reservoir(edge_index=edge_index_list, input=x_list)
        
        if dataset_name == twitter:
            train_mae.append(np.inf)
            val_mae.append(np.inf)
            test_mae.append(np.inf)
            train_mse.append(np.inf)
            val_mse.append(np.inf)
            test_mse.append(np.inf)
        else:
            train_auroc.append(-np.inf)
            val_auroc.append(-np.inf)
            test_auroc.append(-np.inf)

            train_f1.append(-np.inf)
            val_f1.append(-np.inf)
            test_f1.append(-np.inf)
            
            train_acc.append(-np.inf)
            val_acc.append(-np.inf)
            test_acc.append(-np.inf)

            train_balanced_acc.append(-np.inf)
            val_balanced_acc.append(-np.inf)
            test_balanced_acc.append(-np.inf)

            train_loss.append(np.inf)
            val_loss.append(np.inf)
            test_loss.append(np.inf)

        readout = LinearReadout(num_features=units, num_targets=datalist[0].y.shape[-1], link_prediction=dataset_name in [as_733, bitcoin_alpha])
        criterion = torch.nn.L1Loss() if dataset_name == twitter else torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(readout.parameters(), lr=lr, weight_decay=wd)
        best_score = np.inf if dataset_name == twitter else 0.
        best_epoch = 0
        for epoch in range(1000):
            t = 0
            for batch in tr_loader:
                # Reset gradients from previous step
                readout.zero_grad()

                y_pred, y_true = [], []
                for snapshot in batch:
                    out = readout(X[t].to(device), snapshot.to(device))
                    
                    y_pred.append(out.cpu())
                    y_true.append(snapshot.y.cpu())
                    t += 1

                # Perform a backward pass to calculate the gradients
                y_pred = torch.cat(y_pred)
                y_true = torch.cat(y_true)
                loss = criterion(y_pred, y_true)
                loss.backward()

                # Update parameters
                optimizer.step()

            loss_tr, score_tr = eval(readout, X, y, datalist, range(T_train), criterion, device, dataset_name)
            loss_vl, score_vl = eval(readout, X, y, datalist, range(T_train,T_valid), criterion, device, dataset_name)
            loss_ts, score_ts = eval(readout, X, y, datalist, range(T_valid, len(datalist)), criterion, device, dataset_name)
            
            if dataset_name == twitter:
                if score_vl['mae'] < best_score:
                    best_score = score_vl['mae']
                    best_epoch = epoch
                    train_mae[-1] = score_tr['mae']
                    val_mae[-1] = score_vl['mae']
                    test_mae[-1] = score_ts['mae']
                    train_mse[-1] = score_tr['mse']
                    val_mse[-1] = score_vl['mse']
                    test_mse[-1] = score_ts['mse']
            else:
                tmp_score = score_vl['balanced_acc'] if dataset_name == elliptic else score_vl['auroc']
                if tmp_score > best_score:
                    best_score = score_vl['auroc']
                    best_epoch = epoch
                    train_auroc[-1] = score_tr['auroc']
                    val_auroc[-1] = score_vl['auroc']
                    test_auroc[-1] = score_ts['auroc']
                    train_f1[-1] = score_tr['f1']
                    val_f1[-1] = score_vl['f1']
                    test_f1[-1] = score_ts['f1']
                    train_acc[-1] = score_tr['acc']
                    val_acc[-1] = score_vl['acc']
                    test_acc[-1] = score_ts['acc']
                    train_balanced_acc[-1] = score_tr['balanced_acc']
                    val_balanced_acc[-1] = score_vl['balanced_acc']
                    test_balanced_acc[-1] = score_ts['balanced_acc']
                    train_loss[-1] = loss_tr
                    val_loss[-1] = loss_vl
                    test_loss[-1] = loss_ts

            if epoch - best_epoch > 50:
                break

    if dataset_name == twitter:
        results.update({
            'mean_train_mae': np.mean(train_mae), 
            'mean_val_mae': np.mean(val_mae), 
            'mean_test_mae': np.mean(test_mae), 
            'mean_train_mse': np.mean(train_mse), 
            'mean_val_mse': np.mean(val_mse), 
            'mean_test_mse': np.mean(test_mse),
            'train_mae': train_mae, 
            'val_mae': val_mae, 
            'test_mae': test_mae, 
            'train_mse': train_mse, 
            'val_mse': val_mse, 
            'test_mse': test_mse
        })
    else:
        results.update({
            'mean_train_auroc': np.mean(train_auroc),
            'mean_val_auroc': np.mean(val_auroc),
            'mean_test_auroc': np.mean(test_auroc),
            'mean_train_f1': np.mean(train_f1),
            'mean_val_f1': np.mean(val_f1),
            'mean_test_f1': np.mean(test_f1),
            'mean_train_acc': np.mean(train_acc),
            'mean_val_acc': np.mean(val_acc),
            'mean_test_acc': np.mean(test_acc), 
            'mean_train_loss': np.mean(train_loss),
            'mean_val_loss': np.mean(val_loss),
            'mean_test_loss': np.mean(test_loss),
            'train_auroc': train_auroc,
            'val_auroc': val_auroc,
            'test_auroc': test_auroc,
            'train_f1': train_f1,
            'val_f1': val_f1,
            'test_f1': test_f1,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'test_acc': test_acc,
            'train_balanced_acc': train_balanced_acc,
            'val_balanced_acc': val_balanced_acc,
            'test_balanced_acc': test_balanced_acc,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'test_loss': test_loss,
        })
    return results

if __name__ == "__main__":
    t0 = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', help='dataset name')
    parser.add_argument('--path', help='dataset path') # eg /data/as_733.pt
    parser.add_argument('--units', help='reservoir units per layer', type=int, nargs='+', default=[32])
    parser.add_argument('--sigma', help='sigma for recurrent matrix initialization', type=float, nargs='+', default=[0.9])
    parser.add_argument('--leakage', help='leakage constant', type=float, nargs='+', default=[0.9])
    parser.add_argument('--ld', help='readout lambda', type=float, nargs='+', default=[1e-3])
    parser.add_argument('--lr', help='learning rate', type=float, nargs='+', default=[0.01])
    parser.add_argument('--wd', help='weight decay', type=float, nargs='+', default=[0.001])
    parser.add_argument('--batch', help='batch size', type=int, default=8)
    parser.add_argument('--trials', help='number of trials', type=int, default=5)
    parser.add_argument('--device', help='device', type=str, default='cpu')
    parser.add_argument('--ridge', help='use ridge regression/classification as readout', action='store_true')
    args = parser.parse_args()

    assert args.dataset in [as_733, bitcoin_alpha, elliptic, twitter]

    device = torch.device(args.device)

    if args.dataset == as_733:
        print('alpha precomputed')
        alpha = 38.01178806531002
    elif args.dataset == bitcoin_alpha:
        print('alpha precomputed')
        alpha = 1.532752722140408
    elif args.dataset == elliptic:
        alpha = 6.471870352526263
    else:
        _, _, _, alpha = prepare_data(args.path, True)
    print(f'alpha = {alpha}')

    ray_ids = []
    for units in args.units:
        for sigma in args.sigma:
            for leakage in args.leakage:
                for ld in args.ld if args.ridge else [None]:
                    for lr in args.lr if not args.ridge else [None]:
                        for wd in args.wd if not args.ridge else [None]:
                            if args.ridge:
                                ray_ids.append(train_eval_ridge.remote(args.path, alpha, units, sigma, leakage, ld, args.dataset, args.trials, device))
                            else:
                                ray_ids.append(train_eval.remote(args.batch, args.path, alpha, units, sigma, leakage, lr, wd, args.dataset, args.trials, device))
                                    
    all_res = []
    for id_ in tqdm.tqdm(ray_ids):
        res = ray.get(id_) 
        all_res.append(res)
        torch.save(all_res, f'RESULTS/{args.dataset}_dyngesn_{"ridge_" if args.ridge else ""}ckpts.pt')
        
    pandas.DataFrame(all_res).to_csv(f'RESULTS/{args.dataset}_dyngesn_{"ridge_" if args.ridge else ""}results.csv', index=False)
    elapsed = time.time() - t0
    print(str(datetime.timedelta(seconds=int(round((elapsed))))))



#readout = Readout(num_features=units, num_targets=datalist[0].y.shape[-1])
#y = torch.stack(y)
#readout.fit(data=(X[:T_train].view(-1, X.shape[-1]), y[:T_train].view(-1, y.shape[-1])), regularization=ld,
#            validate=lambda weights: validate_on(weights, X[T_train:T_valid].view(-1, X.shape[-1]), y[T_train:T_valid].view(-1, y.shape[-1])))
#mae_tr, mse_tr = test_on((readout.weight, readout.bias), X[:T_train].view(-1, X.shape[-1]), y[:T_train].view(-1, y.shape[-1]))
#mae_vl, mse_vl = test_on((readout.weight, readout.bias), X[T_train:T_valid].view(-1, X.shape[-1]), y[T_train:T_valid].view(-1, y.shape[-1]))
#mae_ts, mse_ts = test_on((readout.weight, readout.bias), X[T_valid:].view(-1, X.shape[-1]), y[T_valid:].view(-1, y.shape[-1]))
#train_mae.append(mae_tr)
#val_mae.append(mae_vl)
#test_mae.append(mae_ts)
#train_mse.append(mse_tr)
#val_mse.append(mse_vl)
#test_mse.append(mse_ts)