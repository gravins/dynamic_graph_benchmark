import torch

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, accuracy_score, confusion_matrix
from typing import Optional
import numpy as np
import itertools
import random


AUC = 'auc'
F1_SCORE = 'f1_score'
ACCURACY = 'accuracy'
AVERAGE_PRECISION = 'ap'
MSE = 'mse'
MAE = 'mae'

CLASSIFICATION_SCORES = {
 AUC: roc_auc_score,
 F1_SCORE: f1_score,
 ACCURACY: accuracy_score,
 AVERAGE_PRECISION: average_precision_score
}

REGRESSION_SCORES = {
 MAE: torch.nn.L1Loss(),
 MSE: torch.nn.MSELoss(),
}

SCORE_NAMES = list(CLASSIFICATION_SCORES) + list(REGRESSION_SCORES)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_stats(data, split, init_time):
    train_data, _, _ = data.train_val_test_split(val_ratio=split[0], test_ratio=split[1])

    last_timestamp_src = dict()
    last_timestamp_dst = dict()
    last_timestamp = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    all_timediffs = []
    for src, dst, t in zip(train_data.src, train_data.dst, train_data.t):
        src, dst, t = src.item(), dst.item(), t.item()

        all_timediffs_src.append(t - last_timestamp_src.get(src, init_time))
        all_timediffs_dst.append(t - last_timestamp_dst.get(dst, init_time))
        all_timediffs.append(t - last_timestamp.get(src, init_time))
        all_timediffs.append(t - last_timestamp.get(dst, init_time))

        last_timestamp_src[src] = t
        last_timestamp_dst[dst] = t
        last_timestamp[src] = t
        last_timestamp[dst] = t
    assert len(all_timediffs_src) == train_data.num_events
    assert len(all_timediffs_dst) == train_data.num_events
    assert len(all_timediffs) == train_data.num_events * 2

    src_and_dst = all_timediffs_src + all_timediffs_dst
    mean_delta_t = np.mean(all_timediffs)
    std_delta_t = np.std(all_timediffs)

    print(f'avg delta_t(src): {np.mean(all_timediffs_src)} +/- {np.std(all_timediffs_src)}')
    print(f'avg delta_t(dst): {np.mean(all_timediffs_dst)} +/- {np.std(all_timediffs_dst)}')
    print(f'avg delta_t(src+dst): {np.mean(src_and_dst)} +/- {np.std(src_and_dst)}')
    print(f'avg delta_t(all): {mean_delta_t} +/- {std_delta_t}')

    return mean_delta_t, std_delta_t


def optimizer_to(optim, device):
    # Code from https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068/3
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def scoring(y_true: torch.Tensor, y_pred: torch.Tensor, y_pred_confidence: torch.Tensor, 
            is_regression: bool = False, require_sigmoid: bool = True, labels: Optional[list] = None):
    s = {}
    if not is_regression:
        for k, func in CLASSIFICATION_SCORES.items():
            if k == AVERAGE_PRECISION or k == AUC:
                y_pc = y_pred_confidence.sigmoid() if require_sigmoid or k == AUC else y_pred_confidence
                f = func(y_true, y_pc)
            else:
                f = func(y_true, y_pred) #, average='weighted')
            s[k] = f
        s["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels)
    else:
        s = {k: func(y_pred, y_true).detach().cpu().item() for k, func in REGRESSION_SCORES.items()}
    return s


def cartesian_product(params):
    # Given a dictionary where for each key is associated a lists of values, the function compute cartesian product
    # of all values. 
    # Example:
    #  Input:  params = {"n_layer": [1,2], "bias": [True, False] }
    #  Output: {"n_layer": [1], "bias": [True]}
    #          {"n_layer": [1], "bias": [False]}
    #          {"n_layer": [2], "bias": [True]}
    #          {"n_layer": [2], "bias": [False]}
    keys = params.keys()
    vals = params.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))

ALL = 'all'
SPLIT = 'split'
dst_strategies = [SPLIT, ALL]
dst_strategies_help = (f'\n\t{ALL}: train, val, and test samplers always uses all the nodes in the data'
                       f'\n\t{SPLIT}: the train_sampler uses only the dst nodes in train set, val_sampler '
                       'uses train+val dst nodes, test_sampler uses all dst nodes in the data')
def get_node_sets(strategy, train_data, val_data, test_data):
    if strategy == ALL:
        src_node_set = torch.cat([train_data.src, val_data.src, test_data.src]).type(torch.long)
        dst_node_set = torch.cat([train_data.dst, val_data.dst, test_data.dst]).type(torch.long)
        train_src_nodes, train_dst_nodes = src_node_set, dst_node_set
        val_src_nodes, val_dst_nodes = src_node_set, dst_node_set
        test_src_nodes, test_dst_nodes = src_node_set, dst_node_set

    elif strategy == SPLIT:
        train_src_nodes, train_dst_nodes = train_data.src.type(torch.long), train_data.dst.type(torch.long)
        val_src_nodes, val_dst_nodes = (
            torch.cat([train_data.src, val_data.src]).type(torch.long),
            torch.cat([train_data.dst, val_data.dst]).type(torch.long)
        )
        test_src_nodes, test_dst_nodes = (
            torch.cat([train_data.src, val_data.src, test_data.src]).type(torch.long),
            torch.cat([train_data.dst, val_data.dst, test_data.dst]).type(torch.long)
        )
    else:
        raise NotImplementedError()
    
    return train_src_nodes, train_dst_nodes, val_src_nodes, val_dst_nodes, test_src_nodes, test_dst_nodes
