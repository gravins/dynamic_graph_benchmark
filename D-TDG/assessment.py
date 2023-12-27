import json
import os.path as osp
from torch_geometric.data import Batch
import torch
import numpy

from pydgn.evaluation.config import Config
from pydgn.experiment.util import s2c
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from models.discrete_model import *
import itertools

device = 'cpu'



model_names = ['evolvegcn_h', 'evolvegcn_o', 'lrgcn', 'gclstm']
dataset_names = ['bitcoin_alpha', 'as_733', 'elliptic']

for model_name, dataset_name in itertools.product(model_names, dataset_names):

    data_splits_file = f'DATA_SPLITS/{dataset_name}/{dataset_name}_outer1_inner5.splits'

    if model_name in ['evolvegcn_h', 'evolvegcn_o', 'lrgcn'] and dataset_name in ['bitcoin_alpha', 'as_733']:
        path = f'/data/gravina/benchmark/save_1_12_2022/new_results_9_January_2023/RESULTS/balanced_link_prediction/{dataset_name}_{model_name}_{dataset_name}/MODEL_ASSESSMENT/OUTER_FOLD_1/'
    elif model_name == 'gclstm' and dataset_name in ['bitcoin_alpha', 'as_733']:
        path = f'/data/gravina/benchmark/save_1_12_2022/new_results_9_January_2023/RESULTS/imbalanced_link_prediction/{dataset_name}_{model_name}_{dataset_name}/MODEL_ASSESSMENT/OUTER_FOLD_1/'
    elif dataset_name == 'elliptic':
        if model_name in ['evolvegcn_h', 'evolvegcn_o', 'gclstm']:
            path = f'/data/gravina/benchmark/save_1_12_2022/benchmark/RESULTS/{dataset_name}_{model_name}_{dataset_name}/MODEL_ASSESSMENT/OUTER_FOLD_1/'
            data_splits_file = f'/data/gravina/benchmark/save_1_12_2022/benchmark/DATA_SPLITS/{dataset_name}/{dataset_name}_outer1_inner5.splits'
        else:
            path = f'/data/gravina/benchmark/save_1_12_2022/new_results_9_January_2023/RESULTS/{dataset_name}_{model_name}_{dataset_name}/MODEL_ASSESSMENT/OUTER_FOLD_1/'

    outer_folds = 1
    inner_folds = 1

    avg_acc = []
    avg_balanced_acc = []
    avg_auc_roc = []
    avg_cm = []
    for i in range(1,6):
        # Get best conf 
        conf_path = osp.join(path, 'outer_results.json')

        with open(conf_path, 'r') as f:
            best_config = json.load(f)['best_config']
            
        config_with_metadata = Config(best_config['config'])

        # Get data
        dataset_getter_class = s2c(config_with_metadata.dataset_getter)
        dataset_getter = dataset_getter_class(config_with_metadata.data_root,
                                            data_splits_file,
                                            s2c(config_with_metadata.dataset_class),
                                            dataset_name,
                                            s2c(config_with_metadata.data_loader),
                                            config_with_metadata.data_loader_args,
                                            outer_folds,
                                            inner_folds)

        dataset_getter.set_inner_k(0)
        dataset_getter.set_outer_k(0)

        # not really used
        dataset_getter.set_exp_seed(0)

        batch_size = 32
        shuffle = False

        # Instantiate the Dataset
        train_loader = dataset_getter.get_outer_train(batch_size=batch_size, shuffle=shuffle)
        val_loader = dataset_getter.get_outer_val(batch_size=batch_size, shuffle=shuffle)
        test_loader = dataset_getter.get_outer_test(batch_size=batch_size, shuffle=shuffle)

        # Call this after the loaders: the datasets may need to be instantiated with additional parameters
        dim_node_features = dataset_getter.get_dim_node_features()
        dim_edge_features = dataset_getter.get_dim_edge_features()
        dim_target = dataset_getter.get_dim_target()

        # Get model
        ckpt_path = osp.join(path + f'final_run{i}', 'last_checkpoint.pth')
        ckpt = torch.load(ckpt_path, map_location='cpu')

        if model_name == 'evolvegcn_h':
            model_class = EvolveGCN_H_Model
        elif model_name == 'evolvegcn_o':
            model_class = EvolveGCN_O_Model
        elif model_name == 'lrgcn':
            model_class = LRGCNModel
        elif model_name == 'gclstm':
            model_class = GCLSTMModel
        else:
            raise NotImplementedError()

        readout_class = s2c(config_with_metadata['supervised_config']['readout'])
        model = model_class(dim_node_features, dim_edge_features, dim_node_features, readout_class, config_with_metadata['supervised_config'])

        model_state = ckpt['model_state']
        model.load_state_dict(ckpt['model_state'])
        model.to(device)
        model.eval()

        y_pred = []
        y_true = []
        y_pred_confidence = []
        
        prev_h = None
        for batch in test_loader:
            # Move data to device
            for snapshot in batch:
                snapshot.to(device)

                output, embs = model.forward(snapshot, prev_h)
                prev_h = embs

                confidence = torch.sigmoid(output)
                if len(output.shape) == 1:
                    output = (confidence>0.5).float()
                elif len(output.shape) == 2:
                    output = torch.argmax(output, 1)
                else:
                    raise NotImplementedError(f"Only shape 1-d or 2-d tensors are implemented, got {output.shape}")
                y_pred_confidence.append(confidence.cpu().detach())
                y_pred.append(output.cpu().detach())
                y_true.append(snapshot.y.cpu().detach())
        y_pred = torch.cat(y_pred).numpy()
        y_true = torch.cat(y_true).numpy()
        y_pred_confidence = torch.cat(y_pred_confidence).numpy()

        avg_acc.append(accuracy_score(y_true, y_pred))
        avg_balanced_acc.append(balanced_accuracy_score(y_true, y_pred))
        avg_auc_roc.append(roc_auc_score(y_true, y_pred_confidence))
        avg_cm.append(confusion_matrix(y_true, y_pred))
        # print(
        #     f'num_true = {y_true.sum()}, num_false = {len(y_true) - y_true.sum()}',
        #     f'num_pred_true = {y_pred.sum()}, num_pred_false = {len(y_pred) - y_true.sum()}',    
        #     avg_acc[-1], avg_cm[-1]
        # )

    print(model_name, dataset_name)
    print('Acc:', numpy.mean(avg_acc), numpy.std(avg_acc))
    print('Balanced acc:',numpy.mean(avg_balanced_acc), numpy.std(avg_balanced_acc))
    print('Roc AUC:',numpy.mean(avg_auc_roc), numpy.std(avg_auc_roc))
    print('Confision Matrix:', numpy.mean(avg_cm), numpy.std(avg_cm))
    print('\n\n')