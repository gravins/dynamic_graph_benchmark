import matplotlib
matplotlib.use('Agg')

from utils import (set_seed, SCORE_NAMES, dst_strategies, dst_strategies_help, 
                   REGRESSION_SCORES, CLASSIFICATION_SCORES, compute_stats)
from train_link import link_prediction, link_prediction_single
from negative_sampler import neg_sampler_names
from datasets import get_dataset, DATA_NAMES
from conf import MODEL_CONFS, dyrep, jodie, edgebank
import pandas as pd
import subprocess
import warnings
import argparse
import datetime
import pickle
import time
import tqdm
import ray
import os
import gc
import pdb

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def compute_row(test_score, val_score, train_score, best_epoch, res_conf):
    row = {}
    for label, score_dict in [('test', test_score), ('val', val_score), ('train', train_score)]:
        for strategy in score_dict.keys(): 
            for k, v in score_dict[strategy].items():
                row[f'{label}_{strategy}_{k}'] = v

    for k in res_conf.keys():
        if isinstance(res_conf[k], dict):
            for sk in res_conf[k]:
                row[f'{k}_{sk}'] = res_conf[k][sk]
        else:
            row[k] = res_conf[k]
    row.update({f'best_epoch': best_epoch})
    return row


def save_log(history, log_path, conf):
    res = {
        'train': defaultdict(list), 
        'val': defaultdict(list)
    }
    for epoch in range(len(history)):
        for mode in ['train', 'val']:
            for k, v in history[epoch][mode].items():
                if 'confusion' in k: 
                    continue
                res[mode][k].append(
                    v.total_seconds() if 'time' in k else v if v != np.nan else 1e13
                )

    for k in res['train'].keys():
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(res['train'][k], label='tr')
        ax.plot(res['val'][k], label='vl')
        ax.set(
            xlabel='Epochs', 
            ylabel=k
        )
        title=str(conf['model_params']) + ' ' + str(conf['optim_params']) + ' ' + str(conf['sampler'])
        ax.set_title(title, loc='center', wrap=True)
        plt.legend(loc='best')
        plt.tight_layout()
        fig.savefig(os.path.join(log_path, f"{conf['conf_id']}_{k}.png"))
        plt.close()



if __name__ == "__main__":
    t0 = time.time()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_dir', help='The path to the directory where data files are stored.', default='./DATA')
    parser.add_argument('--data_name', help='The data name.', default=DATA_NAMES[0], choices=DATA_NAMES)
    parser.add_argument('--save_dir', help='The path to the directory where checkpoints/results are stored.', default='./RESULTS/')
    parser.add_argument('--model', help='The model name.', default=list(MODEL_CONFS)[0], choices=MODEL_CONFS.keys())
    parser.add_argument('--neg_sampler', help='The negative_sampler name.', default=neg_sampler_names[0], choices=neg_sampler_names)
    parser.add_argument('--strategy', help=f'The strategy to sample train, val, and test sets of dst nodes used by the negative_sampler.{dst_strategies_help}', default=dst_strategies[0], choices=dst_strategies)
    parser.add_argument('--use_all_strategies_eval', help='Use all strategies during the final evaluation.', action="store_true")
    parser.add_argument('--no_check_link_existence', help=f'The negative sampler does not check if the sampled negative link exists in the graph during sampling.', action='store_true')
    parser.add_argument('--no_normalize_delta_t', help=f'Do not normalize the time difference between current t and last update.', action='store_true')
    parser.add_argument('--link_regression', help='Instead of link prediction run a link regression task.', action='store_true')
    parser.add_argument('--reset_memory_eval', help='Reset memory before every evaluation (val/test).', action='store_true')
    parser.add_argument('--num_runs', help='The number of random initialization per conf.', default=5, type=int)
    parser.add_argument('--split', help='(val_ratio, test_ratio) split ratios.', nargs=2, default=[.15, .15])
    parser.add_argument('--epochs', help='The number of epochs.', default=5, type=int)
    parser.add_argument('--batch', help='The batch_size.', default=256, type=int)
    parser.add_argument('--patience', help='The early stopping patience, ie train is stopped if no score improvement after X epochs.', default=50, type=int)
    parser.add_argument('--exp_seed', help='The experimental seed.', default=9, type=int)
    parser.add_argument('--metric', help='The optimized metric.', default=list(SCORE_NAMES)[0], choices=list(SCORE_NAMES))

    parser.add_argument('--debug', help='Debug mode.', action='store_true')
    parser.add_argument('--wandb', help='Compute Weights and Biases log.', action='store_true')
    parser.add_argument('--cluster', help='Experiments run on a cluster.', action='store_true')
    parser.add_argument('--slurm', help='Instead of Kubernetes the the experiment is run on a SLURM cluster.', action='store_true')
    parser.add_argument('--parallelism', help='The degree of parallelism, ie, maximum number of parallel jobs.', default=None, type=int)
    parser.add_argument('--overwrite_ckpt', help='Overwrite checkpoint.', action='store_true')
    parser.add_argument('--verbose', help='Every <patience> epochs it prints the average time to compute an epoch.', action='store_true')
    parser.add_argument('--log', help='Plot model history without wandb.', action='store_true')

    parser.add_argument('--num_cpus', help='The number of total available cpus.', default=2, type=int)
    parser.add_argument('--num_gpus', help='The number of total available gpus.', default=0, type=int)
    parser.add_argument('--num_cpus_per_task', help='The number of cpus available for each model config.', default=-1, type=int)
    parser.add_argument('--num_gpus_per_task', help='The number of gpus available for each model config.', default=-1., type=float)

    args = parser.parse_args()
        
    assert not (args.link_regression and args.use_all_dst_strategies_eval), 'Link regression does not require neg sampling strategies'
    assert args.link_regression == (args.metric in REGRESSION_SCORES), 'Link regression requires regression metrics'
    assert args.link_regression != (args.metric in CLASSIFICATION_SCORES), 'Link prediction requires classification metrics'

    cpus_per_task = int(os.environ.get('NUM_CPUS_PER_TASK', -1))
    gpus_per_task = float(os.environ.get('NUM_GPUS_PER_TASK', -1.))
    cpus_per_task = args.num_cpus_per_task if cpus_per_task < 0 else cpus_per_task
    gpus_per_task = args.num_gpus_per_task if gpus_per_task < 0 else gpus_per_task
    assert cpus_per_task > -1, 'You must define the number of CPUS per task, by setting --num_cpus_per_task or exporting the variable NUM_CPUS_PER_TASK'
    assert gpus_per_task > -1, 'You must define the number of GPUS per task, by setting --num_gpus_per_task or exporting the variable NUM_GPUS_PER_TASK'

    if args.model == edgebank:
        if args.num_runs > 1 or args.epochs > 1: print('EdgeBank does not have trainable parameters, we do not require more than 1 epoch or 1 trial')
        args.epochs = 1
        args.num_runs = 1

    set_seed(args.exp_seed)

    if not args.debug:
        if args.cluster:
            # Init ray cluster
            if args.slurm:
                # SLURM cluster init
                ray.init(address=os.environ.get("ip_head"), 
                         _redis_password=os.environ.get("redis_password"))
            else:
                # Kubernetes cluster init
                runtime_env = {
                    "working_dir": os.getcwd(), # working_dir is the directory that contains main.py
                }

                # Get head name
                cmd = ("microk8s kubectl get pods --selector=ray.io/node-type=head -o "
                    "custom-columns=POD:metadata.name --no-headers").split(" ")
                head_name = subprocess.check_output(cmd).decode("utf-8").strip()
                print(head_name)

                # Get head ip
                cmd = ("microk8s kubectl get pod " + head_name + " --template '{{.status.podIP}}'").split(" ")
                head_ip = subprocess.check_output(cmd).decode("utf-8").strip().replace("'", "")
                print(head_ip)
                print(f"ray://{head_ip}:10001")
                ray.init(f"ray://{head_ip}:10001", runtime_env=runtime_env)

            print(f"Resources: cluster")
        else:
            cpus = os.environ.get('NUM_CPUS', None)
            gpus = os.environ.get('NUM_GPUS', None)
            ray.init(num_cpus=int(cpus) if cpus is not None else args.num_cpus, 
                     num_gpus=int(gpus) if gpus is not None else args.num_gpus)
            print(f"Resources: CPUS: {os.environ.get('NUM_CPUS', 2)}, GPUS={os.environ.get('NUM_GPUS', 0)}")

    args.save_dir = os.path.abspath(args.save_dir)
    args.data_dir = os.path.join(os.path.abspath(args.data_dir))
    if not os.path.isdir(args.data_dir): os.makedirs(args.data_dir)
    
    result_path = os.path.join(args.save_dir, args.data_name, args.model)
    if not os.path.isdir(result_path): os.makedirs(result_path)


    ckpt_path = os.path.join(result_path, 'ckpt')
    if not os.path.isdir(ckpt_path): os.makedirs(ckpt_path)

    print(f'\n{args}\n')
    print(f'Data dir: {args.data_dir}')
    print(f'Results dir: {result_path}')
    print(f'Checkpoints dir: {ckpt_path}')
    if args.log:
        log_path = os.path.join(result_path, 'log')
        if not os.path.isdir(log_path): os.makedirs(log_path)
        print(f'Logs dir: {log_path}\n')
    else:
        print('\n')

    if args.model in [dyrep, jodie] and args.no_normalize_delta_t:
        warnings.warn(f'{dyrep} and {jodie} should be runned with delta_t normalization. '
                      'High delta_t values can polarize the output. '
                      'Please consider to remove the --no_normalize_delta_t flag\n')

    partial_res_pkl = os.path.join(result_path, 'partial_results.pkl')
    partial_res_csv = os.path.join(result_path, 'partial_results.csv')
    final_res_csv = os.path.join(result_path, 'model_selection_results.csv')

    data = get_dataset(root=args.data_dir, name=args.data_name, seed=args.exp_seed)
    num_nodes, edge_dim = data.num_nodes, data.msg.shape[-1] 
    node_dim = data.x.shape[-1] if hasattr(data, 'x') else 0
    init_time = data.t[0]

    if args.no_normalize_delta_t:
        mean_delta_t, std_delta_t = 0., 1.
    else:
        stat_path = os.path.join(args.data_dir, args.data_name.lower(), 'delta_t_stats.pkl')
        if os.path.exists(stat_path):
            mean_delta_t, std_delta_t = pickle.load(open(stat_path, 'rb')) 
        else:
            mean_delta_t, std_delta_t = compute_stats(data, args.split, init_time)
            pickle.dump((mean_delta_t, std_delta_t), open(stat_path, 'wb'))
            gc.collect()

    model_instance, get_conf = MODEL_CONFS[args.model]

    num_conf = len(list(get_conf(num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t)))
    pbar = tqdm.tqdm(total= num_conf*args.num_runs)
    df = []
    ray_ids = []
    for conf_id, conf in enumerate(get_conf(num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t)):
        for i in range(args.num_runs):
            conf.update({
                'conf_id': conf_id,
                'seed': i,
                'result_path': result_path,
                'ckpt_path': ckpt_path,
            })
            conf.update(vars(args))

            if args.debug:
                    test_score, val_score, train_score, best_epoch, res_conf, history = link_prediction_single(model_instance, conf)
                    df.append(compute_row(test_score, val_score, train_score, best_epoch, res_conf))
                    if args.log: save_log(history, log_path, res_conf)
                    pickle.dump(df, open(partial_res_pkl, 'wb'))
                    pbar.update(1)
            else:
                ray_ids.append(link_prediction.options(num_cpus=cpus_per_task, num_gpus=gpus_per_task).remote(model_instance, conf))
            
            if args.parallelism is not None:
                while len(ray_ids) > args.parallelism:
                    done_id, ray_ids = ray.wait(ray_ids)
                    test_score, val_score, train_score, best_epoch, res_conf, history = ray.get(done_id[0])
                    df.append(compute_row(test_score, val_score, train_score, best_epoch, res_conf))
                    if args.log: save_log(history, log_path, res_conf)
                    pickle.dump(df, open(partial_res_pkl, 'wb'))
                    pd.DataFrame(df).to_csv(partial_res_csv)
                    pbar.update(1)
                    gc.collect()
            gc.collect()
    
    while len(ray_ids):
        done_id, ray_ids = ray.wait(ray_ids)
        test_score, val_score, train_score, best_epoch, res_conf, history = ray.get(done_id[0])
        df.append(compute_row(test_score, val_score, train_score, best_epoch, res_conf))
        if args.log: save_log(history, log_path, res_conf)
        pickle.dump(df, open(partial_res_pkl, 'wb'))
        pd.DataFrame(df).to_csv(partial_res_csv)
        pbar.update(1)
        gc.collect()

    df = pd.DataFrame(df)

    # Aggregate results over multiple runs
    # and sort them by best val score
    aggregated_df = []
    for conf_id, gdf in df.groupby('conf_id'):
        if args.num_runs == 1:
            row = gdf.iloc[0]
        else:
            row = {}
            for k in gdf.columns:
                if k == 'seed': 
                    row[k] = gdf[k].values 
                if 'test' in k or 'val' in k or 'train' in k or k == 'best_epoch':
                    row[f'{k}_mean'] = gdf[k].values.mean() if 'confusion_matrix' in k else gdf[k].mean()
                    row[f'{k}_std'] = gdf[k].values.std() if 'confusion_matrix' in k else gdf[k].std()
                else:
                    row[k] = gdf.iloc[0][k]
        aggregated_df.append(row)
    aggregated_df = pd.DataFrame(aggregated_df)
    aggregated_df = aggregated_df.sort_values(f'val_{args.strategy}_{args.metric}_mean' if args.num_runs > 1 else f'val_{args.strategy}_{args.metric}', 
                                              ascending=args.link_regression)
    aggregated_df.to_csv(final_res_csv)
    print(aggregated_df.iloc[0].to_string())

    t1 = time.time()
    print(f'Main ended in {datetime.timedelta(seconds=t1 - t0)}')
