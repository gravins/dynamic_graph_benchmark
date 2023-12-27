import os

folder='.'
num_runs=5
epochs=1000
patience=50
cluster=True
log=True
cpus_per_task=2
gpus_per_task=0.

# Best configs:
# Wikipedia 
#     DyRep {'model': {'num_nodes': 9227, 'edge_dim': 172, 'node_dim': 1, 'memory_dim': 96, 'non_linearity': 'tanh', 'readout_hidden': 48, 'mean_delta_t': 42410.88005751506, 'std_delta_t': 180067.0410694675, 'init_time': 'tensor(0)'}, 'optim': {'lr': 0.001, 'wd': 0.0001}, 'sampler': {'size': 5}}
#     JODIE {'model': {'num_nodes': 9227, 'edge_dim': 172, 'node_dim': 1, 'memory_dim': 96, 'time_dim': 16, 'non_linearity': 'tanh', 'readout_hidden': 48, 'mean_delta_t': 42410.88005751506, 'std_delta_t': 180067.0410694675, 'init_time': 'tensor(0)'}, 'optim': {'lr': 0.0001, 'wd': 1e-05}, 'sampler': {'size': 5}}
#     TGAT {'model': {'num_nodes': 9227, 'edge_dim': 172, 'node_dim': 1, 'time_dim': 16, 'gnn_hidden_dim': '[96, 96, 96]', 'gnn_act': 'tanh', 'readout_hidden': 48, 'mean_delta_t': 42410.88005751506, 'std_delta_t': 180067.0410694675, 'init_time': 'tensor(0)'}, 'optim': {'lr': 0.001, 'wd': 0.0001}, 'sampler': {'size': 5}}
#     TGN {'model': {'num_nodes': 9227, 'edge_dim': 172, 'node_dim': 1, 'memory_dim': 96, 'time_dim': 16, 'gnn_hidden_dim': '[48, 48, 48]', 'gnn_act': 'tanh', 'readout_hidden': 48, 'mean_delta_t': 42410.88005751506, 'std_delta_t': 180067.0410694675, 'init_time': 'tensor(0)'}, 'optim': {'lr': 0.001, 'wd': 1e-05}, 'sampler': {'size': 5}}
# Reddit 
#     DyRep {'model': {'num_nodes': 10984, 'edge_dim': 172, 'node_dim': 1, 'memory_dim': 96, 'non_linearity': 'tanh', 'readout_hidden': 48, 'mean_delta_t': 20026.089633173506, 'std_delta_t': 66116.52206122893, 'init_time': 'tensor(0)'}, 'optim': {'lr': 0.0001, 'wd': 0.0001}, 'sampler': {'size': 5}}
#     JODIE {'model': {'num_nodes': 10984, 'edge_dim': 172, 'node_dim': 1, 'memory_dim': 96, 'time_dim': 16, 'non_linearity': 'tanh', 'readout_hidden': 48, 'mean_delta_t': 20026.089633173506, 'std_delta_t': 66116.52206122893, 'init_time': 'tensor(0)'}, 'optim': {'lr': 0.0001, 'wd': 1e-05}, 'sampler': {'size': 5}}
#     TGAT {'model': {'num_nodes': 10984, 'edge_dim': 172, 'node_dim': 1, 'time_dim': 16, 'gnn_hidden_dim': '[96, 96, 96]', 'gnn_act': 'tanh', 'readout_hidden': 48, 'mean_delta_t': 20026.089633173506, 'std_delta_t': 66116.52206122893, 'init_time': 'tensor(0)'}, 'optim': {'lr': 0.001, 'wd': 0.0001}, 'sampler': {'size': 5}}
#     TGN {'model': {'num_nodes': 10984, 'edge_dim': 172, 'node_dim': 1, 'memory_dim': 96, 'time_dim': 16, 'gnn_hidden_dim': '[96]', 'gnn_act': 'tanh', 'readout_hidden': 48, 'mean_delta_t': 20026.089633173506, 'std_delta_t': 66116.52206122893, 'init_time': 'tensor(0)'}, 'optim': {'lr': 0.001, 'wd': 1e-05}, 'sampler': {'size': 5}}
# LastFM 
#     DyRep {'model': {'num_nodes': 1980, 'edge_dim': 2, 'node_dim': 1, 'memory_dim': 96, 'non_linearity': 'tanh', 'readout_hidden': 48, 'mean_delta_t': 98409.48842595662, 'std_delta_t': 1354318.2658307378, 'init_time': 'tensor(0)'}, 'optim': {'lr': 0.0001, 'wd': 0.0001}, 'sampler': {'size': 5}}
#     JODIE {'model': {'num_nodes': 1980, 'edge_dim': 2, 'node_dim': 1, 'memory_dim': 32, 'time_dim': 16, 'non_linearity': 'tanh', 'readout_hidden': 16, 'mean_delta_t': 98409.48842595662, 'std_delta_t': 1354318.2658307378, 'init_time': 'tensor(0)'}, 'optim': {'lr': 0.0001, 'wd': 1e-05}, 'sampler': {'size': 5}}
#     TGAT {'model': {'num_nodes': 1980, 'edge_dim': 2, 'node_dim': 1, 'time_dim': 16, 'gnn_hidden_dim': '[48, 48, 48]', 'gnn_act': 'tanh', 'readout_hidden': 48, 'mean_delta_t': 98409.48842595662, 'std_delta_t': 1354318.2658307378, 'init_time': 'tensor(0)'}, 'optim': {'lr': 0.0001, 'wd': 0.0001}, 'sampler': {'size': 5}}
#     TGN {'model': {'num_nodes': 1980, 'edge_dim': 2, 'node_dim': 1, 'memory_dim': 32, 'time_dim': 16, 'gnn_hidden_dim': '[16, 16, 16]', 'gnn_act': 'tanh', 'readout_hidden': 16, 'mean_delta_t': 98409.48842595662, 'std_delta_t': 1354318.2658307378, 'init_time': 'tensor(0)'}, 'optim': {'lr': 0.001, 'wd': 1e-05}, 'sampler': {'size': 5}}

 
for data in ['Wikipedia', 'Reddit', "LastFM"]:
    for model in ['DyRep', 'JODIE', 'TGAT', 'TGN', 'EdgeBank']:
        if model == 'EdgeBank':
            num_runs=1
            epochs=1
            
        parallelism =  20

        cmd = (f"python3 -u main.py --data_dir {folder}/DATA --data_name {data} --save_dir {folder}/RESULTS "
               f"--model {model} --num_runs {num_runs} --epochs {epochs} --patience {patience} "
               f"--num_cpus_per_task {cpus_per_task} --num_gpus_per_task {gpus_per_task} "
               f"--parallelism {parallelism} "
               f"{'--cluster' if cluster else ''} --verbose {'--log' if log else ''} "
               f"> {folder}/out_same_dim4_{model}_{data} 2> {folder}/err_same_dim4_{model}_{data}")
        print('Running:', cmd)
        os.system(cmd)
