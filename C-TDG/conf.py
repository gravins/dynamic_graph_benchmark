from utils import cartesian_product
from models import *

shared_params = {
    'embedding_dim': [96, 64, 32], 
    'time_dim': [16], 
    'lr': [0.001, 0.0001], 
    'wd': [0.0001, 0.00001],
    'gnn_act': ['tanh'],
    'sampler_size': [5],
    'num_gnn_layers': [3, 1]
}

def get_TGN_conf(num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t):
    confs = shared_params
    confs['half dim'] = [True, False]
    
    for params in cartesian_product(confs):
        yield {
            'model_params':{
                'num_nodes': num_nodes,
                'edge_dim': edge_dim,
                'node_dim': node_dim,
                'memory_dim': params['embedding_dim'],
                'time_dim': params['time_dim'],
                'gnn_hidden_dim': ([params['embedding_dim'] // 2] * params['num_gnn_layers'] if params['half dim']
                                   else [params['embedding_dim']] * params['num_gnn_layers']),
                'gnn_act': params['gnn_act'],
                'readout_hidden': max(1, params['embedding_dim'] // 2),
                'mean_delta_t': mean_delta_t,
                'std_delta_t': std_delta_t,
                'init_time': init_time
            },
            'optim_params':{
                'lr': params['lr'], 
                'wd': params['wd']
            },
            'sampler': {'size': params['sampler_size']},
        }


def get_DyRep_conf(num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t):
    confs = {
        'embedding_dim': shared_params['embedding_dim'],
        'lr': shared_params['lr'],
        'wd': shared_params['wd'],
        'non_linearity': shared_params['gnn_act'],
        'sampler_size': shared_params['sampler_size']
    }

    for params in cartesian_product(confs):
        yield {
            'model_params':{
                'num_nodes': num_nodes,
                'edge_dim': edge_dim,
                'node_dim': node_dim,
                'memory_dim': params['embedding_dim'],
                'non_linearity': params['non_linearity'],
                'readout_hidden': max(1, params['embedding_dim'] // 2),
                'mean_delta_t': mean_delta_t,
                'std_delta_t': std_delta_t,
                'init_time': init_time
            },
            'optim_params':{
                'lr': params['lr'], 
                'wd': params['wd']
            },
            'sampler': {'size': params['sampler_size']},
        }


def get_JODIE_conf(num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t):
    confs = {
        'embedding_dim':  shared_params['embedding_dim'],
        'time_dim':shared_params['time_dim'],
        'lr': shared_params['lr'],
        'wd': shared_params['wd'],
        'non_linearity': shared_params['gnn_act'],
        'sampler_size': shared_params['sampler_size']
    }

    for params in cartesian_product(confs):
        yield {
            'model_params':{
                'num_nodes': num_nodes,
                'edge_dim': edge_dim,
                'node_dim': node_dim,
                'memory_dim': params['embedding_dim'],
                'time_dim': params['time_dim'],
                'non_linearity': params['non_linearity'],
                'readout_hidden': max(1, params['embedding_dim'] // 2),
                'mean_delta_t': mean_delta_t,
                'std_delta_t': std_delta_t,
                'init_time': init_time
            },
            'optim_params':{
                'lr': params['lr'], 
                'wd': params['wd']
            },
            'sampler': {'size': params['sampler_size']},
        }


def get_TGAT_conf(num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t):
    confs = shared_params
    confs['half dim'] = [True, False]

    for params in cartesian_product(confs):
        yield {
            'model_params':{
                'num_nodes': num_nodes,
                'edge_dim': edge_dim,
                'node_dim': node_dim,
                'time_dim': params['time_dim'],
                'gnn_hidden_dim': ([params['embedding_dim'] // 2] * params['num_gnn_layers'] if params['half dim']
                                   else [params['embedding_dim']] * params['num_gnn_layers']),
                'gnn_act': params['gnn_act'],
                'readout_hidden': max(1, params['embedding_dim'] // 2),
                'mean_delta_t': mean_delta_t,
                'std_delta_t': std_delta_t,
                'init_time': init_time
            },
            'optim_params':{
                'lr': params['lr'], 
                'wd': params['wd']
            },
            'sampler': {'size': params['sampler_size']},
        }


def get_EdgeBank_conf(num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t):
    confs = {
        'lr': [0.001],       # this is not used by the model
        'wd': [0.0001],      # this is not used by the model
        'sampler_size': [5], # this is not used by the model
    }
    for params in cartesian_product(confs):
        yield {
            'model_params':{
                'num_nodes': num_nodes,
            },
            'optim_params':{
                'lr': params['lr'], 
                'wd': params['wd']
            },
            'sampler': {'size': params['sampler_size']},
        }

_tgn_fun = lambda num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t: get_TGN_conf(num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t)
_dyrep_fun = lambda num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t: get_DyRep_conf(num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t)
_jodie_fun = lambda num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t: get_JODIE_conf(num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t)
_tgat_fun = lambda num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t: get_TGAT_conf(num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t)
_edgebank_fun = lambda num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t: get_EdgeBank_conf(num_nodes, edge_dim, node_dim, init_time, mean_delta_t, std_delta_t)

tgn, dyrep, jodie, tgat, edgebank = 'TGN', 'DyRep', 'JODIE', 'TGAT', 'EdgeBank'

MODEL_CONFS = {
    tgn: (TGN, _tgn_fun),
    dyrep: (DyRep, _dyrep_fun),
    jodie: (JODIE, _jodie_fun),
    tgat: (TGAT, _tgat_fun),
    edgebank: (EdgeBank, _edgebank_fun),
}
