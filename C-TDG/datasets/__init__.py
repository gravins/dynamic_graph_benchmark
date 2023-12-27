import torch

from torch_geometric.datasets import JODIEDataset
from numpy.random import default_rng
import numpy



OTHER_DATASET_NAMES = [] # if you want to include new datasets please insert the names in this list

JODIE = ['Wikipedia', "Reddit", "LastFM"]
DATA_NAMES = JODIE + OTHER_DATASET_NAMES
def get_dataset(root, name, seed):
    rng = default_rng(seed)
    if name in JODIE:
        dataset = JODIEDataset(root, name)
        data = dataset[0]
        data.x = torch.tensor(rng.random((data.num_nodes,1), dtype=numpy.float32))
    else:
        # if you want to include new datasets please extend this function to return a TemporalData object
        # (see https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.TemporalData.html#torch_geometric.data.TemporalData)
        raise NotImplementedError
    
    return data