import torch
from typing import Callable
from torch_geometric.nn import TransformerConv
from .layers import NormalLinear
from typing import Callable


class GraphAttentionEmbedding(torch.nn.Module):
    ''' GNN layer of the original TGN model '''

    def __init__(self, in_channels: int, out_channels: int, msg_dim: int, time_enc: Callable,
                 mean_delta_t: float = 0., std_delta_t: float = 1.):
        super().__init__()
        self.mean_delta_t = mean_delta_t
        self.std_delta_t = std_delta_t
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = t - last_update[edge_index[0]] 
        rel_t = (rel_t - self.mean_delta_t) / self.std_delta_t # delta_t normalization
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))

        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class JodieEmbedding(torch.nn.Module):
    def __init__(self, out_channels: int,
                 mean_delta_t: float = 0., std_delta_t: float = 1.):
        super().__init__()
        
        self.mean_delta_t = mean_delta_t
        self.std_delta_t = std_delta_t
        self.projector = NormalLinear(1, out_channels)

    def forward(self, x, last_update, t):
        rel_t = t - last_update
        if rel_t.shape[0] > 0:
            rel_t = (rel_t - self.mean_delta_t) / self.std_delta_t # delta_t normalization
            return x * (1 + self.projector(rel_t.view(-1, 1).to(x.dtype))) 
    
        return x