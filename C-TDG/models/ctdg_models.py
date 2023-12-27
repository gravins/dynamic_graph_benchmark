import torch

from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, TimeEncoder
from torch_geometric.nn.resolver import activation_resolver
from typing import Callable, Optional, Any, Dict, Union, List
from .predictors import *
from .memory_layers import *
from .gnn_layers import *


class GenericModel(torch.nn.Module):
    
    def __init__(self, num_nodes, memory=None, gnn=None, gnn_act=None, link_pred=None):
        super(GenericModel, self).__init__()
        self.memory = memory
        self.gnn = gnn
        self.gnn_act = gnn_act
        self.link_pred = link_pred
        self.num_gnn_layers = 1
        self.num_nodes = num_nodes

    def reset_memory(self):
        if self.memory is not None: self.memory.reset_state()

    def update(self, src, pos_dst, t, msg, *args, **kwargs):
        if self.memory is not None: self.memory.update_state(src, pos_dst, t, msg)

    def detach_memory(self):
        if self.memory is not None: self.memory.detach()
    
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        super().reset_parameters()
        if hasattr(self.memory, 'reset_parameters'):
            self.memory.reset_parameters()
        if hasattr(self.gnn, 'reset_parameters'):
                    self.gnn.reset_parameters()
        if hasattr(self.link_pred, 'reset_parameters'):
                    self.link_pred.reset_parameters()

    def forward(self, batch, n_id, msg, t, edge_index, id_mapper):
        src, pos_dst = batch.src, batch.dst
        
        neg_dst = batch.neg_dst if hasattr(batch, 'neg_dst') else None

        # Get updated memory of all nodes involved in the computation.
        m, last_update = self.memory(n_id)

        if hasattr(batch, 'x'):
            z = torch.cat((m, batch.x[n_id]), dim=-1)

        if self.gnn is not None:
            for gnn_layer in self.gnn:
                z = gnn_layer(z, last_update, edge_index, t, msg)
                z = self.gnn_act(z)

        pos_out = self.link_pred(z[id_mapper[src]], z[id_mapper[pos_dst]])
        neg_out = self.link_pred(z[id_mapper[src]], z[id_mapper[neg_dst]]) if neg_dst is not None else None

        return pos_out, neg_out, m[id_mapper[src]], m[id_mapper[pos_dst]]
    

class TGN(GenericModel):
    def __init__(self, 
                 # Memory params
                 num_nodes: int, 
                 edge_dim: int, 
                 memory_dim: int, 
                 time_dim: int, 
                 node_dim: int = 0, 
                 # GNN params
                 gnn_hidden_dim: List[int] = [],
                 gnn_act: Union[str, Callable, None] = 'tanh',
                 gnn_act_kwargs: Optional[Dict[str, Any]] = None,
                 # Link predictor
                 readout_hidden: Optional[int] = None,
                 # Mean and std values for normalization
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 init_time: int = 0
        ):
        # Define memory
        memory = GeneralMemory(
            num_nodes,
            edge_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(edge_dim, memory_dim, time_dim), # TODO we can change this with a MLP
            aggregator_module=LastAggregator(),
            rnn='GRUCell',
            init_time=init_time
        )

        # Define GNN
        gnn = torch.nn.Sequential()
        gnn_act = activation_resolver(gnn_act, **(gnn_act_kwargs or {}))
        h_prev = memory_dim + node_dim
        for h in gnn_hidden_dim:
            gnn.append(GraphAttentionEmbedding(h_prev, h, edge_dim, time_enc=memory.time_enc, 
                                               mean_delta_t=mean_delta_t, std_delta_t=std_delta_t))
            h_prev = h * 2 # We double the input dimension because GraphAttentionEmbedding has 2 concatenated heads

        # Define the link predictor
        # NOTE: We double the input dimension because GraphAttentionEmbedding has 2 concatenated heads
        link_pred = LinkPredictor(gnn_hidden_dim[-1] * 2, readout_hidden)

        super().__init__(num_nodes, memory, gnn, gnn_act, link_pred)
        self.num_gnn_layers = len(gnn_hidden_dim)


class DyRep(GenericModel):
    def __init__(self, 
                 # Memory params
                 num_nodes: int,
                 edge_dim: int, 
                 memory_dim: int, 
                 node_dim: int = 0, 
                 non_linearity: str = 'tanh',
                 # Link predictor
                 readout_hidden: Optional[int] = None,
                 # Mean and std values for normalization
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 init_time: int = 0
        ):
        # Define memory
        memory = DyRepMemory(
            num_nodes,
            edge_dim,
            memory_dim,
            message_module=DyRepMessage(edge_dim, memory_dim, 1),
            aggregator_module=LastAggregator(),
            non_linearity=non_linearity,
            mean_delta_t=mean_delta_t, 
            std_delta_t=std_delta_t,
            init_time=init_time
        )
       
        # Define the link predictor
        link_pred = LinkPredictor(memory_dim + node_dim, readout_hidden)

        super().__init__(num_nodes, memory, link_pred=link_pred)
        self.num_gnn_layers = 1


class JODIE(GenericModel):
    def __init__(self, 
                 # Memory params
                 num_nodes: int,
                 edge_dim: int, 
                 memory_dim: int,
                 time_dim: int,
                 node_dim: int = 0, 
                 non_linearity: str = 'tanh',
                 # Link predictor
                 readout_hidden: Optional[int] = None,
                 # Mean and std values for normalization
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 init_time: int = 0
        ):
        # Define memory
        memory = GeneralMemory(
            num_nodes,
            edge_dim,
            memory_dim,
            time_dim,
            message_module=IdentityMessage(edge_dim, memory_dim, time_dim),
            aggregator_module=LastAggregator(),
            rnn='RNNCell',
            non_linearity=non_linearity,
            init_time = init_time
        )

        # Define the link predictor
        link_pred = LinkPredictor(memory_dim + node_dim, readout_hidden)

        super().__init__(num_nodes, memory, link_pred=link_pred)
        self.num_gnn_layers = 1
        self.projector_src = JodieEmbedding(memory_dim + node_dim, mean_delta_t=mean_delta_t, std_delta_t=std_delta_t)
        self.projector_dst = JodieEmbedding(memory_dim + node_dim, mean_delta_t=mean_delta_t, std_delta_t=std_delta_t)

    def reset_parameters(self):
        super().reset_parameters()
        if hasattr(self.projector_src, 'reset_parameters'):
            self.projector_src.reset_parameters()
        if hasattr(self.projector_dst, 'reset_parameters'):
                    self.projector_dst.reset_parameters()
       

    def forward(self, batch, n_id, msg, t, edge_index, id_mapper):
        src, pos_dst = batch.src, batch.dst
        
        neg_dst = batch.neg_dst if hasattr(batch, 'neg_dst') else None

        # Get updated memory of all nodes involved in the computation.
        m, last_update = self.memory(n_id)

        if hasattr(batch, 'x'):
            z = torch.cat((m, batch.x[n_id]), dim=-1)

        # Compute the projected embeddings
        z_src =  self.projector_src(z[id_mapper[src]], last_update[id_mapper[src]], batch.t)
        z_pos_dst =  self.projector_dst(z[id_mapper[pos_dst]], last_update[id_mapper[pos_dst]], batch.t)

        pos_out = self.link_pred(z_src, z_pos_dst)

        if neg_dst is not None:
            z_neg_dst =  self.projector_dst(z[id_mapper[neg_dst]], last_update[id_mapper[neg_dst]], batch.t)
            neg_out = self.link_pred(z_src, z_neg_dst)
        else:
            neg_out = None

        return pos_out, neg_out, m[id_mapper[src]], m[id_mapper[pos_dst]]


class TGAT(GenericModel):
    def __init__(self, 
                 # Memory params
                 num_nodes: int, 
                 edge_dim: int, 
                 time_dim: int, 
                 node_dim: int = 0, 
                 # GNN params
                 gnn_hidden_dim: List[int] = [],
                 gnn_act: Union[str, Callable, None] = 'relu',
                 gnn_act_kwargs: Optional[Dict[str, Any]] = None,
                 # Link predictor
                 readout_hidden: Optional[int] = None,
                 # Mean and std values for normalization
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 init_time: int = 0
        ):

        memory = LastUpdateMemory(num_nodes, init_time)

        # Define GNN
        gnn = torch.nn.Sequential()
        gnn_act = activation_resolver(gnn_act, **(gnn_act_kwargs or {}))
        h_prev = node_dim
        for h in gnn_hidden_dim:
            gnn.append(GraphAttentionEmbedding(h_prev, h, edge_dim, time_enc=TimeEncoder(time_dim), 
                                               mean_delta_t=mean_delta_t, std_delta_t=std_delta_t))
            h_prev = h * 2 # We double the input dimension because GraphAttentionEmbedding has 2 concatenated heads

        # Define the link predictor
        # NOTE: We double the input dimension because GraphAttentionEmbedding has 2 concatenated heads
        link_pred = LinkPredictor(gnn_hidden_dim[-1] * 2, readout_hidden)

        super().__init__(num_nodes, memory, gnn, gnn_act, link_pred)
        self.num_gnn_layers = len(gnn_hidden_dim)

    def forward(self, batch, n_id, msg, t, edge_index, id_mapper):
        src, pos_dst = batch.src, batch.dst
        
        neg_dst = batch.neg_dst if hasattr(batch, 'neg_dst') else None

        last_update = self.memory(n_id)
        z = batch.x[n_id]

        if self.gnn is not None:
            for gnn_layer in self.gnn:
                z = gnn_layer(z, last_update, edge_index, t, msg)
                z = self.gnn_act(z)

        pos_out = self.link_pred(z[id_mapper[src]], z[id_mapper[pos_dst]])
        neg_out = self.link_pred(z[id_mapper[src]], z[id_mapper[neg_dst]]) if neg_dst is not None else None

        return pos_out, neg_out, None, None

    
class EdgeBank(GenericModel):
    def __init__(self, num_nodes):
        super().__init__(num_nodes, memory=None, gnn=None, gnn_act=None, link_pred=None)
        self.num_gnn_layers = 0
        self.edgebank = torch.nn.parameter.Parameter(torch.zeros(num_nodes, num_nodes), requires_grad=False)
    
    def update(self, src, pos_dst, t, msg, src_emb, pos_dst_emb):
        self.edgebank[src, pos_dst] = 1

    def forward(self, batch, n_id, msg, t, edge_index, id_mapper):
        src, pos_dst = batch.src, batch.dst
        neg_dst = batch.neg_dst if hasattr(batch, 'neg_dst') else None

        pos_out = self.edgebank[src, pos_dst]
        neg_out = self.edgebank[src, neg_dst] if neg_dst is not None else None
        emb_src, emb_pos_dst = None, None

        return pos_out.unsqueeze(1), neg_out.unsqueeze(1), emb_src, emb_pos_dst
