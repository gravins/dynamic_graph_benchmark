import torch
from torch_geometric.nn import TransformerConv, TGNMemory
from typing import Callable, Optional, Dict, Tuple
from torch_geometric.nn.inits import zeros, ones
from torch_geometric.utils import scatter
from .layers import IdentityLayer

TGNMessageStoreType = Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]


class GeneralMemory(TGNMemory):
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 time_dim: int, message_module: Callable,
                 aggregator_module: Callable,
                 rnn: Optional[str] = None,
                 non_linearity: str = 'tanh',
                 init_time: int = 0,
                 message_batch: int = 10000):
        
        super().__init__(num_nodes, raw_msg_dim, memory_dim, time_dim, message_module, aggregator_module)

        self.message_batch = message_batch
        if rnn is None:
             self.gru = IdentityLayer()
        else:
            rnn_instance = getattr(torch.nn, rnn)
            if 'RNN' in rnn:
                self.gru = rnn_instance(message_module.out_channels, memory_dim, nonlinearity=non_linearity)
            else:
                self.gru = rnn_instance(message_module.out_channels, memory_dim)

        self.memory[:] = torch.zeros(num_nodes, memory_dim).type_as(self.memory)
        self.last_update[:] = torch.ones(num_nodes).type_as(self.last_update) * init_time

        if hasattr(self.gru, 'reset_parameters'):
            self.gru.reset_parameters()

    def train(self, mode: bool = True):
        """Sets the module in training mode."""
        if self.training and not mode:
            # Flush message store to memory in case we just entered eval mode.
            # Do it in batches of nodes, otherwise CUDA runs out of memory for datasets with millions of nodes
            for i in range(0, self.num_nodes, self.message_batch):
                self._update_memory(
                    torch.arange(i, min(self.num_nodes, i + self.message_batch), device=self.memory.device))
            self._reset_message_store()
        super(TGNMemory, self).train(mode)
        

class DyRepMessage(torch.nn.Module):
    def __init__(self, raw_msg_dim: int, memory_dim: int, time_dim: int):
        super().__init__()
        self.out_channels = raw_msg_dim + memory_dim + time_dim

    def forward(self, z_dst, raw_msg, t_enc):
        return torch.cat([z_dst, raw_msg, t_enc], dim=-1)


class DyRepMemory(GeneralMemory):
    def __init__(self, num_nodes: int, raw_msg_dim: int, memory_dim: int,
                 message_module: Callable,
                 aggregator_module: Callable,
                 non_linearity: str = 'tanh',
                 # Mean and std values for normalization
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 init_time: int = 0
                ):

        super().__init__(num_nodes=num_nodes, raw_msg_dim=raw_msg_dim, memory_dim=memory_dim, time_dim=1, 
                         message_module=message_module, aggregator_module=aggregator_module, rnn='RNNCell', 
                         non_linearity=non_linearity, init_time=init_time)
        self.conv = TransformerConv(memory_dim, memory_dim, edge_dim=raw_msg_dim,
                                    root_weight=False, aggr='max')
        self.mean_delta_t = mean_delta_t
        self.std_delta_t = std_delta_t
        
        self.register_buffer('_mapper', torch.empty(num_nodes,
                                                   dtype=torch.long))
        
        if hasattr(self.conv, 'reset_parameters'):
            self.conv.reset_parameters()

    def _compute_msg(self, n_id: torch.Tensor, msg_store: TGNMessageStoreType,
                     msg_module: Callable):
        data = [msg_store[i] for i in n_id.tolist()]
        src, dst, t, raw_msg = list(zip(*data))
        src = torch.cat(src, dim=0)
        dst = torch.cat(dst, dim=0)
        t = torch.cat(t, dim=0)
        raw_msg = torch.cat(raw_msg, dim=0)

        node_id = torch.cat([src, dst], dim=0).unique()
        self._mapper[node_id] = torch.arange(node_id.size(0), device=n_id.device)
        edge_index = torch.stack((self._mapper[src], self._mapper[dst])).long()
        x = self.memory[node_id]

        h_struct = self.conv(x, edge_index, edge_attr=raw_msg)

        t_rel = (t - self.last_update[src]).view(-1, 1)
        t_rel = (t_rel - self.mean_delta_t) / self.std_delta_t # delta_t normalization

        msg = msg_module(h_struct[self._mapper[dst]], raw_msg, t_rel)

        return msg, t, src, dst


class SimpleMemory(torch.nn.Module):
    # Memory without RNN-based architectures
    def __init__(self, num_nodes: int, memory_dim: int, aggregator_module: Callable, init_time: int = 0) -> None:
        super().__init__()

        self.num_nodes = num_nodes
        self.memory_dim = memory_dim
        self.init_time = init_time
        self.aggr_module = aggregator_module

        self.register_buffer('memory', torch.zeros(num_nodes, memory_dim))
        last_update = torch.ones(self.num_nodes, dtype=torch.long) * init_time
        self.register_buffer('last_update', last_update)
        self.register_buffer('_assoc', torch.empty(num_nodes,
                                                   dtype=torch.long))

    def update_state(self, src, pos_dst, t, src_emb, pos_dst_emb):
        idx = torch.cat([src, pos_dst], dim=0)
        _idx = idx.unique()
        self._assoc[_idx] = torch.arange(_idx.size(0), device=_idx.device)
        
        t = torch.cat([t, t], dim=0)
        last_update = scatter(t, self._assoc[idx], 0, _idx.size(0), reduce='max')

        emb = torch.cat([src_emb, pos_dst_emb], dim=0)
        aggr = self.aggr_module(emb, self._assoc[idx], t, _idx.size(0))

        self.last_update[_idx] = last_update
        self.memory[_idx] = aggr.detach()

    def reset_state(self):
        zeros(self.memory)
        ones(self.last_update) 
        self.last_update *= self.init_time

    def detach(self):
        """Detaches the memory from gradient computation."""
        self.memory.detach_()

    def forward(self, n_id):
        return self.memory[n_id], self.last_update[n_id]
    

class LastUpdateMemory(torch.nn.Module):
    # Memory that only stores the last update information
    def __init__(self, num_nodes: int, init_time: int = 0) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        last_update = torch.ones(self.num_nodes, dtype=torch.long) * init_time
        self.register_buffer('last_update', last_update)

    def update_state(self, src, pos_dst, t, *args, **kwargs):
        idx = torch.cat([src, pos_dst], dim=0)
        t = torch.cat([t, t], dim=0)
        dim_size = self.last_update.size(0)
        last_update = scatter(t, idx, 0, dim_size, reduce='max')
        _idx = idx.unique()
        self.last_update[_idx] = last_update[_idx]

    def reset_state(self):
        zeros(self.last_update)

    def detach(self):
        return

    def forward(self, n_id):
        return self.last_update[n_id]
    

