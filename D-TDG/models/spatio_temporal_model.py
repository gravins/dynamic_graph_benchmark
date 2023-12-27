import torch
from torch import nn
from torch_geometric_temporal.nn.recurrent import DCRNN, TGCN, GConvGRU, GConvLSTM, A3TGCN
from torch_geometric.utils import get_laplacian
import copy

class SpatioTemporal(nn.Module):
    """
    Spatio-Temporal Deep Graph Network
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        """
        Initializes the model.
        :param dim_node_features: input node features
        :param dim_edge_features: input edge features
        :param dim_target: target size
        :param readout_class: the class of the predictor that will classify node/graph embeddings produced by this DGN
        :param config: the configuration dictionary to extract further hyper-parameters
        """

        super().__init__()

        self.dim_node_features = dim_node_features
        self.dim_target = dim_target
        self.dim_embedding = config['dim_embedding']

        self.model = None

        # self.predictor is a LinearNodePredictor
        self.predictor = readout_class(dim_node_features = self.dim_embedding,
                                         dim_edge_features = dim_edge_features,
                                         dim_target = dim_target,
                                         config = config)

    def forward(self, snapshot, prev_state=None):
        assert self.model, 'The graph encoder is not initialized'

        # snapshot.x: Tensor of size (num_nodes_t x node_ft_size)
        # snapshot.edge_index: Adj of size (num_nodes_t x num_nodes_t)
        x, edge_index, mask = snapshot.x, snapshot.edge_index, snapshot.mask

        h = self.model(x, edge_index, H=prev_state)
        h = torch.relu(h)

        out, _ = self.predictor(h, None)

        #if mask.shape[0] > 1:
        #    out = out[mask]

        return out, h


class DCRNNModel(SpatioTemporal):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)
        self.k = config['filter_size']
        self.model = DCRNN(in_channels = dim_node_features,
                           out_channels = self.dim_embedding,
                           K = self.k)


class TGCNModel(SpatioTemporal):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)
        self.model = TGCN(in_channels = dim_node_features,
                          out_channels = self.dim_embedding)


class GCRN_LSTM_Model(SpatioTemporal):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)
        
        self.k = config['K']
        self.normalization = config.get('normalization', None)
        
        self.model = GConvLSTM(in_channels = self.dim_node_features,
                               out_channels = self.dim_embedding,
                               K = self.k,
                               normalization = self.normalization)
    
    def forward(self, snapshot, prev_state=None):
        assert self.model, 'The graph encoder is not initialized'

        # snapshot.x: Tensor of size (num_nodes_t x node_ft_size)
        # snapshot.edge_index: Adj of size (num_nodes_t x num_nodes_t)
        x, edge_index, mask = snapshot.x, snapshot.edge_index, snapshot.mask

        if prev_state is None:
            h, c = None, None
        else:
            h, c = prev_state
        
        _, edge_weight = get_laplacian(edge_index, normalization=self.normalization)
        h, c = self.model(x, edge_index, H=h, C=c, lambda_max=edge_weight.max())
        h = torch.relu(h)

        out, _ = self.predictor(h, None)

        #if mask.shape[0] > 1:
        #    out = out[mask]

        return out, torch.stack((h, c))

class GCRN_GRU_Model(SpatioTemporal):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)
        
        self.k = config['K']
        self.normalization = config.get('normalization', None)
        self.model = GConvGRU(in_channels = self.dim_node_features,
                              out_channels = self.dim_embedding,
                              K = self.k,
                              normalization = self.normalization)
    
    def forward(self, snapshot, prev_state=None):
        assert self.model, 'The graph encoder is not initialized'

        # snapshot.x: Tensor of size (num_nodes_t x node_ft_size)
        # snapshot.edge_index: Adj of size (num_nodes_t x num_nodes_t)
        x, edge_index, mask = snapshot.x, snapshot.edge_index, snapshot.mask

        _, edge_weight = get_laplacian(edge_index, normalization=self.normalization)
        h = self.model(x, edge_index, H=prev_state, lambda_max=edge_weight.max())
        h = torch.relu(h)

        out, _ = self.predictor(h, None)

        #if mask.shape[0] > 1:
        #    out = out[mask]

        return out, h


class A3TGCNModel(SpatioTemporal):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)
        self.model = A3TGCN(in_channels = self.dim_node_features,
                            out_channels = self.dim_embedding,
                            periods = 1)

    def forward(self, snapshot, prev_state=None):
        assert self.model, 'The graph encoder is not initialized'

        # snapshot.x: Tensor of size (num_nodes_t x node_ft_size)
        # snapshot.edge_index: Adj of size (num_nodes_t x num_nodes_t)
        x, edge_index, mask = snapshot.x, snapshot.edge_index, snapshot.mask

        h = self.model(x.view(x.shape[0], x.shape[1], 1), edge_index, H=prev_state) # A3TGCN input must have size [num_nodes, num_features, num_periods]
        h = torch.relu(h)

        out, _ = self.predictor(h, None)

        #if mask.shape[0] > 1:
        #    out = out[mask]

        return out, h