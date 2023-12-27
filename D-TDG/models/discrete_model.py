import torch
from torch import nn
from torch_geometric_temporal.nn.recurrent import EvolveGCNO, EvolveGCNH, GCLSTM, LRGCN
from torch_geometric.utils import get_laplacian
import copy


class EvolveGCNModel(nn.Module):
    """
    EvolveGCN model
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        """
        Initializes the model.
        :param dim_node_features: input node features
        :param dim_edge_features: input edge features
        :param dim_target: target size
        :param readout_class: the class of the predictor that will classify node/graph embeddings produced by>
        :param config: the configuration dictionary to extract further hyper-parameters
        """

        super().__init__()

        self.dim_node_features = dim_node_features
        self.dim_target = dim_target
        #self.version = config['encoder_version']
        
        self.predictor = readout_class(dim_node_features = self.dim_node_features,
                                         dim_edge_features = dim_edge_features,
                                         dim_target = dim_target,
                                         config = config)

    def forward(self, snapshot, prev_state=None):
        x, edge_index, mask = snapshot.x, snapshot.edge_index, snapshot.mask
        
        node_mask = snapshot.node_mask if hasattr(snapshot, 'node_mask') else None

        h = self.model(x, edge_index)
        h = torch.relu(h)

        if 'link_pred_ids' in snapshot:
            source, target = snapshot.link_pred_ids
            h_cat = torch.cat((h[source], h[target]),
                                     dim=-1)
        else:
            h_cat = h
        out, _ = self.predictor(h_cat, None)

        if node_mask is not None:
            out = out[node_mask]

        #if mask.shape[0] > 1:
        #    out = out[mask]

        return out, h


class EvolveGCN_H_Model(EvolveGCNModel):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)
        num_nodes = config['num_nodes']
        normalize = config['normalize']
        self.model = EvolveGCNH(num_of_nodes = num_nodes,
                                in_channels = self.dim_node_features,
                                normalize = normalize)


class EvolveGCN_O_Model(EvolveGCNModel):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, readout_class, config)
        normalize = config['normalize']
        self.model = EvolveGCNO(in_channels = self.dim_node_features,
                                normalize = normalize)


class GCLSTMModel(nn.Module):
    """
    GCLSTM model
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        """
        Initializes the model.
        :param dim_node_features: input node features
        :param dim_edge_features: input edge features
        :param dim_target: target size
        :param readout_class: the class of the predictor that will classify node/graph embeddings produced by>
        :param config: the configuration dictionary to extract further hyper-parameters
        """

        super().__init__()

        self.dim_node_features = dim_node_features
        self.dim_target = dim_target

        self.dim_embedding = config['dim_embedding']
        self.k = config['K']
        self.normalization = config.get('normalization', None)

        self.model = GCLSTM(in_channels = self.dim_node_features,
                            out_channels = self.dim_embedding,
                            K = self.k,
                            normalization = self.normalization)

        self.predictor = readout_class(dim_node_features = self.dim_embedding,
                                         dim_edge_features = dim_edge_features,
                                         dim_target = dim_target,
                                         config = config)

    def forward(self, snapshot, prev_state=(None,None)):
        x, edge_index, mask = snapshot.x, snapshot.edge_index, snapshot.mask
        node_mask = snapshot.node_mask if hasattr(snapshot, 'node_mask') else None

        h, c = prev_state if prev_state is not None else (None, None)

        _, edge_weight = get_laplacian(edge_index, normalization=self.normalization)
        h, c = self.model(x, edge_index, H=h, C=c, lambda_max=edge_weight.max())
        h = torch.relu(h)

        if 'link_pred_ids' in snapshot:
            source, target = snapshot.link_pred_ids
            h_cat = torch.cat((h[source], h[target]),
                                     dim=-1)
        else:
            h_cat = h
        out, _ = self.predictor(h_cat, None)

        if node_mask is not None:
            out = out[node_mask]

        #if mask.shape[0] > 1:
        #    out = out[mask]

        return out, torch.stack((h, c), dim=0)


class LRGCNModel(nn.Module):
    """
    LRGCN model
    """
    def __init__(self, dim_node_features, dim_edge_features, dim_target, readout_class, config):
        """
        Initializes the model.
        :param dim_node_features: input node features
        :param dim_edge_features: input edge features
        :param dim_target: target size
        :param readout_class: the class of the predictor that will classify node/graph embeddings produced by>
        :param config: the configuration dictionary to extract further hyper-parameters
        """

        super().__init__()

        self.dim_node_features = dim_node_features
        self.dim_target = dim_target

        self.dim_embedding = config['dim_embedding']
        self.model = LRGCN(in_channels = self.dim_node_features,
                           out_channels = self.dim_embedding,
                           num_relations = config['num_relations'],
                           num_bases = config['num_bases'])

        self.predictor = readout_class(dim_node_features = self.dim_embedding,
                                         dim_edge_features = dim_edge_features,
                                         dim_target = dim_target,
                                         config = config)

    def forward(self, snapshot, prev_state=None):
        x, edge_index, edge_type, mask = snapshot.x, snapshot.edge_index, snapshot.relation_type, snapshot.mask
        node_mask = snapshot.node_mask if hasattr(snapshot, 'node_mask') else None

        h, c = prev_state if prev_state is not None else (None, None)

        h, c = self.model(x, edge_index, edge_type, H=h, C=c)
        h = torch.relu(h)

        if 'link_pred_ids' in snapshot:
            source, target = snapshot.link_pred_ids
            h_cat = torch.cat((h[source], h[target]),
                                     dim=-1)
        else:
            h_cat = h
        out, _ = self.predictor(h_cat, None)

        if node_mask is not None:
            out = out[node_mask]

        #if mask.shape[0] > 1:
        #    out = out[mask]

        return out, torch.stack((h, c), dim=0)