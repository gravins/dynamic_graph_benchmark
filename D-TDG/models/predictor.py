import torch
import torch.nn as nn
from pydgn.model.interface import ReadoutInterface


class LinearLinkPredictor(ReadoutInterface):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)
        self.W = nn.Linear(dim_node_features*2, dim_target, bias=True)

    def forward(self, x, batch):
        # Assume x is the concatenation of the involved node embeddings
        return self.W(x), x



class LinearNodePredictor(ReadoutInterface):
    def __init__(self, dim_node_features, dim_edge_features, dim_target, config):
        super().__init__(dim_node_features, dim_edge_features, dim_target, config)
        self.W = nn.Linear(dim_node_features, dim_target, bias=True)

    def forward(self, x, batch):
        # Assume x is node embedding
        return self.W(x), x