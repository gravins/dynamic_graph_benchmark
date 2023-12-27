import torch
import math


class IdentityLayer(torch.nn.Module):
    # NOTE: this object is used to implement those models that do not have a RNN-based memory
    def __init__(self):
          super().__init__()
          self.I = torch.nn.Identity()
    
    def forward(self, x, *args, **kwargs):
         return self.I(x)


class NormalLinear(torch.nn.Linear):
    # From Jodie code
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv)

