import torch
from numpy.random import default_rng
from collections import defaultdict
from typing import Iterable

neg_sampler_names = ['NegativeSampler']

class NegativeSampler:
    def __init__(self, src_nodes: Iterable, dst_nodes: Iterable, name: str, seed: int = 9, 
                 check_link_existence: bool = True) -> None:
        
        self.neighs = defaultdict(set)
        if check_link_existence:
            for src, dst in zip(src_nodes, dst_nodes):
                if torch.is_tensor(src): src = src.item()
                if torch.is_tensor(dst): dst = dst.item()
                self.neighs[src].add(dst)

        self.src_nodes = src_nodes.unique().to('cpu')
        self.dst_nodes = dst_nodes.unique().to('cpu')
        self.seed = seed
        self.rng = default_rng(seed)
        self.name = name
        self.check_link_existence = check_link_existence

    def sample(self, src: torch.Tensor, eval: bool = False, eval_seed: int = 9,  *args, **kwargs) -> torch.Tensor:
        rng = default_rng(eval_seed) if eval else self.rng
        neg_dst = rng.choice(self.dst_nodes, size=src.shape[0])

        if self.check_link_existence:
            for i in range(src.shape[0]):
                j = 0
                while self._exists(src[i].item(), neg_dst[i]) or j > 100:
                    neg_dst[i] = rng.choice(self.dst_nodes, size=1)
                    j += 1
                if j > 100:
                    print(f'NegativeSampler: after 100 attemps failed to find an unseen neg_dst for node {src[i]}')

        return torch.tensor(neg_dst, dtype=torch.long, device=src.device)

    def _exists(self, src, dst):
        if torch.is_tensor(src): src = src.item()
        if torch.is_tensor(dst): dst = dst.item()        
        return dst in self.neighs[src]

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.name}, seed={self.seed})'

