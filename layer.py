import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptPairTensor, Size, OptTensor
from torch_scatter import scatter
from torch_geometric.nn import global_add_pool, global_max_pool


class NeighborPropagate(MessagePassing):
    def __init__(self, aggr: str = 'mean', **kwargs,):
        kwargs['aggr'] = aggr if aggr != 'lstm' else None
        super().__init__(**kwargs)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)

        return out

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def aggregate(self, x: Tensor, index: Tensor, ptr: Optional[Tensor] = None, dim_size: Optional[int] = None) -> Tensor:
        return scatter(x, index, dim=self.node_dim, dim_size=dim_size, reduce=self.aggr)
