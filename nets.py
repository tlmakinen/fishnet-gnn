import math
from typing import List, Optional, Tuple, Union


import torch
import torch.nn.functional as F
from ogb.nodeproppred import Evaluator, PygNodePropPredDataset
from torch.nn import LayerNorm, Linear, ReLU
from tqdm import tqdm

from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.nn.aggr import Aggregation
from torch_geometric.utils import softmax, scatter
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Dropout,
    InstanceNorm1d,
    LayerNorm,
    ReLU,
    Sequential,
)
from torch_geometric.nn.dense.linear import Linear

import cloudpickle as pickle

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)
        
def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)



# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fill_triangular_torch(x):
    m = x.shape[0] # should be n * (n+1) / 2
    # solve for n
    n = int(math.sqrt((0.25 + 2 * m)) - 0.5)
    idx = torch.tensor(m - (n**2 - m))
    
    x_tail = x[idx:]
        
    return torch.cat([x_tail, torch.flip(x, [0])], 0).reshape(n, n)

def fill_diagonal_torch(a, val):
    a[..., torch.arange(0, a.shape[0]), torch.arange(0, a.shape[0])] = val
    #a[..., torch.arange(0, a.shape[0]).to(device), torch.arange(0, a.shape[0]).to(device)] = val
    return a

def construct_fisher_matrix_multiple_torch(outputs):
    Q = torch.vmap(fill_triangular_torch)(outputs)
    # vmap the jnp.diag function for the batch
    _diag = torch.vmap(torch.diag)
    
    middle = _diag(torch.triu(Q) - torch.nn.Softplus()(torch.triu(Q))).to(device)
        
    padding = torch.zeros(Q.shape).to(device)
    
    # vmap the fill_diagonal code
    L = Q - torch.vmap(fill_diagonal_torch)(padding, middle)

    return torch.einsum('...ij,...jk->...ik', L, torch.permute(L, (0, 2, 1)))



## ADD IN AN MLP TO GET US TO THE RIGHT DIMENSIONALITY
class MLP(Sequential):
    def __init__(self, channels: List[int], norm: Optional[str] = None,
                 bias: bool = True, dropout: float = 0.):
        m = []
        for i in range(1, len(channels)):
            m.append(Linear(channels[i - 1], channels[i], bias=bias))

            if i < len(channels) - 1:
                if norm and norm == 'batch':
                    m.append(BatchNorm1d(channels[i], affine=True))
                elif norm and norm == 'layer':
                    m.append(LayerNorm(channels[i], elementwise_affine=True))
                elif norm and norm == 'instance':
                    m.append(InstanceNorm1d(channels[i], affine=False))
                elif norm:
                    raise NotImplementedError(
                        f'Normalization layer "{norm}" not supported.')
                m.append(ReLU())
                m.append(Dropout(dropout))

        super().__init__(*m)



class FishnetsAggregation(Aggregation):
    r"""Fishnets aggregation for GNNs

    .. math::
        \mathrm{var}(\mathcal{X}) = \mathrm{mean}(\{ \mathbf{x}_i^2 : x \in
        \mathcal{X} \}) - \mathrm{mean}(\mathcal{X})^2.

    Args:
        n_p (int): latent space size
        semi_grad (bool, optional): If set to :obj:`True`, will turn off
            gradient calculation during :math:`E[X^2]` computation. Therefore,
            only semi-gradients are used during backpropagation. Useful for
            saving memory and accelerating backward computation.
            (default: :obj:`False`)
    """
    def __init__(self, n_p: int, in_size: int = None, semi_grad: bool = False):
        super().__init__()
        
        self.n_p = n_p
        
        if in_size is None:
            in_size = n_p
        
        self.in_size = in_size
        self.semi_grad = semi_grad
        fdim = n_p + ((n_p * (n_p + 1)) // 2)
        self.fishnets_dims = fdim
        from torch_geometric.nn import Linear
        self.lin_1 = Linear(in_size, fdim, bias=True).to(device)
        self.lin_2 = Linear(n_p, in_size, bias=True).to(device)

    def forward(self, x: Tensor, index: Optional[Tensor] = None,
                ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
                dim: int = -2) -> Tensor:
        
        # GET X TO THE RIGHT DIMENSIONALITY
        x = self.lin_1(x)
        
        # CONSTRUCT SCORE AND FISHER
        # the input x will be n_p + (n_p*(n_p + 1) // 2) long
        score = x[..., :self.n_p]
        fisher = x[..., self.n_p:]
        
        # reduce the score
        score = self.reduce(score, index, ptr, dim_size, dim, reduce='sum')
        
        # construct the fisher
        fisher = construct_fisher_matrix_multiple_torch(fisher)

        # sum the fishers
        fisher = self.reduce(fisher.reshape(-1, self.n_p**2), 
                             index, ptr, dim_size, dim, reduce='sum').reshape(-1, self.n_p, self.n_p)
        
        # add in the prior 
        fisher += torch.eye(self.n_p).to(device)
        
        # calculate inverse-dot product
        mle = torch.einsum('...jk,...k->...j', torch.linalg.inv(fisher), score)
        
        # if we decide to bottleneck, send through linear back to node dimensionality
        if self.in_size != self.n_p:
            mle = self.lin_2(mle)
          
        return mle


class FishnetGCN(torch.nn.Module):
    def __init__(self, n_p, num_layers, hidden_channels=None,  
                 xdim=8, edgedim=8, ydim=112):
        super().__init__()
        
        # need some extra channels for the fisher matrix
        fishnets_channels = n_p + ((n_p * (n_p + 1)) // 2)
        
        if hidden_channels is None:
            hidden_channels = n_p

        self.node_encoder = Linear(xdim, hidden_channels)
        self.edge_encoder = Linear(edgedim, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, 
                           aggr=FishnetsAggregation(in_size=hidden_channels, n_p=n_p),
                           t=1.0, learn_t=False, 
                           num_layers=2, norm='layer')
            # output of conv is n_p size
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, ydim)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)
        
        #print("x", x.shape)

        x = self.layers[0].conv(x, edge_index, edge_attr)
        
        #print("x", x.shape)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)
            #print("x", x.shape)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)
    
    

class DeeperGCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, 
                 xdim=8, edgedim=8, ydim=112):
        super().__init__()

        self.node_encoder = Linear(xdim, hidden_channels)
        self.edge_encoder = Linear(edgedim, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(hidden_channels, hidden_channels, aggr='softmax',
                           t=1.0, learn_t=True, num_layers=2, norm='layer')
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1,
                                 ckpt_grad=i % 3)
            self.layers.append(layer)

        self.lin = Linear(hidden_channels, ydim)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        x = self.layers[0].conv(x, edge_index, edge_attr)

        for layer in self.layers[1:]:
            x = layer(x, edge_index, edge_attr)

        x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        return self.lin(x)
