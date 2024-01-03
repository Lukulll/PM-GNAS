import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torch.autograd import Variable
from torch_geometric.nn.conv import *

def get_graph_model_from_arch_str(keep_mask,link, hp, in_dim, out_dim, dname):
    model = Net(
        link = link,
        hp = hp,
        in_dim = in_dim,
        out_dim = out_dim,
        dname = dname,
        keep_mask= keep_mask
    )
    return model

gnn_list = [
    "gat",  # GAT
    "gcn",  # GCN
    "gin",  # GIN
    "cheb",  # chebnet
    "sage",  # sage
    "arma",
    "graph",  
    "fc",  # fully-connected
    "skip"  # skip connection
]

def gnn_map(gnn_name, in_dim, out_dim, norm=True, bias=True) -> Module:
    '''
    :param gnn_name:
    :param in_dim:
    :param out_dim:
    :param concat: for gat, concat multi-head output or not
    :return: GNN model
    '''
    if gnn_name == "gat":
        return GATConv(in_dim, out_dim // 4, 4, bias=bias, concat = True, add_self_loops=norm)
    elif gnn_name == "gcn":
        return GCNConv(in_dim, out_dim, add_self_loops=True, normalize=norm)
    elif gnn_name == "gin":
        return GINConv(torch.nn.Linear(in_dim, out_dim))
    elif gnn_name == "cheb":
        return ChebConv(in_dim, out_dim, K=2, bias=bias)
    elif gnn_name == "sage":
        return SAGEConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "arma":
        return ARMAConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "monet":
        return GMMConv(in_dim, out_dim, dim = out_dim, kernel_size = 1)
    elif gnn_name == "graph":
        return GraphConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "fc":
        return LinearConv(in_dim, out_dim, bias=bias)
    elif gnn_name == "skip":
        return SkipConv(in_dim, out_dim, bias=bias)
    else:
        raise ValueError("No such GNN name") 

class LinearConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True):
        super(LinearConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.linear = torch.nn.Linear(in_channels, out_channels, bias)

    def forward(self, x, edge_index, edge_weight=None):
        tmp = self.linear(x)
        return tmp

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class SkipConv(Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=True):
        super(SkipConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, edge_weight=None):
        return x

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class SearchCell(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        affine,
        track_running_stats,
        use_bn=True,
        num_nodes=4,
        keep_mask=None,
    ):
        super(SearchCell, self).__init__()
        self.num_nodes = num_nodes
        self.options = nn.ModuleList()
        for curr_node in range(self.num_nodes - 1):
            for prev_node in range(curr_node + 1):
                for _op_name in OPS.keys():
                    op = OPS[_op_name](
                        in_channels,
                        out_channels,
                        stride,
                        affine,
                        track_running_stats,
                        use_bn,
                    )
                    self.options.append(op)

        if keep_mask is not None:
            self.keep_mask = keep_mask
        else:
            self.keep_mask = [True] * len(self.options)

    def forward(self, x):
        outs = [x]

        idx = 0
        for curr_node in range(self.num_nodes - 1):
            edges_in = []
            for prev_node in range(curr_node + 1):  # n-1 prev nodes
                for op_idx in range(len(OPS.keys())):
                    if self.keep_mask[idx]:
                        edges_in.append(self.options[idx](outs[prev_node]))
                    idx += 1
            node_output = sum(edges_in)
            outs.append(node_output)

        return outs[-1]


class Cell(nn.Module):
    def __init__(self, link, hp, in_dim, out_dim, dname, keep_mask=None):
        super(Cell, self).__init__()
        self.keep_mask = keep_mask
        self.link = link
        self.hp = hp
        self.dname = dname
        self.ops = nn.ModuleList()
        self.out = [True for i in range(len(link) + 1)]
        # dim is in_dim in res
        self.res_in_dim = [False for i in range(len(link) + 1)]
        #self.queue_ori_in_dim(ops)
        for i in link:
            self.out[i] = False

        if self.hp.num_pro:
            self.fc = nn.Linear(sum(self.out) * hp.dim, hp.dim)
        else:
            self.fc = nn.Linear(sum(self.out) * hp.dim, out_dim)
        self.bns = nn.ModuleList()

        for keep_mask, lk in zip(self.keep_mask, link):
            if self.res_in_dim[lk]:
                for op_name in gnn_list:
                  self.ops.append(gnn_map(op_name, in_dim, hp.dim, dname != 'ogbn-proteins'))
            else:
                for op_name in gnn_list:
                  self.ops.append(gnn_map(op_name, hp.dim, hp.dim, dname != 'ogbn-proteins'))
            self.bns.append(nn.BatchNorm1d(hp.dim))

    def queue_ori_in_dim(self, ops):
        if self.hp.num_pre:
            return
        self.res_in_dim[0] = True
        for op, lk, i in zip(ops, self.link, range(len(ops))):
            if self.res_in_dim[lk] and op == 'skip':
                self.res_in_dim[i + 1] = True

    def forward(self, x, data):
        res = [x]
        node_idx = 0
        for link, bn in zip(self.link, self.bns):
            inp = res[link]
            if self.dname != 'ogbn-proteins' or isinstance(op, GCNConv):
                adjs = data.edge_index
            else:
                adjs = data.adj_t
            if not self.res_in_dim[link]:
                inp = bn(inp)
                inp = F.relu(inp)
                inp = F.dropout(inp, p=self.hp.dropout, training=self.training)
            edges_in = []
            for op_idx in range(len(gnn_list)): # call the supernet gcn module
                    if self.keep_mask[len(gnn_list)* node_idx + op_idx]:
                        edges_in.append(self.ops[len(gnn_list)* node_idx + op_idx](inp, adjs))
            edges_in = torch.stack(edges_in).to('cuda:0' if torch.cuda.is_available() else 'cpu')
            res.append(sum(edges_in,0)) 
            node_idx += 1
        res = sum([[res[i]] if out else [] for i, out in enumerate(self.out)], [])
        fin = torch.cat(res, 1)
        fin = F.relu(fin)
        fin = self.fc(fin)
        return fin

class Net(nn.Module):
    def __init__(self, link, hp, in_dim, out_dim, dname, keep_mask):
        super().__init__()
        self.hp = hp
        self.dname = dname
        self.prep = nn.ModuleList()
        for i in range(hp.num_pre):
            idim = in_dim if i == 0 else hp.dim             
            self.prep.append(LinearConv(idim, hp.dim))

        self.prop = nn.ModuleList()
        for i in range(hp.num_pro):
            odim = out_dim if i == hp.num_pro - 1 else hp.dim             
            self.prop.append(LinearConv(hp.dim, odim))

        self.cells = nn.ModuleList()
        for i in range(hp.num_cells):
            cell = Cell(link, hp, in_dim, out_dim, dname, keep_mask)
            self.cells.append(cell)
    def forward(self, data):
        x = data.x
        adjs=  data.edge_index
        x = F.dropout(x, p=self.hp.dropout, training=self.training).double()
        #print(x)
        for i, prep in enumerate(self.prep):
            x = prep(x, adjs)
            if i == self.hp.num_pre - 2:
                x = F.elu(x)

        for cell in self.cells:
            
            x = cell(x, data)


        for i, prop in enumerate(self.prop):
            x = prop(x, adjs)
            if i == self.hp.num_pro - 2:
                x = F.elu(x)
        return x
