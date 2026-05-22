import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
import dgl.function as Fn
import torch.nn.functional as F

from dgl.ops import edge_softmax
from dgl.nn.pytorch import TypedLinear
import dgl


class SimpleHGN(nn.Module):
    def __init__(self, input_dim, output_dim, company_num, person_num, n_layer=2, n_heads=1):
        super(SimpleHGN, self).__init__()
        self.company_num = company_num
        self.person_num = person_num
        self.att_dim = 32
        self.proj = nn.Linear(input_dim + self.att_dim, input_dim, bias=False)

        self.n_layer = n_layer
        self.company_emb = nn.Embedding(company_num, input_dim)
        self.person_emb = nn.Embedding(person_num, input_dim)
        self.simplehgn_layers = nn.ModuleList()
        for i in range(n_layer):
            if i == 0:
                self.simplehgn_layers.append(
                    SimpleHGNConv(input_dim, input_dim, output_dim // n_heads, n_heads, num_etypes=20))
            else:
                self.simplehgn_layers.append(
                    SimpleHGNConv(input_dim, output_dim, output_dim // n_heads, n_heads, num_etypes=20))

    def forward(self, hete_graph, idx, x):
        edge_index, edge_type, edge_weight = hete_graph
        edge_type = torch.LongTensor(edge_type)
        edge_index = torch.LongTensor(edge_index).transpose(0, 1)
        g = dgl.DGLGraph((edge_index[0], edge_index[1]))
        g.is_block = True
        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        person_emb = self.person_emb(torch.LongTensor([i for i in range(self.person_num)]))
        company_attr_cal = torch.zeros((self.company_num, self.att_dim))
        company_attr_cal[idx] = torch.FloatTensor(x)
        company_emb = self.proj(torch.cat((company_emb, company_attr_cal), dim=1))
        emb = torch.cat((company_emb, person_emb), dim=0)
        for i in range(self.n_layer):
            emb = self.simplehgn_layers[i](g, emb, None, edge_type)
        return emb[idx]


class SimpleHGNConv(nn.Module):
    r"""
    The SimpleHGN convolution layer.

    Parameters
    ----------
    edge_dim: int
        the edge dimension
    num_etypes: int
        the number of the edge type
    in_dim: int
        the input dimension
    out_dim: int
        the output dimension
    num_heads: int
        the number of heads
    num_etypes: int
        the number of edge type
    feat_drop: float
        the feature drop rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    activation: str
        the activation function
    beta: float
        the hyperparameter used in edge residual
    """

    def __init__(self, edge_dim, in_dim, out_dim, num_heads, num_etypes, feat_drop=0.0,
                 negative_slope=0.2, residual=True, activation=F.elu, beta=0.2):
        super(SimpleHGNConv, self).__init__()
        self.edge_dim = edge_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_etypes = num_etypes
        self.beta = beta

        self.edge_emb = nn.Parameter(torch.empty(size=(num_etypes, edge_dim)))

        self.W = nn.Parameter(torch.FloatTensor(in_dim, out_dim * num_heads))
        self.W_r = TypedLinear(edge_dim, edge_dim * num_heads, num_etypes)

        self.a_l = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_r = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_e = nn.Parameter(torch.empty(size=(1, num_heads, edge_dim)))

        nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_e.data, gain=1.414)

        self.feat_drop = nn.Dropout(feat_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.activation = activation

        if residual:
            self.residual = nn.Linear(in_dim, out_dim * num_heads)
        else:
            self.register_buffer("residual", None)

    def forward(self, g, h, ntype, etype, presorted=False):
        """
        The forward part of the SimpleHGNConv.

        Parameters
        ----------
        g : object

        self.beta = beta

            the dgl homogeneous graph
        h: tensor
            the original features of the graph
        ntype: tensor
            the node type of the graph
        etype: tensor
            the edge type of the graph
        presorted: boolean
            if the ntype and etype are preordered, default: ``False``

        Returns
        -------
        tensor
            The embeddings after aggregation.
        """
        emb = self.feat_drop(h[:g.num_dst_nodes()])
        emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.out_dim)
        emb[torch.isnan(emb)] = 0.0

        edge_emb = self.W_r(self.edge_emb[etype], etype, presorted).view(-1, self.num_heads, self.edge_dim)

        row = g.edges()[0]  # edge_index[0]
        col = g.edges()[1]  # edge_index[1]

        h_l = (self.a_l * emb).sum(dim=-1)[row]
        h_r = (self.a_r * emb).sum(dim=-1)[col]
        h_e = (self.a_e * edge_emb).sum(dim=-1)

        edge_attention = self.leakyrelu(h_l + h_r + h_e)
        edge_attention = edge_softmax(g, edge_attention)  # 用图求边的注意力

        if 'alpha' in g.edata.keys():  # 一般是没有的
            res_attn = g.edata['alpha']
            edge_attention = edge_attention * (1 - self.beta) + res_attn * self.beta
        if self.num_heads == 1:
            edge_attention = edge_attention[:, 0]
            edge_attention = edge_attention.unsqueeze(1)

        with g.local_scope():  # 为了计算值并不想改变原始图，获得 h_output 的值
            emb = emb.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = edge_attention
            g.srcdata['emb'] = emb  # 数据维度不匹配 6366/6381
            # g.srcdata['emb'] = emb[:g.num_src_nodes()]
            g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'), Fn.sum('m', 'emb'))
            h_output = g.dstdata['emb'].view(-1, self.out_dim * self.num_heads)

        g.edata['alpha'] = edge_attention
        if g.is_block:
            h = h[:g.num_dst_nodes()]
        if self.residual:
            res = self.residual(h)
            h_output += res
        if self.activation is not None:
            h_output = self.activation(h_output)

        return h_output
