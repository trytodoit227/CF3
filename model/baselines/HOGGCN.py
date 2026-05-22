import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from utils.model_utils import get_adj


class GraphConvolution_homo(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_homo, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_bi = Parameter(torch.FloatTensor(in_features, out_features))
        self.w = Parameter(torch.FloatTensor(1))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        # self.adjacency_mask = Parameter(adj.clone())
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        stdv_bi = 1. / math.sqrt(self.weight_bi.size(1))
        self.weight_bi.data.uniform_(-stdv_bi, stdv_bi)
        self.w.data.uniform_(0.5, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, bi_adj, output, labels_for_lp):

        new_bi = bi_adj.clone()
        # new_bi = new_bi * self.adjacency_mask
        new_bi = F.normalize(new_bi, p=1, dim=1)
        identity = torch.eye(adj.shape[0])
        output = output.exp()
        homo_matrix = torch.matmul(output, output.t())
        homo_matrix = 0.4 * homo_matrix + 1 * new_bi
        y_hat = torch.mm(new_bi, labels_for_lp)

        bi_adj = torch.mul(bi_adj, homo_matrix)

        with torch.no_grad():
            bi_row_sum = torch.sum(bi_adj, dim=1, keepdim=True)
            bi_r_inv = torch.pow(bi_row_sum, -1).flatten()  # np.power(rowsum, -1).flatten()
            bi_r_inv[torch.isinf(bi_r_inv)] = 0.
            bi_r_mat_inv = torch.diag(bi_r_inv)
        bi_adj = torch.matmul(bi_r_mat_inv, bi_adj)

        support = torch.mm(input, self.weight)
        support_bi = torch.mm(input, self.weight_bi)
        output = torch.spmm(identity, support)
        output_bi = torch.spmm(bi_adj, support_bi)
        output = output + torch.mul(self.w, output_bi)

        if self.bias is not None:
            return output + self.bias, y_hat, homo_matrix
        else:
            return output, y_hat, homo_matrix

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN_homo(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN_homo, self).__init__()
        self.gc1 = GraphConvolution_homo(nfeat, nhid)
        self.gc2 = GraphConvolution_homo(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj, bi_adj, output, labels_for_lp):
        x, y_hat, mask = self.gc1(x, adj, bi_adj, output, labels_for_lp)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x_3, y_hat, mask = self.gc2(x, adj, bi_adj, output, labels_for_lp)
        return x_3, y_hat, mask


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=32):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


""" ---- MLP -> HGCN """


class HOGGCN_MLP(nn.Module):
    def __init__(self, n_feat, n_hid, nclass):
        super(HOGGCN_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_feat, n_hid),
            nn.ReLU(),
            nn.Linear(n_hid, nclass),
        )

    def forward(self, x):
        return self.mlp(x)

    def get_emb(self, x):
        return self.mlp[0](x).detach()


class HOGGCN(nn.Module):
    def __init__(self, input_dim, output_dim, company_num, person_num, dropout=0.5):
        super(HOGGCN, self).__init__()
        self.company_num = company_num
        self.person_num = person_num

        self.company_emb = nn.Embedding(company_num, input_dim)
        self.person_emb = nn.Embedding(person_num, input_dim)

        self.att_dim = 32
        self.proj = nn.Linear(input_dim + self.att_dim, input_dim, bias=False)

        self.GCN1 = GCN_homo(input_dim, input_dim, output_dim, dropout)
        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(output_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.attention = Attention(output_dim)
        self.tanh = nn.Tanh()

    def forward(self, hete_graph, idx, x, **kwargs):
        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        person_emb = self.person_emb(torch.LongTensor([i for i in range(self.person_num)]))
        company_attr_cal = torch.zeros((self.company_num, self.att_dim))
        company_attr_cal[idx] = torch.FloatTensor(x)
        company_emb = self.proj(torch.cat((company_emb, company_attr_cal), dim=1))
        emb = torch.cat((company_emb, person_emb), dim=0)

        output = kwargs.get("output")
        labels_for_lp = kwargs.get("labels_for_lp")
        adj = get_adj(hete_graph, emb.shape[0])
        adj = adj[idx][:, idx]
        bi_adj = adj.mm(adj)
        emb, y_hat, mask = self.GCN1(emb[idx], adj, bi_adj, output, labels_for_lp)
        return y_hat, emb
