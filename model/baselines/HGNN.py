import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math


class HGNN(nn.Module):
    def __init__(self, input_dim, output_dim, company_num, n_layer=1):
        super(HGNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = n_layer

        self.att_dim = 32
        self.proj = nn.Linear(input_dim + self.att_dim, input_dim, bias=False)

        self.company_num = company_num
        self.company_emb = nn.Embedding(company_num, input_dim)

        self.hgnn = nn.ModuleList()
        for i in range(n_layer):
            self.hgnn.append(HGNNLayer(input_dim, output_dim, output_dim))

    def forward(self, hyp_graph, idx, x):
        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        company_attr_cal = torch.zeros((self.company_num, self.att_dim))
        company_attr_cal[idx] = torch.FloatTensor(x)
        company_emb = self.proj(torch.cat((company_emb, company_attr_cal), dim=1))
        #  hyp_graph is a list consist of sparse.coo_matrix
        hyp_graph = [h.incident_matrix().toarray() for h in hyp_graph]
        H = np.hstack(hyp_graph)
        G = generate_G_from_H(H)
        G = torch.Tensor(G)
        for i in range(self.layer_num):
            X = self.hgnn[i](company_emb, G)

        return X[idx]


class HGNNLayer(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNNLayer, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = torch.matmul(G, x)
        return x


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV + 1e-6, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G
