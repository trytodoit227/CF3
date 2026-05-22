import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.model_utils import sparse_dropout
import numpy as np
import math
from utils.model_utils import generate_G_from_H, v2v


class HGNNP(nn.Module):
    def __init__(self, input_dim, output_dim, company_num, n_layer=1):
        super(HGNNP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layer_num = n_layer

        self.att_dim = 32
        self.proj = nn.Linear(input_dim + self.att_dim, input_dim, bias=False)

        self.company_num = company_num
        self.company_emb = nn.Embedding(company_num, input_dim)

        self.hgnnp = nn.ModuleList()
        for i in range(n_layer):
            self.hgnnp.append(HGNNPLayer(input_dim, output_dim, output_dim))

    def forward(self, hyp_graph, idx, x):
        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        company_attr_cal = torch.zeros((self.company_num, self.att_dim))
        company_attr_cal[idx] = torch.FloatTensor(x)
        company_emb = self.proj(torch.cat((company_emb, company_attr_cal), dim=1))
        #  hyp_graph is a list consist of sparse.coo_matrix
        hyp_graph = [h.incident_matrix().toarray() for h in hyp_graph]
        H = np.hstack(hyp_graph)
        H_dict, G = generate_G_from_H(H)
        G = torch.Tensor(G)
        for i in range(self.layer_num):
            # x = self.hgnnp[i](company_emb, G, **{'H_T': HT, 'D_e_neg_1': invDE, 'W_e': W})
            x = self.hgnnp[i](company_emb, G, **{'H_dict': H_dict})

        return x[idx]


class HGNNPLayer(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.2):
        super(HGNNPLayer, self).__init__()
        # self.dropout = dropout
        # self.hgc1 = HGNNP_conv(in_ch, n_hid)
        # self.hgc2 = HGNNP_conv(n_hid, n_class)

        self.layers = nn.ModuleList()
        self.layers.append(
            HGNNP_conv(in_ch, n_hid, drop_rate=dropout)
        )
        self.layers.append(
            HGNNP_conv(n_hid, n_class, is_last=True)
        )

    def forward(self, x, G, **kwargs):
        for layer in self.layers:
            x = layer(x, G, **kwargs)
        return x


class HGNNP_conv(nn.Module):
    def __init__(self, in_ft, out_ft, drop_rate=0.2, is_last=False, bias=True):
        super(HGNNP_conv, self).__init__()

        self.weight = nn.Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.is_last = is_last
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(drop_rate)
        self.theta = nn.Linear(in_ft, out_ft, bias=False)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, G, **kwargs):
        x_a = x.matmul(self.weight)
        if self.bias is not None:
            x_a = x_a + self.bias

        x = self.theta(x)
        x = v2v(x, aggr="mean", **kwargs)
        if not self.is_last:
            x = self.drop(self.act(x))

        x = x + x_a
        return x

