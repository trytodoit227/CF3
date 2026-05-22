from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.model_utils import get_adj


class BMGCN_MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, dropout=0.5):
        super(BMGCN_MLP, self).__init__()
        if num_layers == 1:
            hidden_size = out_size

        self.pipeline = nn.Sequential(OrderedDict([
            ('layer_0', nn.Linear(in_size, hidden_size, bias=(num_layers != 1))),
            ('dropout_0', nn.Dropout(dropout)),
            ('relu_0', nn.ReLU())
        ]))

        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.pipeline.add_module('layer_{}'.format(i), nn.Linear(hidden_size, out_size, bias=True))
            else:
                self.pipeline.add_module('layer_{}'.format(i), nn.Linear(hidden_size, hidden_size, bias=True))
                self.pipeline.add_module('dropout_{}'.format(i), nn.Dropout(dropout))
                self.pipeline.add_module('relu_{}'.format(i), nn.ReLU())

        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, feature):
        return self.pipeline(feature)


class GraphConv(nn.Module):
    def __init__(self, in_size, out_size, bias=True):
        super(GraphConv, self).__init__()
        self.W = nn.Linear(in_size, out_size, bias)

    def forward(self, g, feature):
        h = torch.mm(g, feature)
        return self.W(h)


class GCN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, num_layers, dropout):
        super(GCN, self).__init__()
        if num_layers == 1:
            hidden_size = out_size

        self.num_layers = num_layers
        if dropout > 0.:
            self.feat_drop = nn.Dropout(dropout)
        else:
            self.feat_drop = lambda x: x

        self.W = nn.ModuleList([nn.Linear(in_size, hidden_size, bias=False)])
        self.gnn_layers = nn.ModuleList([GraphConv(in_size, hidden_size)])
        for i in range(1, num_layers):
            if i == num_layers - 1:
                self.W.append(nn.Linear(hidden_size, out_size, bias=False))
                self.gnn_layers.append(GraphConv(hidden_size, out_size, bias=False))
            else:
                self.W.append(nn.Linear(hidden_size, hidden_size, bias=False))
                self.gnn_layers.append(GraphConv(hidden_size, hidden_size, bias=False))

        self.weights_init()

    def weights_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, g, feature):
        h = feature
        for i, layer in enumerate(self.gnn_layers):
            if i == self.num_layers - 1:
                h = layer(g, h) + self.W[i](h)
            else:
                h = self.feat_drop(h)
                h = layer(g, h) + self.W[i](h)
                h = F.relu(h)
        return h


class BMGCN(nn.Module):
    def __init__(self, input_dim, output_dim, company_num, person_num):
        super(BMGCN, self).__init__()
        self.hidden_dim = 16
        self.company_num = company_num
        self.person_num = person_num

        self.company_emb = nn.Embedding(company_num, input_dim)
        self.person_emb = nn.Embedding(person_num, input_dim)

        self.att_dim = 32
        self.proj = nn.Linear(input_dim + self.att_dim, input_dim, bias=False)

        # 做好的效果：self.hidden_dim=input_dim,num_layers=4,dropout=0.02
        self.gcn = GCN(input_dim, self.hidden_dim, output_dim, num_layers=4, dropout=0.02)

        bias = np.ones((2, 2))
        np.fill_diagonal(bias, 2.0)
        self.bias = torch.FloatTensor(bias)

    def forward(self, hete_graph, idx, x, **kwargs):
        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        person_emb = self.person_emb(torch.LongTensor([i for i in range(self.person_num)]))
        company_attr_cal = torch.zeros((self.company_num, self.att_dim))
        company_attr_cal[idx] = torch.FloatTensor(x)
        company_emb = self.proj(torch.cat((company_emb, company_attr_cal), dim=1))
        emb = torch.cat((company_emb, person_emb), dim=0)
        adj = get_adj(hete_graph, emb.shape[0])
        adj = adj[idx][:, idx]

        B = kwargs.get("output")
        labels_oneHot = kwargs.get("labels_oneHot")

        H = get_block_matrix(adj, labels_oneHot)

        Q = torch.mm(H, H.t())
        Q = Q * self.bias
        Q = Q / torch.sum(Q, dim=1, keepdim=True)

        score = torch.mm(torch.mm(B, Q), B.t()) * adj
        zero_vec = -9e15 * torch.ones_like(score)
        g = torch.where(adj > 0, score, zero_vec)
        g = F.softmax(g, dim=1)

        output = self.gcn(g, emb[idx])

        return output.detach()


def get_block_matrix(adj, y):

    H = torch.mm(y.t(), adj)
    H = torch.mm(H, y) / torch.mm(H, torch.ones_like(y))
    return H
