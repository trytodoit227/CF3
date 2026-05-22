import torch
import torch.nn as nn
import dgl.function as Fn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch_geometric.nn.inits import glorot
import numpy as np

from dgl.ops import edge_softmax
from dgl.nn.pytorch import TypedLinear
import dgl

import math

from utils.model_utils import scipy_sparse_mat_to_torch_sparse_tensor


class HeteGNNConv(nn.Module):
    def __init__(self, edge_dim, input_dim, output_dim, num_heads, num_etypes, feat_drop=0.0,
                 negative_slope=0.2, residual=True, beta=0.2):
        super(HeteGNNConv, self).__init__()
        self.edge_dim = edge_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.beta = beta

        self.edge_emb = nn.Parameter(torch.empty(size=(num_etypes, edge_dim)))

        self.W = nn.Parameter(torch.FloatTensor(output_dim, output_dim * num_heads))
        self.W_r = TypedLinear(edge_dim, edge_dim * num_heads, num_etypes)

        self.a_l = nn.Parameter(torch.empty(size=(1, num_heads, output_dim)))
        self.a_r = nn.Parameter(torch.empty(size=(1, num_heads, output_dim)))
        self.a_e = nn.Parameter(torch.empty(size=(1, num_heads, edge_dim)))

        nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_e.data, gain=1.414)

        self.activation = F.elu
        self.bn = nn.BatchNorm1d(output_dim)
        self.feat_drop = nn.Dropout(feat_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.proj_com = nn.Linear(input_dim, output_dim)
        self.proj_per = nn.Linear(input_dim, output_dim)

        if residual:
            self.residual = nn.Linear(output_dim, output_dim * num_heads)
        else:
            self.register_buffer("residual", None)

    def forward(self, company_emb, person_emb, edge_index, edge_type, presorted=False):
        company_emb = self.proj_com(company_emb)
        person_emb = self.proj_per(person_emb)
        emb = torch.cat((company_emb, person_emb), dim=0)
        h = self.bn(emb)
        edge_index = torch.LongTensor(edge_index).transpose(0, 1)
        edge_type = torch.LongTensor(edge_type)
        g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=6381)
        g.is_block = True
        # emb = self.feat_drop(h[:g.num_dst_nodes()])
        emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.output_dim)
        emb[torch.isnan(emb)] = 0.0

        edge_emb = self.W_r(self.edge_emb[edge_type], edge_type, presorted).view(-1, self.num_heads, self.edge_dim)

        row = g.edges()[0]  # edge_index[0] 节点一边
        col = g.edges()[1]  # edge_index[1] 节点另一边

        # 初始化bias
        a_n_bias = nn.Parameter(torch.zeros(edge_type.shape[0], 1))
        a_e_bias = nn.Parameter(torch.zeros(edge_type.shape[0], 1))
        trunc_normal_(a_n_bias, std=.02)
        trunc_normal_(a_e_bias, std=.02)

        h_l = (self.a_l * emb).sum(dim=-1)[row]  # q
        h_r = (self.a_r * emb).sum(dim=-1)[col]  # k
        h_e = (self.a_e * edge_emb).sum(dim=-1)  # v

        node_attention = self.leaky_relu(h_l * h_r + a_n_bias)
        node_attention = nn.functional.softmax(node_attention, dim=1)

        edge_attention = self.leaky_relu(h_e + a_e_bias)
        edge_attention = edge_softmax(g, edge_attention)  # 用图求边的注意力

        attention = (node_attention * edge_attention) / math.sqrt(self.output_dim)

        if self.num_heads == 1:
            attention = attention[:, 0]
            attention = attention.unsqueeze(1)

        with g.local_scope():  # 为了计算值并不想改变原始图，获得 h_output 的值
            emb = emb.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = attention
            g.srcdata['emb'] = emb
            g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'), Fn.sum('m', 'emb'))
            h_output = g.dstdata['emb'].view(-1, self.output_dim * self.num_heads)

        g.edata['alpha'] = attention
        if g.is_block:
            h = h[:g.num_dst_nodes()]

        if self.residual:
            res = self.residual(h)
            h_output = h_output + res

        h_output = self.activation(h_output)

        res_c, res_p = h_output[:company_emb.shape[0]], h_output[company_emb.shape[0]:]
        return res_c, res_p


class HyperGNNConv(nn.Module):
    def __init__(self, input_dim, output_dim, hyper_edge_num=3, num_layer=1, negative_slope=0.2):
        super(HyperGNNConv, self).__init__()
        self.negative_slope = negative_slope
        self.proj = nn.Linear(input_dim, output_dim, bias=False)

        self.act = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_dim)
        self.drop = nn.Dropout(self.negative_slope)
        residual = True
        if residual:
            self.residual = nn.Linear(output_dim, output_dim)
        else:
            self.register_buffer("residual", None)

    def forward(self, company_emb, hyp_graph):
        x = self.proj(company_emb)
        h = self.bn(x)
        res_x = 0
        for i in range(len(hyp_graph)):
            laplacian = scipy_sparse_mat_to_torch_sparse_tensor(hyp_graph[i].laplacian())  # 其实还是用到了超图的信息的
            rs = laplacian @ x
            rs = self.drop(self.act(rs))
            if self.residual:
                res = self.residual(h)
                rs = rs + res
            res_x += rs

        return res_x


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, company_num, person_num, rel_num, device, n_layer=3, num_heads=1,
                 dropout=0.2):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.company_num = company_num
        self.person_num = person_num
        self.rel_num = rel_num
        self.cause_type = 11
        self.device = device
        self.court_type = 4
        self.category_ = 4
        self.num_heads = num_heads
        self.dropout = dropout
        self.n_layer = n_layer
        self.hyp_n_layer = 1

        self.company_emb = nn.Embedding(company_num, input_dim)
        self.person_emb = nn.Embedding(person_num, input_dim)

        self.att_dim = 32
        self.proj = nn.Linear(input_dim + self.att_dim, input_dim)

        self.hetegnn_layers = nn.ModuleList()
        self.hypergnn_layers = HyperGNNConv(input_dim, output_dim, num_layer=1)

        for i in range(n_layer):
            if i == 0:
                self.hetegnn_layers.append(
                    HeteGNNConv(input_dim, input_dim, output_dim // num_heads, num_heads, num_etypes=20))
            else:
                self.hetegnn_layers.append(
                    HeteGNNConv(input_dim, output_dim, output_dim // num_heads, num_heads, num_etypes=20))

        self.final_proj = nn.Sequential(nn.Linear(output_dim, output_dim, bias=False), nn.Mish(),
                                        nn.Linear(output_dim, output_dim, bias=False))

        # self.alpha = torch.full_like(torch.ones((1)), 0.5)
        self.alpha = torch.ones(self.company_num, 1)
        glorot(self.company_emb)
        glorot(self.person_emb)

    def forward(self, hete_graph, hyp_graph, idx, x):

        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        person_emb = self.person_emb(torch.LongTensor([i for i in range(self.person_num)]))
        company_attr_cal = torch.zeros((self.company_num, self.att_dim))
        company_attr_cal[idx] = torch.FloatTensor(x)
        company_emb_info = self.proj(torch.cat((company_emb, company_attr_cal), dim=1))

        company_emb_hyper = self.hypergnn_layers(company_emb_info, hyp_graph)

        # 可视化度节点
        # edge_index=torch.LongTensor(edge_index).transpose(0,1)
        # # edge_index=add_self_loops(to_undirected(edge_index))
        # edge_index=to_undirected(edge_index)
        # # print(edge_index[0])
        # dg=degree(edge_index[0]).numpy()

        # g = nx.Graph()
        # src = edge_index[0].numpy()
        # dst = edge_index[1].numpy()
        # edgelist = zip(src, dst)
        # for i, j in edgelist:
        #     g.add_edge(i, j)
        # # nx.draw(g, pos=nx.spring_layout(g),with_labels=g.nodes)
        # # nx.draw(g, pos=nx.spring_layout(g,k=0.02))
        # nx.draw(g, pos=nx.shell_layout(g))

        # # plt.savefig('test.png')
        # plt.show()
        # plt.hist(dg)
        # plt.hist(dg, bins=40, rwidth=0.9, density=True)
        # plt.show()

        # print('node degree statistic: %s'% degree(edge_index[0]))

        edge_index, edge_type, edge_weight = hete_graph
        for i in range(self.n_layer):
            if i == 0:
                company_emb_hete, person_emb = self.hetegnn_layers[i](company_emb_info, person_emb, edge_index,
                                                                      edge_type)
            else:
                company_emb_hete, person_emb = self.hetegnn_layers[i](company_emb_hete, person_emb, edge_index,
                                                                      edge_type)

        alpha = torch.sigmoid(self.alpha)  # 转换到 0~1 之间
        company_emb_final = alpha * company_emb_hete + (1 - alpha) * company_emb_hyper
        company_emb_final = self.final_proj(company_emb_final)
        return company_emb_final[idx].to(self.device)