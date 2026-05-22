import torch
import torch.nn as nn
import dgl.function as Fn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
import numpy as np

from dgl.ops import edge_softmax
from dgl.nn.pytorch import TypedLinear
import dgl

from utils.model_utils import generate_G_from_H, v2v


class HeteGNNConv(nn.Module):
    def __init__(self, edge_dim, input_dim, output_dim, num_heads, num_etypes, feat_drop=0.0,
                 negative_slope=0.2, residual=True, beta=0.2):
        super(HeteGNNConv, self).__init__()
        self.edge_dim = edge_dim
        self.in_dim = input_dim
        self.out_dim = output_dim
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
        emb = self.feat_drop(h[:g.num_dst_nodes()])
        emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.out_dim)
        emb[torch.isnan(emb)] = 0.0

        edge_emb = self.W_r(self.edge_emb[edge_type], edge_type, presorted).view(-1, self.num_heads, self.edge_dim)

        row = g.edges()[0]  # edge_index[0]
        col = g.edges()[1]  # edge_index[1]

        h_l = (self.a_l * emb).sum(dim=-1)[row]
        h_r = (self.a_r * emb).sum(dim=-1)[col]
        h_e = (self.a_e * edge_emb).sum(dim=-1)
        edge_attention = self.leaky_relu(h_l + h_r + h_e)

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
            g.srcdata['emb'] = emb
            g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'), Fn.sum('m', 'emb'))
            h_output = g.dstdata['emb'].view(-1, self.out_dim * self.num_heads)

        g.edata['alpha'] = edge_attention
        if g.is_block:
            h = h[:g.num_dst_nodes()]
        if self.residual:
            res = self.residual(h)
            h_output = h_output + res

        h_output = self.activation(h_output)

        res_c, res_p = h_output[:company_emb.shape[0]], h_output[company_emb.shape[0]:]
        return res_c, res_p


class HyperGNNConv(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.2, residual=True):
        super(HyperGNNConv, self).__init__()

        self.proj_com = nn.Linear(input_dim, output_dim)

        self.act = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_dim)
        self.drop = nn.Dropout(dropout)
        self.theta = nn.Linear(output_dim, output_dim)
        if residual:
            self.residual = nn.Linear(output_dim, output_dim)
        else:
            self.register_buffer("residual", None)

    def forward(self, company_emb, **kwargs):
        company_emb = self.proj_com(company_emb)
        x = self.theta(company_emb)
        h = self.bn(x)
        x = v2v(x, aggr="mean", **kwargs)
        x = self.drop(self.act(x))
        if self.residual:
            res = self.residual(h)
            x = x + res
        return x


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
        self.hyp_n_layer = 3

        self.company_emb = nn.Embedding(company_num, input_dim)
        self.person_emb = nn.Embedding(person_num, input_dim)

        self.att_dim = 32
        self.proj = nn.Linear(input_dim + self.att_dim, input_dim)

        self.hetegnn_layers = nn.ModuleList()
        self.hypergnn_layers = nn.ModuleList()

        for i in range(self.hyp_n_layer):
            if i == 0:
                self.hypergnn_layers.append(HyperGNNConv(input_dim, output_dim))
            else:
                self.hypergnn_layers.append(HyperGNNConv(output_dim, output_dim))

        for i in range(n_layer):
            if i == 0:
                self.hetegnn_layers.append(
                    HeteGNNConv(input_dim, input_dim, output_dim // num_heads, num_heads, num_etypes=20))
            else:
                self.hetegnn_layers.append(
                    HeteGNNConv(input_dim, output_dim, output_dim // num_heads, num_heads, num_etypes=20))

        self.final_proj = nn.Sequential(nn.Linear(output_dim, output_dim, bias=False), nn.Mish(),
                                        nn.Linear(output_dim, output_dim, bias=False))

        self.alpha = torch.full_like(torch.ones((1)), 2)
        glorot(self.company_emb)
        glorot(self.person_emb)

    def forward(self, hete_graph, hyp_graph, idx, x):

        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        person_emb = self.person_emb(torch.LongTensor([i for i in range(self.person_num)]))
        company_attr_cal = torch.zeros((self.company_num, self.att_dim))
        company_attr_cal[idx] = torch.FloatTensor(x)
        company_emb_info = self.proj(torch.cat((company_emb, company_attr_cal), dim=1))

        hg = [h.incident_matrix().toarray() for h in hyp_graph]
        H = np.hstack(hg)
        H_dict, _ = generate_G_from_H(H)
        for i in range(self.hyp_n_layer):
            if i == 0:
                company_emb_hyper = self.hypergnn_layers[i](company_emb_info, **{'H_dict': H_dict})
            else:
                company_emb_hyper = self.hypergnn_layers[i](company_emb_hyper, **{'H_dict': H_dict})

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


class Model_Ablation_Hyper(nn.Module):
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
        self.hyp_n_layer = 3

        self.company_emb = nn.Embedding(company_num, input_dim)
        self.person_emb = nn.Embedding(person_num, input_dim)

        self.att_dim = 32
        self.proj = nn.Linear(input_dim + self.att_dim, input_dim)

        self.hetegnn_layers = nn.ModuleList()

        for i in range(n_layer):
            if i == 0:
                self.hetegnn_layers.append(
                    HeteGNNConv(input_dim, input_dim, output_dim // num_heads, num_heads, num_etypes=20))
            else:
                self.hetegnn_layers.append(
                    HeteGNNConv(input_dim, output_dim, output_dim // num_heads, num_heads, num_etypes=20))

        self.final_proj = nn.Sequential(nn.Linear(output_dim, output_dim, bias=False), nn.Mish(),
                                        nn.Linear(output_dim, output_dim, bias=False))

        self.alpha = torch.full_like(torch.ones((1)), 2)
        glorot(self.company_emb)
        glorot(self.person_emb)

    def forward(self, hete_graph, hyp_graph, idx, x):

        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        person_emb = self.person_emb(torch.LongTensor([i for i in range(self.person_num)]))
        company_attr_cal = torch.zeros((self.company_num, self.att_dim))
        company_attr_cal[idx] = torch.FloatTensor(x)
        company_emb_info = self.proj(torch.cat((company_emb, company_attr_cal), dim=1))

        edge_index, edge_type, edge_weight = hete_graph
        for i in range(self.n_layer):
            if i == 0:
                company_emb_hete, person_emb = self.hetegnn_layers[i](company_emb_info, person_emb, edge_index,
                                                                      edge_type)
            else:
                company_emb_hete, person_emb = self.hetegnn_layers[i](company_emb_hete, person_emb, edge_index,
                                                                      edge_type)

        company_emb_final = self.final_proj(company_emb_hete)
        return company_emb_final[idx]


class Model_Ablation_Hete(nn.Module):
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
        self.hyp_n_layer = 3

        self.company_emb = nn.Embedding(company_num, input_dim)
        self.person_emb = nn.Embedding(person_num, input_dim)

        self.att_dim = 32
        self.proj = nn.Linear(input_dim + self.att_dim, input_dim)

        self.hypergnn_layers = nn.ModuleList()

        for i in range(self.hyp_n_layer):
            if i == 0:
                self.hypergnn_layers.append(HyperGNNConv(input_dim, output_dim))
            else:
                self.hypergnn_layers.append(HyperGNNConv(output_dim, output_dim))

        self.final_proj = nn.Sequential(nn.Linear(output_dim, output_dim, bias=False), nn.Mish(),
                                        nn.Linear(output_dim, output_dim, bias=False))

        self.alpha = torch.full_like(torch.ones((1)), 2)
        glorot(self.company_emb)
        glorot(self.person_emb)

    def forward(self, hete_graph, hyp_graph, idx, x):

        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        person_emb = self.person_emb(torch.LongTensor([i for i in range(self.person_num)]))
        company_attr_cal = torch.zeros((self.company_num, self.att_dim))
        company_attr_cal[idx] = torch.FloatTensor(x)
        company_emb_info = self.proj(torch.cat((company_emb, company_attr_cal), dim=1))

        hg = [h.incident_matrix().toarray() for h in hyp_graph]
        H = np.hstack(hg)
        H_dict, _ = generate_G_from_H(H)
        for i in range(self.hyp_n_layer):
            if i == 0:
                company_emb_hyper = self.hypergnn_layers[i](company_emb_info, **{'H_dict': H_dict})
            else:
                company_emb_hyper = self.hypergnn_layers[i](company_emb_hyper, **{'H_dict': H_dict})

        company_emb_final = self.final_proj(company_emb_hyper)
        return company_emb_final[idx]
