import torch
import torch.nn as nn
import dgl.function as Fn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from torch.nn.parameter import Parameter

from dgl.ops import edge_softmax
from dgl.nn.pytorch import TypedLinear
import dgl
import numpy as np

from utils.model_utils import *


def get_existing_edge(new_edge_index, new_edge_weight, edge_index):
    keep_edge_idx = []
    for i in range(len(new_edge_index.T)):
        elmt = np.array(new_edge_index.T[i])
        pos_new_edge = np.where(np.all(np.array(edge_index.T) == elmt, axis=1))[0]
        if pos_new_edge.size > 0:
            keep_edge_idx.append(pos_new_edge[0])
    kept_edges = edge_index.T[keep_edge_idx]
    kept_edges = np.array(kept_edges)
    # kept_edge_type = edge_type[keep_edge_idx]
    kept_edge_weight = new_edge_weight[keep_edge_idx]
    if kept_edges.ndim == 1:
        kept_edges = kept_edges.reshape(0, 2)
    return kept_edges, kept_edge_weight


def get_new_edge(new_edge_index, new_edge_weight, edge_index):
    new_added_edges = []
    new_added_edge_idx = []
    for i in range(len(new_edge_index.T)):
        elmt = np.array(new_edge_index.T[i])
        pos_new_edge = np.where(np.all(np.array(edge_index.T) == elmt, axis=1))[0]
        if pos_new_edge.size == 0:
            new_added_edges.append(elmt)
            new_added_edge_idx.append(i)
    new_added_edges = np.array(new_added_edges)
    new_added_edge_weight = new_edge_weight[new_added_edge_idx]
    if new_added_edges.ndim == 1:
        new_added_edges = new_added_edges.reshape(0, 2)
    return new_added_edges, new_added_edge_weight


def identity(x, batch):
    return x


def get_readout_layers(readout):
    readout_func_dict = {
        # "mean": global_mean_pool,
        # "sum": global_add_pool,
        # "max": global_max_pool,
        "identity": identity,
        # "cat_max_sum": cat_max_sum,
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    return readout_func_dict[readout.lower()]


class GNNPool(nn.Module):
    def __init__(self, readout):
        super().__init__()
        self.readout = get_readout_layers(readout)

    def forward(self, x, batch):
        return self.readout(x, batch)


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


class MODELPerturb(nn.Module):
    def __init__(self, input_dim, output_dim, adj, company_num, person_num, rel_num, device, beta, n_layer=3,
                 num_heads=1, dropout=0.2):
        super(MODELPerturb, self).__init__()
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

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.adj = adj
        self.beta = beta
        self.num_nodes = self.adj.shape[0]
        self.num_layers = 2
        self.hidden_dim = 12
        self.dropout = 0.0
        self.num_nodes = self.adj.shape[0]
        self.readout_layer = GNNPool('identity')
        self.edge_additions = False
        self.P_vec_size = (int((self.num_nodes * self.num_nodes - self.num_nodes) / 2) + self.num_nodes)
        self.P_vec = Parameter(torch.FloatTensor(torch.ones(self.P_vec_size)))
        self.reset_parameters()

    def reset_parameters(self, eps=10 ** -4):
        # Think more about how to initialize this
        with torch.no_grad():
            if self.edge_additions:
                adj_vec = create_vec_from_symm_matrix(self.adj, self.P_vec_size).numpy()
                for i in range(len(adj_vec)):
                    if i < 1:
                        adj_vec[i] = adj_vec[i] - eps
                    else:
                        adj_vec[i] = adj_vec[i] + eps
                torch.add(
                    self.P_vec, torch.FloatTensor(adj_vec)
                )  # self.P_vec is all 0s
            else:
                torch.sub(self.P_vec, eps)

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

        return company_emb_final[idx]

    def forward_emb(self, *args, **kwargs):
        batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=kwargs['x'].device)
        # node embedding for GNN
        emb = self.get_emb(*args, **kwargs)
        x = self.readout_layer(emb, batch)
        self.logits = kwargs['classifier'].forward(x).reshape(-1, 2)
        self.probs = F.softmax(self.logits, dim=1)
        return F.log_softmax(self.logits, dim=1)

    def forward_emb_prediction(self, *args, **kwargs):
        batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=kwargs['x'].device)
        # node embedding for GNN
        emb, self.P = self.get_emb_prediction(*args, **kwargs)
        x = self.readout_layer(emb, batch)
        self.logits = kwargs['classifier'].forward(x).reshape(-1, 2)
        self.probs = F.softmax(self.logits, dim=1)
        return F.log_softmax(self.logits, dim=1), self.P

    def get_emb(self, *args, **kwargs):
        mapping_dict = kwargs['mapping']
        swapped_mapping_dict = {v: k for k, v in mapping_dict.items()}
        x = kwargs['x']
        edge_index = torch.LongTensor(kwargs['hete_graph'][0]).transpose(0, 1)
        edge_type = torch.LongTensor(kwargs['hete_graph'][1])
        hyp_graph = kwargs['hyp_graph']
        self.sub_adj = kwargs["adj"]
        # self.sub_adj = from_edge_index_to_adj_torch(edge_index, edge_weight, self.num_nodes)
        # Get adj matrix with only edges that have nonzero attention weights
        # Same as normalize_adj in utils.py except includes P_hat in A_tilde
        self.P_hat_symm = create_symm_matrix_from_vec(self.P_vec, self.num_nodes)  # Ensure symmetry

        A_tilde = torch.FloatTensor(self.num_nodes, self.num_nodes)
        A_tilde.requires_grad = True

        if self.edge_additions:  # Learn new adj matrix directly
            A_tilde = F.sigmoid(self.P_hat_symm) + torch.eye(self.num_nodes)  # Use sigmoid to bound P_hat in [0,1]
        else:  # Learn P_hat that gets multiplied element-wise with adj -- only edge deletions
            A_tilde = F.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(
                self.num_nodes)  # Use sigmoid to bound P_hat in [0,1]
        D_tilde = get_degree_matrix(A_tilde).detach()  # Don't need gradient of this
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
        new_edge_index, new_edge_weight = from_adj_to_edge_index_torch(norm_adj)
        new_edge_weight = new_edge_weight.detach().numpy()

        kept_edges, kept_edge_weight = get_existing_edge(new_edge_index, new_edge_weight, edge_index)
        new_added_edges, new_added_edge_weight = get_new_edge(new_edge_index, new_edge_weight, edge_index)
        new_edge_index = torch.LongTensor(np.concatenate((kept_edges, new_added_edges), 0).T)
        new_edge_weight = torch.FloatTensor(np.concatenate((kept_edge_weight, new_added_edge_weight), 0))
        for i in range(new_edge_index.size(0)):
            new_edge_index[i] = torch.tensor([swapped_mapping_dict[idx.item()] for idx in new_edge_index[i]])
        new_edge_type = get_new_edge_type_from_new_edge_index(new_edge_index, new_edge_weight, edge_type, edge_index)
        new_edge_index_list = new_edge_index.transpose(0, 1).tolist()
        new_edge_type_list = [int(i) for i in new_edge_type.tolist()]
        new_edge_weight_list = new_edge_weight.tolist()
        hete_graph = [new_edge_index_list, new_edge_type_list, new_edge_weight_list]

        # for layer in self.convs:
        #     x = layer(x, new_edge_index, new_edge_attr * new_edge_weight[:, None])
        #     x = F.relu(x)
        #     x = F.dropout(x, self.dropout, training=self.training)
        idx = list(mapping_dict.keys())
        x = self.forward(hete_graph, hyp_graph, idx, x.float())
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x.to(self.device)

    def get_emb_prediction(self, *args, **kwargs):
        mapping_dict = kwargs['mapping']
        swapped_mapping_dict = {v: k for k, v in mapping_dict.items()}
        x = kwargs['x']
        edge_index = torch.LongTensor(kwargs['hete_graph'][0]).transpose(0, 1)
        edge_type = torch.LongTensor(kwargs['hete_graph'][1])
        hyp_graph = kwargs['hyp_graph']
        # Same as forward but uses P instead of P_hat ==> non-differentiable
        # but needed for actual predictions
        self.P = (F.sigmoid(self.P_hat_symm) >= 0.5).float()  # threshold P_hat
        if self.edge_additions:
            A_tilde = self.P + torch.eye(self.num_nodes)
        else:
            A_tilde = self.P * self.adj + torch.eye(self.num_nodes)

        D_tilde = get_degree_matrix(A_tilde)
        # Raise to power -1/2, set all infs to 0s
        D_tilde_exp = D_tilde ** (-1 / 2)
        D_tilde_exp[torch.isinf(D_tilde_exp)] = 0

        # Create norm_adj = (D + I)^(-1/2) * (A + I) * (D + I) ^(-1/2)
        norm_adj = torch.mm(torch.mm(D_tilde_exp, A_tilde), D_tilde_exp)
        new_edge_index, new_edge_weight = from_adj_to_edge_index_torch(norm_adj)
        new_edge_weight = new_edge_weight.detach().numpy()
        kept_edges, kept_edge_weight = get_existing_edge(new_edge_index, new_edge_weight, edge_index)
        new_added_edges, new_added_edge_weight = get_new_edge(new_edge_index, new_edge_weight, edge_index)
        new_edge_index = torch.LongTensor(np.concatenate((kept_edges, np.array(new_added_edges)), 0).T)
        new_edge_weight = torch.FloatTensor(np.concatenate((kept_edge_weight, new_added_edge_weight), 0))
        for i in range(new_edge_index.size(0)):
            new_edge_index[i] = torch.tensor([swapped_mapping_dict[idx.item()] for idx in new_edge_index[i]])
        new_edge_type = get_new_edge_type_from_new_edge_index(new_edge_index, new_edge_weight, edge_type, edge_index)
        new_edge_index_list = new_edge_index.transpose(0, 1).tolist()
        new_edge_type_list = [int(i) for i in new_edge_type.tolist()]
        new_edge_weight_list = new_edge_weight.tolist()
        hete_graph = [new_edge_index_list, new_edge_type_list, new_edge_weight_list]

        ###################
        idx = list(mapping_dict.keys())
        x = self.forward(hete_graph, hyp_graph, idx, x.float())
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        return x.to(self.device), self.P.to(self.device)

    def cf_loss(self, output, y_pred_orig, y_pred_new_actual):
        pred_same = (y_pred_new_actual == y_pred_orig).float()
        # Need dim >=2 for F.nll_loss to work
        if output.ndim < 2:
            output = output.unsqueeze(0)

        if self.edge_additions:
            cf_adj = self.P.to(self.device)
        else:
            cf_adj = self.P * self.adj.to(self.device)
        cf_adj.requires_grad = (True)  # Need to change this otherwise loss_graph_dist has no gradient
        # Want negative in front to maximize loss instead of minimizing it to find CFs
        '''
        loss function: -log(1-softmax(Pred)[i]) * label[i] sum over i
        pred: [N, num_class]
        label: [N, ]
        '''
        # print("output", output)
        bsz, C = output.size()
        inf_diag = torch.diag(-torch.ones((C)) / 0).unsqueeze(0).repeat(bsz, 1, 1).to(output.device)
        neg_prop = (output.unsqueeze(1).expand(bsz, C, C) + inf_diag).logsumexp(-1) - output.logsumexp(-1).unsqueeze(
            1).repeat(1, C)
        criterion = torch.nn.NLLLoss()
        loss_pred = criterion(neg_prop, y_pred_orig.unsqueeze(0).to(self.device))
        loss_graph_dist = (sum(sum(abs(cf_adj - self.adj.to(self.device)))) / 2)  # Number of edges changed (symmetrical)

        # Zero-out loss_pred with pred_same if prediction flips
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss_total, loss_pred, loss_graph_dist, cf_adj
