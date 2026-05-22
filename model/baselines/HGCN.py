import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import degree


class HGCN(nn.Module):
    # def __init__(self, in_hid, out_hid, company_num, person_num, type_fusion='att', type_att_size=64):
    # Best Test Prob: 0.5777 Best Test Acc: 0.6646 Best Test Pre: 0.6947 Best Test Recall: 0.8599 Best Test F1: 0.7686 Best Test ROC: 0.7563
    def __init__(self, in_hid, out_hid, company_num, person_num, type_fusion='mean', type_att_size=64):
        super(HGCN, self).__init__()
        self.company_num = company_num
        self.person_num = person_num
        self.company_emb = nn.Embedding(company_num, in_hid)
        self.person_emb = nn.Embedding(person_num, in_hid)
        self.att_dim = 32
        self.attr_proj = nn.Linear(in_hid + self.att_dim, in_hid)
        # self.attr_proj = nn.Linear(in_hid + self.att_dim, in_hid, bias=False)   # 在 type_fusion='mean' 的条件下acc会达到0.79
        # TODO: company:0,person:1
        # net_schema = {0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [8, 9, 10, 11]}
        net_schema = {0: [0, 1, 2, 3, 4, 5, 6, 7], 1: [8, 9, 10, 11]}
        layer_shape = [in_hid, in_hid, out_hid, out_hid, out_hid]
        label_keys = [0, 1]

        self.hgc1 = HeteGCNLayer(net_schema, layer_shape[0], layer_shape[1], type_fusion, type_att_size)
        self.hgc2 = HeteGCNLayer(net_schema, layer_shape[1], layer_shape[2], type_fusion, type_att_size)
        self.hgc3 = HeteGCNLayer(net_schema, layer_shape[2], layer_shape[3], type_fusion, type_att_size)
        self.hgc4 = HeteGCNLayer(net_schema, layer_shape[3], layer_shape[4], type_fusion, type_att_size)

        self.embd2class = nn.ParameterDict()
        self.bias = nn.ParameterDict()
        self.label_keys = label_keys
        for k in label_keys:
            self.embd2class[str(k)] = nn.Parameter(torch.FloatTensor(layer_shape[-2], layer_shape[-1]))
            nn.init.xavier_uniform_(self.embd2class[str(k)].data, gain=1.414)
            self.bias[str(k)] = nn.Parameter(torch.FloatTensor(1, layer_shape[-1]))
            nn.init.xavier_uniform_(self.bias[str(k)].data, gain=1.414)

    def forward(self, hete_graph, idx, x):
        edge_index, edge_type, edge_weight = hete_graph
        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        person_emb = self.person_emb(torch.LongTensor([i for i in range(self.person_num)]))
        edge_type = torch.LongTensor(edge_type)
        edge_index = torch.LongTensor(edge_index).transpose(0, 1)
        company_attr_cal = torch.zeros((self.company_num, self.att_dim))
        company_attr_cal[idx] = torch.FloatTensor(x)
        company_emb = self.attr_proj(torch.cat((company_emb, company_attr_cal), dim=1))

        adj_dict = {}
        ft_dict = {0: company_emb, 1: person_emb}
        node_num = [self.company_num, self.person_num]

        # TODO: investment, branch, rev_investment,rev_branch:C-C
        rel_tp = 0
        # for i in [0, 1, 2, 3]:
        for i in [3, 4, 5, 6]:
        # for i in [3, 9, 4, 10, 5, 6, 7, 8, 11, 12, 13, 14]:
            index = (edge_type == i) & (edge_index[0] < self.company_num) & (edge_index[1] < self.company_num)
            sub_edge_index = edge_index[:, index]
            val = torch.ones(sub_edge_index.shape[1])
            dg = degree(sub_edge_index[0], num_nodes=node_num[0]).pow(-1)
            dg[dg == float('inf')] = 0
            adj_dict[rel_tp] = [torch.sparse.FloatTensor(sub_edge_index, val, (node_num[0], node_num[0])),
                                dg.unsqueeze(1)]
            rel_tp += 1
        # TODO::C-P
        # for i in [0, 2, 4, 8]:
        for i in [0, 1, 2, 3]:
            index = (edge_type == i) & (edge_index[0] < self.company_num) & (edge_index[1] >= self.company_num)
            sub_edge_index = edge_index[:, index]
            sub_edge_index[1] = sub_edge_index[1] - node_num[0]
            val = torch.ones(sub_edge_index.shape[1])
            dg = degree(sub_edge_index[0], num_nodes=node_num[0]).pow(-1)
            dg[dg == float('inf')] = 0
            adj_dict[rel_tp] = [torch.sparse.FloatTensor(sub_edge_index, val, (node_num[0], node_num[1])),
                                dg.unsqueeze(1)]
            rel_tp += 1
        # # TODO: P-C
        # # for i in [1, 3, 5, 9]:
        for i in [0, 1, 2, 3]:
            index = (edge_type == i) & (edge_index[0] >= self.company_num) & (edge_index[1] < self.company_num)
            sub_edge_index = edge_index[:, index]
            sub_edge_index[0] = sub_edge_index[0] - node_num[0]
            val = torch.ones(sub_edge_index.shape[1])
            dg = degree(sub_edge_index[0], num_nodes=node_num[1]).pow(-1)
            dg[dg == float('inf')] = 0
            adj_dict[rel_tp] = [torch.sparse.FloatTensor(sub_edge_index, val, (node_num[1], node_num[0])),
                                dg.unsqueeze(1)]
            rel_tp += 1

        x_dict = self.hgc1(ft_dict, adj_dict)
        x_dict = self.non_linear(x_dict)
        x_dict = self.dropout_ft(x_dict, 0.5)

        x_dict = self.hgc2(x_dict, adj_dict)
        x_dict = self.non_linear(x_dict)
        x_dict = self.dropout_ft(x_dict, 0.5)

        x_dict = self.hgc3(x_dict, adj_dict)
        x_dict = self.non_linear(x_dict)
        x_dict = self.dropout_ft(x_dict, 0.5)

        x_dict = self.hgc4(x_dict, adj_dict)

        logits = {}
        embd = {}
        for k in self.label_keys:
            embd[k] = x_dict[k]
            logits[k] = torch.mm(x_dict[k], self.embd2class[str(k)]) + self.bias[str(k)]
        feat = torch.cat((logits[0], logits[1]), dim=0)
        return feat[idx]

    def non_linear(self, x_dict):
        y_dict = {}
        for k in x_dict:
            y_dict[k] = F.elu(x_dict[k])
        return y_dict

    def dropout_ft(self, x_dict, dropout):
        y_dict = {}
        for k in x_dict:
            y_dict[k] = F.dropout(x_dict[k], dropout, training=self.training)
        return y_dict


class HeteGCNLayer(nn.Module):

    def __init__(self, net_schema, in_layer_shape, out_layer_shape, type_fusion, type_att_size):
        super(HeteGCNLayer, self).__init__()

        self.net_schema = net_schema
        self.in_layer_shape = in_layer_shape
        self.out_layer_shape = out_layer_shape

        self.hete_agg = nn.ModuleDict()
        for k in net_schema:
            self.hete_agg[str(k)] = HeteAggregateLayer(k, net_schema[k], in_layer_shape, out_layer_shape, type_fusion,
                                                       type_att_size)

    def forward(self, x_dict, adj_dict):

        ret_x_dict = {}
        for k in self.hete_agg.keys():
            ret_x_dict[int(k)] = self.hete_agg[str(k)](x_dict, adj_dict)

        return ret_x_dict


class HeteAggregateLayer(nn.Module):

    def __init__(self, curr_k, nb_list, in_layer_shape, out_shape, type_fusion, type_att_size):
        super(HeteAggregateLayer, self).__init__()

        self.nb_list = nb_list
        self.curr_k = curr_k
        self.type_fusion = type_fusion
        # TODO: change for directed edge
        # self.target_node_type=[0,0,0,0,0,0,1,1,1,1,0,0,0,0]
        self.target_node_type = [0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]
        # self.target_node_type=[0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0]

        self.W_rel = nn.ParameterDict()
        for k in nb_list:
            self.W_rel[str(k)] = nn.Parameter(torch.FloatTensor(in_layer_shape, out_shape))
            nn.init.xavier_uniform_(self.W_rel[str(k)].data, gain=1.414)

        self.w_self = nn.Parameter(torch.FloatTensor(in_layer_shape, out_shape))
        nn.init.xavier_uniform_(self.w_self.data, gain=1.414)

        self.bias = nn.Parameter(torch.FloatTensor(1, out_shape))
        nn.init.xavier_uniform_(self.bias.data, gain=1.414)

        if type_fusion == 'att':
            self.w_query = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
            nn.init.xavier_uniform_(self.w_query.data, gain=1.414)
            self.w_keys = nn.Parameter(torch.FloatTensor(out_shape, type_att_size))
            nn.init.xavier_uniform_(self.w_keys.data, gain=1.414)
            self.w_att = nn.Parameter(torch.FloatTensor(2 * type_att_size, 1))
            nn.init.xavier_uniform_(self.w_att.data, gain=1.414)

    def forward(self, x_dict, adj_dict):

        self_ft = torch.mm(x_dict[self.curr_k], self.w_self)

        nb_ft_list = [self_ft]
        nb_name = [self.curr_k]
        for k in self.nb_list:
            nb_ft = torch.mm(x_dict[self.target_node_type[k]], self.W_rel[str(k)])
            nb_ft = torch.matmul(adj_dict[k][0], nb_ft) * adj_dict[k][1]
            # nb_ft = torch.spmm(adj_dict[k][0], nb_ft)*adj_dict[k][1]
            # print(nb_ft.shape,k)
            nb_ft_list.append(nb_ft)
            nb_name.append(k)

        if self.type_fusion == 'mean':
            agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mean(1)

        elif self.type_fusion == 'att':
            att_query = torch.mm(self_ft, self.w_query).repeat(len(nb_ft_list), 1)
            att_keys = torch.mm(torch.cat(nb_ft_list, 0), self.w_keys)
            # print(att_query.shape,att_keys.shape,self_ft.shape,len(nb_ft_list))
            att_input = torch.cat([att_keys, att_query], 1)
            att_input = F.dropout(att_input, 0.5, training=self.training)
            e = F.elu(torch.matmul(att_input, self.w_att))
            attention = F.softmax(e.view(len(nb_ft_list), -1).transpose(0, 1), dim=1)
            agg_nb_ft = torch.cat([nb_ft.unsqueeze(1) for nb_ft in nb_ft_list], 1).mul(attention.unsqueeze(-1)).sum(1)
            # print('curr key: ', self.curr_k, 'nb att: ', nb_name, attention.mean(0).tolist())

        output = agg_nb_ft + self.bias

        return output
