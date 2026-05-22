import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing


class HETGNN(MessagePassing):
    def __init__(self, embed_d, output_dim, company_num, person_num, rel_num, n_layer=2):
        super(HETGNN, self).__init__()
        self.embed_d = embed_d
        self.company_num = company_num
        self.person_num = person_num
        self.rel_num = rel_num
        self.output_dim = output_dim

        # self.ca_emb=nn.Embedding(cause_type_num,11)
        # self.court_emb=nn.Embedding(court_type_num,4)
        # self.cate_emb=nn.Embedding(res_num,4)

        # self.c_content_rnn = nn.LSTM(embed_d, embed_d, 1, bidirectional = True)

        self.neigh_rnn = nn.ModuleList()
        for i in range(n_layer):
            self.neigh_rnn.append(nn.LSTM(output_dim, int(output_dim / 2), 1, bidirectional=True))
        # self.c_neigh_rnn = nn.LSTM(embed_d, embed_d, 1, bidirectional = True)
        # self.p_neigh_rnn = nn.LSTM(embed_d, embed_d, 1, bidirectional = True)
        self.out_proj = nn.Linear(output_dim * rel_num, output_dim, bias=False)

        self.c_neigh_att = nn.Parameter(torch.ones(output_dim, 1), requires_grad=True)
        self.p_neigh_att = nn.Parameter(torch.ones(output_dim, 1), requires_grad=True)

        self.proj_c = nn.Linear(32 + embed_d, output_dim, bias=False)
        self.person_emb = nn.Embedding(person_num, output_dim)
        self.company_emb = nn.Embedding(company_num, embed_d)

        self.softmax = nn.Softmax(dim=1)
        self.act = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(embed_d)

        # def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
                nn.init.xavier_normal_(m.weight.data)
                # nn.init.normal_(m.weight.data)
                # m.bias.data.fill_(0.1)

    def forward(self, hete_graph, idx, x):
        # com_emb=torch.zeros((self.company_num,self.embed_d))
        # for index in risk_data:
        #     cause=self.ca_emb[risk_data[index][:,0]]
        #     court=self.court_emb[risk_data[index][:,1]]
        #     cate=self.cate_emb[risk_data[index][:2]]
        #     com_attr=company_attr
        #     risk=torch.cat((cause,court,cate,com_attr[index]),dim=0)

        #     time_label=risk_data[index][:,4]

        #     risk=scatter(risk, time_label, 0, dim_size=self.time_lable_num, reduce='sum').view(self.time_lable_num,1,self.input_dim)
        #     all_state, last_state=self.c_content_rnn(risk)
        #     com_emb[index]=torch.mean(all_state, 0)
        person_emb = self.person_emb(torch.LongTensor([i for i in range(self.person_num)]))
        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        com_attribute = torch.Tensor(x)
        com_attr_total = torch.zeros(self.company_num, com_attribute.shape[1])
        com_attr_total[idx] = com_attribute
        com_emb = self.proj_c(torch.cat((company_emb, com_attr_total), dim=1))
        # com_emb=self.proj_c(com_emb)

        emb = torch.cat((com_emb, person_emb), dim=0)
        rs = torch.zeros_like(emb)
        edge_index, edge_type, edge_weight = hete_graph
        node_type = torch.cat((torch.ones(self.company_num) * 0, torch.ones(self.person_num)), dim=0)
        edge_index = torch.LongTensor(edge_index).transpose(0, 1)
        edge_type = torch.LongTensor(edge_type)
        res = []
        for rel in range(self.rel_num):
            if rel in [0, 1, 2, 3, 4, 5, 8, 9]:
                index = (edge_type == rel)
                sub_edge_index = edge_index[:, index]
                msg = self.propagate(sub_edge_index, node_inp=emb, edge_type=0)
                res += [msg]
            # TODO: change for directed edge
            # c-c relations
            elif rel in [6, 7, 10, 11]:
                index = (edge_type == rel)
                sub_edge_index = edge_index[:, index]
                msg = self.propagate(sub_edge_index, node_inp=emb, edge_type=1)
                res += [msg]
            else:
                index = (edge_type == rel)
                sub_edge_index = edge_index[:, index]
                msg = self.propagate(sub_edge_index, node_inp=emb, edge_type=1)
                res += [msg]

        conc = self.out_proj(torch.cat(res, dim=1))  # (N,r*d)
        for i in range(2):
            index = (node_type == i)
            # nd_feat=node_feat[idx]
            if i == 0:
                temp = self.act(conc[index] @ self.c_neigh_att)
            if i == 1:
                temp = self.act(conc[index] @ self.p_neigh_att)
            att = self.softmax(temp)
            # rs[idx]+=torch.bmm(att, conc[idx])
            rs[index] = att * conc[index]

        return rs[idx]

    def message(self, node_inp_j, edge_type):
        edge_num = node_inp_j.shape[0]
        node_inp_j = node_inp_j.view(1, edge_num, self.output_dim)
        all_state, last_state = self.neigh_rnn[edge_type](node_inp_j)
        return all_state.squeeze(0)
