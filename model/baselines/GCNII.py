import torch
import torch.nn as nn
from torch_geometric.nn import GCN2Conv


class GCNII(nn.Module):
    def __init__(self, input_dim, output_dim, company_num, person_num, n_layer=1):
        super(GCNII, self).__init__()
        self.company_num = company_num
        self.person_num = person_num

        self.att_dim = 32
        self.proj = nn.Linear(input_dim + self.att_dim, input_dim)
        self.final_proj = nn.Linear(input_dim, output_dim)

        self.n_layer = n_layer
        self.company_emb = nn.Embedding(company_num, input_dim)
        self.person_emb = nn.Embedding(person_num, input_dim)
        self.gcn_layers = nn.ModuleList()
        for i in range(n_layer):
            self.gcn_layers.append(GCN2Conv(input_dim, 5e-4))

    def forward(self, hete_graph, idx, x):
        edge_index, edge_type, edge_weight = hete_graph
        edge_index = torch.LongTensor(edge_index).transpose(0, 1)
        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        person_emb = self.person_emb(torch.LongTensor([i for i in range(self.person_num)]))
        company_attr_cal = torch.zeros((self.company_num, self.att_dim))
        company_attr_cal[idx] = torch.FloatTensor(x)
        company_emb = self.proj(torch.cat((company_emb, company_attr_cal), dim=1))
        emb = torch.cat((company_emb, person_emb), dim=0)
        for i in range(self.n_layer):
            emb = self.gcn_layers[i](emb, emb, edge_index)
        emb = self.final_proj(emb)
        return emb[idx]
