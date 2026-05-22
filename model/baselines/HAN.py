import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class HAN(nn.Module):
    def __init__(self, input_dim, output_dim, company_num, person_num, rel_num, num_heads=1, dropout=0.2):
        super(HAN, self).__init__()
        self.company_num = company_num
        self.person_num = person_num
        self.num_rel = rel_num
        self.company_emb = nn.Embedding(company_num, input_dim)
        self.person_emb = nn.Embedding(person_num, input_dim)
        self.han_layers = nn.ModuleList()
        self.att_dim = 32
        self.proj = nn.Linear(input_dim + self.att_dim, input_dim, bias=False)
        for i in range(rel_num):
            self.han_layers.append(GATConv(input_dim, output_dim, num_heads, concat=False, dropout=dropout))
        self.semantic_attention = SemanticAttention(in_size=output_dim)

    def forward(self, hete_graph, idx, x):
        edge_index, edge_type, edge_weight = hete_graph
        edge_type = torch.LongTensor(edge_type)
        edge_index = torch.LongTensor(edge_index).transpose(0, 1)
        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        person_emb = self.person_emb(torch.LongTensor([i for i in range(self.person_num)]))
        company_attr_cal = torch.zeros((self.company_num, self.att_dim))
        company_attr_cal[idx] = torch.FloatTensor(x)
        company_emb = self.proj(torch.cat((company_emb, company_attr_cal), dim=1))
        emb = torch.cat((company_emb, person_emb), dim=0)
        semantic_embeddings = []

        for i in range(self.num_rel):
            index = (edge_type == i)
            sub_edge_index = edge_index[:, index]
            semantic_embeddings.append(self.han_layers[i](emb, sub_edge_index).flatten(1))  # (N,D)
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D )

        return self.semantic_attention(semantic_embeddings)[idx]  # (N, D )


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D )
