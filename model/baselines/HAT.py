import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class FeatureAttention(nn.Module):

    def __init__(self, in_size, hidden_size=16):
        super(FeatureAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)  # (13489, 5, 1)
        beta = torch.softmax(w, dim=1)  # (13489, 5, 1)

        return (beta * z).sum(1)  # (N, D * K)


# metapath-level attention and network level attention
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

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []
        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](h, g).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)
        # print(semantic_embeddings.shape,self.num_meta_paths)
        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


# HAN uses a dual-level attention
class HANConv(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, num_heads=[1], dropout=0.5):
        super(HANConv, self).__init__()
        self.i_dim = 42
        # self.FeatureAttention = FeatureAttention(self.i_dim)
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0]))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l - 1],
                                        hidden_size, num_heads[l], dropout))

    def forward(self, g, h):
        # c_i_embedding = self.FeatureAttention(c_ineigh_feature)
        # h = torch.cat((h,c_i_embedding),1)
        for gnn in self.layers:
            h = gnn(g, h)

        return h


# HAT uses a triple level attention
class HATConv(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads=[1], dropout=0.5):
        super(HATConv, self).__init__()
        # self.network_emebdding_indim = 144  #network_embedding's dimension generated from HAN
        # self.network_embedding_outdim = 64
        self.project = nn.Linear(hidden_size, hidden_size)
        self.network_attention = SemanticAttention(in_size=hidden_size)
        self.num_meta_paths = num_meta_paths
        self.layers = nn.ModuleList()
        for i in range(len(self.num_meta_paths)):
            self.layers.append(HANConv(self.num_meta_paths[i], in_size, hidden_size, num_heads, dropout))
        # self.layers = HANConv(num_meta_paths, in_size, hidden_size, num_heads, dropout)
        self.predict = nn.Linear(hidden_size, out_size)

    def forward(self, g, h, num_of_network):
        network_embedding = []
        for i in range(num_of_network):
            network_embedding.append(self.layers[i](g[i], h))

        for i in range(len(network_embedding)):
            network_embedding[i] = self.project(network_embedding[i])
        network_embedding = torch.stack(network_embedding, dim=1)

        global_embedding = self.network_attention(network_embedding)

        return self.predict(global_embedding)


class HAT(nn.Module):
    def __init__(self, input_dim, output_dim, company_num, person_num, rel_num, device):
        super(HAT, self).__init__()
        self.company_num = company_num
        self.person_num = person_num
        self.device = device
        # TODO: change for directed edge
        self.num_meta_paths = [3, 2, 2]
        # self.num_meta_paths=[3,4,8]
        self.num_rel = rel_num
        self.company_emb = nn.Embedding(company_num, input_dim)
        self.person_emb = nn.Embedding(person_num, input_dim)
        self.FeatureAttention = FeatureAttention(input_dim)
        self.att_dim = 32
        self.attr_proj = nn.Linear(input_dim + self.att_dim, input_dim)
        self.hat_layer = HATConv(self.num_meta_paths, input_dim, output_dim, output_dim)

    def forward(self, hete_graph, idx, x):
        person_emb = self.person_emb(torch.LongTensor([i for i in range(self.person_num)]))
        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        edge_index, edge_type, edge_weight = hete_graph
        edge_type = torch.LongTensor(edge_type)
        edge_index = torch.LongTensor(edge_index).transpose(0, 1)
        occupation_network = []
        invest_network = []
        justfy_network = []
        for i in range(self.num_rel):
            # TODO: change for directed edge
            if i in [0, 1, 2]:
                # if i in [0,1,2]:
                index = (edge_type == i)
                sub_edge_index = edge_index[:, index]
                occupation_network += [sub_edge_index]
            # TODO: change for directed edge
            elif i in [3, 4]:
                # elif i in [3,4,9,10]:
                index = (edge_type == i)
                sub_edge_index = edge_index[:, index]
                invest_network += [sub_edge_index]
            # TODO: change for directed edge
            # elif i in [5,6,7,8]:
            elif i in [5, 6]:
                # elif i in [5,6,7,8,11,12,13,14]:
                index = (edge_type == i)
                sub_edge_index = edge_index[:, index]
                justfy_network += [sub_edge_index]
        # occupation_network=torch.cat(occupation_network,dim=1)
        # invest_network=torch.cat(invest_network,dim=1)
        network = [occupation_network, invest_network, justfy_network]
        # company_attr_emb=initializae_company_info(risk_data,company_attr,company_num=None,cause_type_num=None,court_type=None,category=None)
        company_attr_cal = torch.zeros((self.company_num, self.att_dim))
        company_attr_cal[idx] = torch.FloatTensor(x)
        company_emb = self.attr_proj(torch.cat((company_emb, company_attr_cal), dim=1))
        emb = torch.cat((company_emb, person_emb), dim=0)
        emb = self.hat_layer(network, emb, 3)

        return emb[idx].to(self.device)
