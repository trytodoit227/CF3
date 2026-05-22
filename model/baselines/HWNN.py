import torch
import torch.nn as nn
import torch.nn.functional as F


class HWNN(nn.Module):
    def __init__(self, input_dim, output_dim, company_num, hyp_type=3, n_layer=3, dropout=0.5):
        super(HWNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.company_num = company_num
        self.company_emb = nn.Embedding(company_num, input_dim)
        self.hwnn = nn.ModuleList()
        for i in range(n_layer):
            if i == 0:
                self.hwnn.append(HWNNLayer(input_dim, input_dim, company_num))
            else:
                self.hwnn.append(HWNNLayer(input_dim, output_dim, company_num))
        self.par = torch.nn.Parameter(torch.Tensor(hyp_type))
        self.init_parameters()
        self.s = 1.0

        self.att_dim = 32
        self.proj = nn.Linear(input_dim + self.att_dim, input_dim)

    def init_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight_matrix)
        # torch.nn.init.uniform_(self.diagonal_weight_filter, 0.99, 1.01)
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def forward(self, hyp_graph, idx, x):
        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        company_attr_cal = torch.zeros((self.company_num, self.att_dim))
        company_attr_cal[idx] = torch.FloatTensor(x)
        company_emb = self.proj(torch.cat((company_emb, company_attr_cal), dim=1))
        feat_list = []
        #  hyp_graph is a list consist of sparse.coo_matrix
        for i in range(len(hyp_graph)):
            g = hyp_graph[i]
            Laplacian = torch.Tensor(g.laplacian().toarray())
            fourier_e, fourier_v = torch.symeig(Laplacian, eigenvectors=True)
            wavelets = fourier_v @ torch.diag(torch.exp(-1.0 * fourier_e * self.s)) @ torch.transpose(fourier_v, 0, 1)
            wavelets_inv = fourier_v @ torch.diag(torch.exp(fourier_e * self.s)) @ torch.transpose(fourier_v, 0, 1)
            # wavelets_t = torch.transpose(wavelets, 0, 1)
            wavelets[wavelets < 0.00001] = 0
            wavelets_inv[wavelets_inv < 0.00001] = 0
            # wavelets_t[wavelets_t < 0.00001] = 0
            localized_features = self.hwnn[0](company_emb, wavelets, wavelets_inv)
            localized_features_1 = F.dropout(F.relu(localized_features), self.dropout)
            localized_features_2 = self.hwnn[1](localized_features_1, wavelets, wavelets_inv)
            feat_list += [localized_features_2]

        feat = torch.zeros_like(feat_list[0])
        for i in range(len(feat_list)):
            feat += self.par[i] * feat_list[i]
        return feat[idx]


class HWNNLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, ncount, K1=2, K2=2, data=None):
        super(HWNNLayer, self).__init__()
        self.data = data
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ncount = ncount
        self.K1 = K1
        self.K2 = K2
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.diagonal_weight_filter = torch.nn.Parameter(torch.Tensor(self.ncount))
        self.par = torch.nn.Parameter(torch.Tensor(self.K1 + self.K2))
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.uniform_(self.diagonal_weight_filter, 0.99, 1.01)
        torch.nn.init.uniform_(self.par, 0, 0.99)

    def forward(self, features, wavelets, wavelets_inverse):
        diagonal_weight_filter = torch.diag(self.diagonal_weight_filter)
        features = features
        local_fea_1 = wavelets @ diagonal_weight_filter @ wavelets_inverse @ features @ self.weight_matrix

        localized_features = local_fea_1
        return localized_features
