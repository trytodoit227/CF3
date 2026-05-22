from matplotlib.pyplot import delaxes
import torch
import random
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
import time
from sklearn import preprocessing
import scipy.sparse as sp


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


class Classifier(nn.Module):
    def __init__(self, n_hid, n_out):
        super(Classifier, self).__init__()
        self.n_hid = n_hid
        self.n_out = n_out
        self.linear = nn.Linear(n_hid, n_out)

    def forward(self, x):
        tx = self.linear(x)
        return torch.log_softmax(tx.squeeze(), dim=-1)

    def __repr__(self):
        return '{}(n_hid={}, n_out={})'.format(
            self.__class__.__name__, self.n_hid, self.n_out)


def generate_G_from_H(H):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    invDV = np.mat(np.diag(np.power(DV + 1e-6, -1)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T
    G = invDV * H * W * invDE * HT

    H_dict = dict()
    H_dict['H'] = H
    H_dict['H_T'] = HT
    H_dict['D_e_neg_1'] = invDE
    H_dict['D_v_neg_1'] = invDV
    H_dict['W_e'] = W

    return H_dict, G


# refer to https://github.com/iMoonLab/THU-HyperG/blob/master/hyperg/hyperg.py
class HyperG:
    def __init__(self, H, X=None, w=None):
        """ Initial the incident matrix, node feature matrix and hyperedge weight vector of hypergraph
        :param H: scipy coo_matrix of shape (n_nodes, n_edges)
        :param X: numpy array of shape (n_nodes, n_features)
        :param w: numpy array of shape (n_edges,)
        """
        assert sparse.issparse(H)
        assert H.ndim == 2

        self._H = H
        self._n_nodes = self._H.shape[0]
        self._n_edges = self._H.shape[1]

        if X is not None:
            assert isinstance(X, np.ndarray) and X.ndim == 2
            self._X = X
        else:
            self._X = None

        if w is not None:
            self.w = w.reshape(-1)
            assert self.w.shape[0] == self._n_edges
        else:
            self.w = np.ones(self._n_edges)

        self._DE = None
        self._DV = None
        self._INVDE = None
        self._DV2 = None
        self._THETA = None
        self._L = None

    def num_edges(self):
        return self._n_edges

    def num_nodes(self):
        return self._n_nodes

    def incident_matrix(self):
        return self._H

    def hyperedge_weights(self):
        return self.w

    def node_features(self):
        return self._X

    def node_degrees(self):
        if self._DV is None:
            H = self._H.tocsr()
            dv = H.dot(self.w.reshape(-1, 1)).reshape(-1)
            self._DV = sparse.diags(dv, shape=(self._n_nodes, self._n_nodes))
        return self._DV

    def edge_degrees(self):
        if self._DE is None:
            H = self._H.tocsr()
            de = H.sum(axis=0).A.reshape(-1)
            self._DE = sparse.diags(de, shape=(self._n_edges, self._n_edges))
        return self._DE

    def inv_edge_degrees(self):
        if self._INVDE is None:
            self.edge_degrees()
            inv_de = np.power(self._DE.data.reshape(-1), -1.)
            self._INVDE = sparse.diags(inv_de, shape=(self._n_edges, self._n_edges))
        return self._INVDE

    def inv_square_node_degrees(self):
        if self._DV2 is None:
            self.node_degrees()
            dv2 = np.power(self._DV.data.reshape(-1) + 1e-6, -0.5)
            self._DV2 = sparse.diags(dv2, shape=(self._n_nodes, self._n_nodes))
        return self._DV2

    def theta_matrix(self):
        if self._THETA is None:
            self.inv_square_node_degrees()
            self.inv_edge_degrees()

            W = sparse.diags(self.w)
            self._THETA = self._DV2.dot(self._H).dot(W).dot(self._INVDE).dot(self._H.T).dot(self._DV2)

        return self._THETA

    def laplacian(self):
        if self._L is None:
            self.theta_matrix()
            self._L = sparse.eye(self._n_nodes) - self._THETA
        return self._L

    def update_hyedge_weights(self, w):
        assert isinstance(w, (np.ndarray, list)), \
            "The hyperedge array should be a numpy.ndarray or list"

        self.w = np.array(w).reshape(-1)
        assert w.shape[0] == self._n_edges

        self._DV = None
        self._DV2 = None
        self._THETA = None
        self._L = None

    def update_incident_matrix(self, H):
        assert sparse.issparse(H)
        assert H.ndim == 2
        assert H.shape[0] == self._n_nodes
        assert H.shape[1] == self._n_edges

        # TODO: reset hyperedge weights?

        self._H = H
        self._DE = None
        self._DV = None
        self._INVDE = None
        self._DV2 = None
        self._THETA = None
        self._L = None


def v2v(x, aggr='mean', drop_rate=0.0, **kwargs):
    x = v2e(x, aggr, None, None, drop_rate=drop_rate, **kwargs)
    x = e2v(x, aggr, None, drop_rate=drop_rate, **kwargs)
    return x


def v2e(x, aggr='mean', v2e_weight=None, e_weight=None, drop_rate=0.0, **kwargs):
    H_dict = kwargs.get('H_dict')
    H_T = torch.FloatTensor(H_dict['H_T'])
    D_e_neg_1 = torch.FloatTensor(H_dict['D_e_neg_1'])
    W_e = torch.FloatTensor(H_dict['W_e'])

    # v2e_aggregation
    if v2e_weight is None:
        if drop_rate > 0.0:
            P = sparse_dropout(H_T, drop_rate)
        else:
            P = H_T
        if aggr == "mean":
            x = torch.sparse.mm(P, x)
            x = torch.sparse.mm(D_e_neg_1, x)
        elif aggr == "sum":
            x = torch.sparse.mm(P, x)
        elif aggr == "softmax_then_sum":
            P = torch.sparse.softmax(P, dim=1)
            x = torch.sparse.mm(P, x)
    else:
        P = torch.sparse_coo_tensor(H_T._indices(), v2e_weight, H_T.shape)
        if drop_rate > 0.0:
            P = sparse_dropout(P, drop_rate)
        # message passing
        if aggr == "mean":
            x = torch.sparse.mm(P, x)
            D_e_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
            D_e_neg_1[torch.isinf(D_e_neg_1)] = 0
            x = D_e_neg_1 * x
        elif aggr == "sum":
            x = torch.sparse.mm(P, x)
        elif aggr == "softmax_then_sum":
            P = torch.sparse.softmax(P, dim=1)
            x = torch.sparse.mm(P, x)

    # v2e_update
    if e_weight is None:
        x = torch.sparse.mm(W_e, x)
    else:
        e_weight = e_weight.view(-1, 1)
        x = e_weight * x

    return x


def e2v(x, aggr='mean', e2v_weight=None, drop_rate=0.0, **kwargs):
    H_dict = kwargs.get('H_dict')
    H = torch.FloatTensor(H_dict['H'])
    D_v_neg_1 = torch.FloatTensor(H_dict['D_v_neg_1'])

    # e2v_aggregation
    if e2v_weight is None:
        if drop_rate > 0.0:
            P = sparse_dropout(H, drop_rate)
        else:
            P = H
        if aggr == "mean":
            x = torch.sparse.mm(P, x)
            x = torch.sparse.mm(D_v_neg_1, x)
        elif aggr == "sum":
            x = torch.sparse.mm(P, x)
        elif aggr == "softmax_then_sum":
            P = torch.sparse.softmax(P, dim=1)
            x = torch.sparse.mm(P, x)
    else:
        P = torch.sparse_coo_tensor(H._indices(), e2v_weight, H.shape)
        if drop_rate > 0.0:
            P = sparse_dropout(P, drop_rate)
        # message passing
        if aggr == "mean":
            x = torch.sparse.mm(P, x)
            D_v_neg_1 = torch.sparse.sum(P, dim=1).to_dense().view(-1, 1)
            D_v_neg_1[torch.isinf(D_v_neg_1)] = 0
            x = D_v_neg_1 * x
        elif aggr == "sum":
            x = torch.sparse.mm(P, x)
        elif aggr == "softmax_then_sum":
            P = torch.sparse.softmax(P, dim=1)
            x = torch.sparse.mm(P, x)

    # e2v_update

    return x


def sparse_dropout(sp_mat, p, fill_value=0.0):
    r"""Dropout function for sparse matrix. This function will return a new sparse matrix with the same shape as the input sparse matrix, but with some elements dropped out.

    Args:
        ``sp_mat`` (``torch.Tensor``): The sparse matrix with format ``torch.sparse_coo_tensor``.
        ``p`` (``float``): Probability of an element to be dropped.
        ``fill_value`` (``float``): The fill value for dropped elements. Defaults to ``0.0``.
    """
    device = sp_mat.device
    sp_mat = sp_mat.coalesce()
    assert 0 <= p <= 1
    if p == 0:
        return sp_mat
    p = torch.ones(sp_mat._nnz(), device=device) * p
    keep_mask = torch.bernoulli(1 - p).to(device)
    fill_values = torch.logical_not(keep_mask) * fill_value
    new_sp_mat = torch.sparse_coo_tensor(
        sp_mat._indices(),
        sp_mat._values() * keep_mask + fill_values,
        size=sp_mat.size(),
        device=sp_mat.device,
        dtype=sp_mat.dtype,
    )
    return new_sp_mat


def print_log(message):
    """
    :param message: str,
    :return:
    """
    print("[{}] {}".format(time.strftime("%Y-%m-%d %X", time.localtime()), message))


import numpy as np
import scipy.sparse as sparse


def gen_attribute_hg(n_nodes, attr_dict, X=None):
    """
    :param attr_dict: dict, eg. {'attri_1': [node_idx_1, node_idx_1, ...], 'attri_2':[...]} (zero-based indexing)
    :param n_nodes: int,
    :param X: numpy array, shape = (n_samples, n_features) (optional)
    :return: instance of HyperG
    """

    if X is not None:
        assert n_nodes == X.shape[0]

    n_edges = len(attr_dict)
    node_idx = []
    edge_idx = []

    for idx, attr in enumerate(attr_dict):
        nodes = sorted(attr_dict[attr])
        node_idx.extend(nodes)
        edge_idx.extend([idx] * len(nodes))

    node_idx = np.asarray(node_idx)
    edge_idx = np.asarray(edge_idx)
    values = np.ones(node_idx.shape[0])

    H = sparse.coo_matrix((values, (node_idx, edge_idx)), shape=(n_nodes, n_edges))
    return HyperG(H, X=X)


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    """
    将scipy的sparse matrix转换成torch的sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# refer to https://github.com/alge24/DyGNN/blob/b161555a5df69bd3fa9cc3ae5d4f5cd65ebe3a0f/decayer.py
class Decayer(nn.Module):
    def __init__(self, w1=0.01, w2=0.1, decay_method='rev'):
        # def __init__(self, w1=100,w2=200, decay_method='rev'):
        super(Decayer, self).__init__()
        self.decay_method = decay_method
        self.w1 = w1
        self.w2 = w2

    def exponetial_decay(self, w, delta_t):
        return torch.exp(-w * delta_t)

    def log_decay(self, w, delta_t):
        return 1 / torch.log(2.7183 + w * delta_t)

    def rev_decay(self, w, delta_t):
        return 1 / (1 + w * delta_t)

    def forward(self, delta_t):
        seq = torch.zeros_like(delta_t)

        idx1 = (delta_t <= 24)
        idx2 = (delta_t > 24)

        # # print(delta_t)
        if self.decay_method == 'exp':
            seq[idx1] = self.exponetial_decay(self.w1, delta_t[idx1])
            seq[idx2] = self.exponetial_decay(self.w2, delta_t[idx2])
        elif self.decay_method == 'result_log':
            seq[idx1] = self.log_decay(self.w1, delta_t[idx1])
            seq[idx2] = self.log_decay(self.w2, delta_t[idx2])
        elif self.decay_method == 'rev':
            seq[idx1] = self.rev_decay(self.w1, delta_t[idx1])
            seq[idx2] = self.rev_decay(self.w2, delta_t[idx2])

        else:
            seq[idx1] = self.exponetial_decay(delta_t[idx1])
            seq[idx2] = self.exponetial_decay(delta_t[idx2])
        # print(seq,"----")

        return seq


def initialize_company_info(sxjl_data, zdgz_data, xzxk_data, risk_data, company_attr, company_num, cause_type_num,
                            court_type, category, idx=None):
    if idx:
        idx_dict = {index: ser for ser, index in enumerate(idx)}
    len_sxjl = 3
    len_xzxk = 4
    company_risk = np.zeros((company_num, cause_type_num + court_type + category + 1))
    company_sxjl = np.zeros((company_num, len_sxjl))
    company_zdgz = np.zeros((company_num, 1))
    # company_xzcf = np.zeros((company_num, 2 + 1 + 1))
    company_xzxk = np.zeros((company_num, len_xzxk + 1))
    s = preprocessing.RobustScaler()
    for index in risk_data:
        risk_info = risk_data[index]  # 获得某个公司的对应信息
        cause_info = [0 for i in range(cause_type_num)]  # 11种案由类别
        court_info = [0 for i in range(court_type)]  # 4种法院类别
        res_info = [0 for i in range(category)]  # 4种类别，原告or被告，赢or输
        time_info = []
        for i in range(len(risk_info)):
            justify = risk_info[i]
            cause = justify[0]
            court = justify[1]
            res = justify[2]
            time = justify[3]
            cause_info[cause] += 1
            court_info[court] += 1
            res_info[res] += 1
            time_info += [time]
        time_ave = [np.average(time_info)]
        if idx:
            company_risk[idx_dict[index]] = np.concatenate((cause_info, court_info, res_info, time_ave), axis=0)
        else:
            company_risk[index] = np.concatenate((cause_info, court_info, res_info, time_ave), axis=0)

    for index in sxjl_data:
        sxjl_info = sxjl_data[index]
        pjnd_info = [0 for i in range(len_sxjl)]
        for i in sxjl_info:
            pjnd_info[i] += 1
        if idx:
            company_sxjl[idx_dict[index]] = np.concatenate((pjnd_info,), axis=0)
        else:
            company_sxjl[index] = np.concatenate((pjnd_info,), axis=0)

    # for index in xzcf_data:
    #     xzcf_info = zdgz_data[index]
    #     type_info = [0 for i in range(2)]  # 类型
    #     amount_info = []
    #     time_info = []
    #     for i in range(len(xzcf_info)):
    #         one_info = xzcf_info[i]
    #         type = one_info[0]
    #         time = one_info[1]
    #         amount = one_info[2]
    #         type_info[type] += 1
    #         amount_info += [amount]
    #         time_info += [time]
    #     time_ave = [np.average(time_info)]
    #     amount_ave = [np.average(amount_info)]
    #     if idx:
    #         company_xzcf[idx_dict[index]] = np.concatenate((type_info, amount_ave, time_ave), axis=0)
    #     else:
    #         company_xzcf[index] = np.concatenate((type_info, amount_ave, time_ave), axis=0)

    for index in xzxk_data:
        xzxk_info = xzxk_data[index]
        type_info = [0 for i in range(len_xzxk)]  # 类型
        time_info = []
        for i in range(len(xzxk_info)):
            one_info = xzxk_info[i]
            type = one_info[0]
            time = one_info[1]
            type_info[type] += 1
            time_info += [time]
        time_ave = [np.average(time_info)]
        if idx:
            company_xzxk[idx_dict[index]] = np.concatenate((type_info, time_ave), axis=0)
        else:
            company_xzxk[index] = np.concatenate((type_info, time_ave), axis=0)

    # add time label
    # for index in zdgz_data:
    #     zdgz_info = zdgz_data[index]
    #     time_info = []
    #     for i in range(len(zdgz_info)):
    #         one_info = zdgz_info[i]
    #         time = one_info[0]
    #         time_info += [time]
    #     time_ave = [np.average(time_info)]
    #     if idx:
    #         company_zdgz[idx_dict[index]] = np.concatenate((time_ave, ), axis=0)
    #     else:
    #         company_zdgz[index] = np.concatenate((time_ave,), axis=0)

    # no time label
    for index in zdgz_data:
        zdgz_info = zdgz_data[index]
        time_info = []
        for i in zdgz_info:
            time_info += [i]
        time_ave = [np.average(time_info)]
        if idx:
            company_zdgz[idx_dict[index]] = np.concatenate((time_ave,), axis=0)
        else:
            company_zdgz[index] = np.concatenate((time_ave,), axis=0)

    company_attr = np.array(company_attr)

    company_info = np.concatenate((company_attr, company_sxjl, company_xzxk, company_zdgz, company_risk), axis=1)
    return company_info


def get_edge_index(hete_graph):
    ei, _, _ = hete_graph
    edge_index = torch.LongTensor(ei).transpose(0, 1)
    return edge_index


def get_adj(hete_graph, num):
    ei, _, _ = hete_graph
    sedges = np.array(list(np.array(ei)), dtype=np.int32).reshape(np.array(ei).shape)
    sadj = sp.coo_matrix((np.ones(sedges.shape[0]), (sedges[:, 0], sedges[:, 1])), shape=(num, num),
                         dtype=np.float32)
    sadj = sadj + sadj.T.multiply(sadj.T > sadj) - sadj.multiply(sadj.T > sadj)
    sadj = sadj + sp.eye(sadj.shape[0])
    sadj = torch.tensor(sadj.todense(), dtype=torch.float32)
    return sadj


def one_hot_embedding(num_classes, soft):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    soft = torch.argmax(soft.exp(), dim=1)
    y = torch.eye(num_classes)
    return y[soft]


def get_sub_info(node_idx, edge_index, hard_edge_mask, hete_graph, hyper_graph, test_idx_dic, test_company_num,
                 hete_gprah_flag=True):
    total_company_num = 3976
    # sub_idx
    idx = list(np.sort(np.unique(edge_index)))  # list(set(edge_index[0].tolist()))
    sub_idx = [i for i in idx if i < total_company_num and test_idx_dic[i] < test_company_num]
    if node_idx not in sub_idx:
        sub_idx.append(node_idx)
    # sub_hete_graph
    if hete_gprah_flag:
        sub_edge_index = edge_index.transpose(0, 1).tolist()
        sub_edge_type = [hete_graph[1][i.item()] for i in list(hard_edge_mask.nonzero())]
        sub_edge_weight = [hete_graph[2][i.item()] for i in list(hard_edge_mask.nonzero())]
        sub_hete_graph = [sub_edge_index, sub_edge_type, sub_edge_weight]
    else:
        sub_hete_graph = None
    # sub_hyp_graph
    hyp_graph = get_sub_hyp_graph(idx, hyper_graph, total_company_num)

    return sub_idx, sub_hete_graph, hyp_graph


def get_sub_hyp_graph(idx, hyper_graph, total_company_num):
    sub_hyp_graph = {'industry': {}, 'area': {}, 'qualify': {}}
    for k, v in hyper_graph.items():
        for i in hyper_graph[k]:
            for j in idx:
                if j in hyper_graph[k][i]:
                    if sub_hyp_graph[k].get(i) is None:
                        sub_hyp_graph[k][i] = [j]
                    else:
                        sub_hyp_graph[k][i].append(j)
    hyp_graph = []
    for i in ['industry', 'area', 'qualify']:
        hyp_graph += [gen_attribute_hg(total_company_num, sub_hyp_graph[i], X=None)]

    return hyp_graph


def convert_coo_to_tensor(adj):
    values = adj.data
    indices = np.vstack((adj.row, adj.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = adj.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape)).to_dense()


def from_adj_to_edge_index_torch(adj):
    adj_sparse = adj.to_sparse()
    edge_index = adj_sparse.indices().to(dtype=torch.long)
    edge_attr = adj_sparse.values()
    # if adj.requires_grad:
    # edge_index.requires_grad = True
    # edge_attr.requires_grad = True
    return edge_index, edge_attr


def get_new_edge_type_from_new_edge_index(new_edge_index, new_edge_weight, edge_type, edge_index):
    indices = []
    for i in range(edge_index.size(1)):
        idx = (edge_index[0][i], edge_index[1][i])
        for j in range(len(new_edge_index[0])):
            if (new_edge_index[0][j], new_edge_index[1][j]) == idx:
                indices.append(j)
    new_edge_type = torch.zeros(new_edge_weight.shape).fill_(6)
    edge_type_dict = dict(zip(indices, edge_type.numpy()))
    for k, v in edge_type_dict.items():
        new_edge_type[k] = v

    return new_edge_type


def from_edge_index_to_adj_torch(edge_index, edge_weight, max_n):
    adj = torch.sparse_coo_tensor(
        edge_index,
        edge_weight,
        size=(max_n, max_n),
        dtype=torch.float32,
        device=edge_index.device,
    )
    if edge_index.requires_grad:
        adj.requires_grad = True
    return adj.to_dense()


def get_degree_matrix(adj):
    return torch.diag(sum(adj))


def create_symm_matrix_from_vec(vector, n_rows):
    matrix = torch.zeros(n_rows, n_rows)
    idx = torch.tril_indices(n_rows, n_rows)
    matrix[idx[0], idx[1]] = vector
    symm_matrix = torch.tril(matrix) + torch.tril(matrix, -1).t()
    return symm_matrix


def create_vec_from_symm_matrix(matrix, P_vec_size):
    idx = torch.tril_indices(matrix.shape[0], matrix.shape[0])
    vector = matrix[idx[0], idx[1]]
    return vector
