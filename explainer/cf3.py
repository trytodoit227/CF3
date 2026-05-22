"""
除了对异质边信息的处理，
还加入了对节点特征的扰动处理、对边扰动的处理
以及正态分布初始化特征矩阵、特征矩阵对称处理
"""
from math import sqrt
import math
import copy
from typing import Optional
from inspect import signature
from collections import defaultdict

import torch
from torch.nn.parameter import Parameter
from tqdm import tqdm
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import networkx as nx
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

from utils.model_utils import get_sub_info, gen_attribute_hg, get_sub_hyp_graph

EPS = 1e-15


class CF3_GCN(nn.Module):
    def __init__(self, input_dim, output_dim, n_layer, company_num=3976, person_num=2405):
        super(CF3_GCN, self).__init__()
        self.company_num = company_num
        self.person_num = person_num

        self.att_dim = 32
        self.proj = nn.Linear(input_dim + self.att_dim, input_dim)

        self.n_layer = n_layer
        self.company_emb = nn.Embedding(company_num, input_dim)
        self.person_emb = nn.Embedding(person_num, input_dim)
        self.gcn_layers = nn.ModuleList()
        for i in range(n_layer):
            self.gcn_layers.append(GCNConv(input_dim, output_dim))

    def forward(self, hete_graph, idx, x_binary1, x_binary2):
        edge_index, edge_type, edge_weight = hete_graph
        edge_index = torch.LongTensor(edge_index).transpose(0, 1)

        x = x_binary1
        for i in range(self.n_layer):

            company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
            person_emb = self.person_emb(torch.LongTensor([i for i in range(self.person_num)]))
            company_attr_cal = torch.zeros((self.company_num, self.att_dim))
            company_attr_cal[idx] = torch.FloatTensor(x)
            company_emb = self.proj(torch.cat((company_emb, company_attr_cal), dim=1))
            emb = torch.cat((company_emb, person_emb), dim=0)

            emb = self.gcn_layers[i](emb, edge_index)

            if i != self.n_layer - 1:
                x_binary2 = x_binary2 * emb[:len(idx)]
                x_binary2 = (x_binary2 >= 0.5).float()
                x = x_binary2

        return emb


class CF3(torch.nn.Module):
    coeffs = {
        'edge_size': 0.005,  # 0.005
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,  # 1.0
        'node_feat_ent': 0.1,
        'weight_loss': 0.1
    }

    def __init__(self, model, num, temp, x_shape, device, feat_mask_type: str = "feature", epochs: int = 100, lr: float = 0.01,
                 num_hops: Optional[int] = None, log: bool = True):
        super(CF3, self).__init__()
        assert feat_mask_type in ["feature", "individual_feature"]
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.num_hops_sub = num_hops - 1
        self.log = log
        self.num = num
        self.feat_mask_type = feat_mask_type
        self.node = True
        self.temp = temp
        self.device = device

        self.mlp1 = nn.Linear(x_shape, x_shape)
        self.mlp2 = nn.Linear(x_shape, x_shape)

        self.gcn1 = CF3_GCN(input_dim=16, output_dim=32, n_layer=2)
        self.gcn2 = CF3_GCN(input_dim=16, output_dim=32, n_layer=2)
        self.gcn3 = CF3_GCN(input_dim=16, output_dim=32, n_layer=2)
        self.gcn4 = CF3_GCN(input_dim=16, output_dim=32, n_layer=2)

        self.weight_loss = torch.nn.Parameter(torch.rand(1)).to(self.device)
        self.weight_loss1 = torch.nn.Parameter(torch.rand(1)).to(self.device)
        self.weight_loss2 = torch.nn.Parameter(torch.rand(1)).to(self.device)
        self.weight_loss3 = torch.nn.Parameter(torch.rand(1)).to(self.device)

        self.weight_decay = 0.001

    def __set_masks__(self, x, edge_index, edge_masks=None, edge_mask=None):
        (N, F), E = x.size(), edge_index.size(1)  # N：子图节点数量；F：子图特征维度；E：子图边的数量

        if edge_mask is not None:
            for module in self.model.modules():
                if isinstance(module, MessagePassing):
                    module.__explain__ = True
                    module.__edge_mask__ = edge_mask
            return

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_masks = [torch.nn.Parameter(torch.randn(len(v.nonzero())) * std) for i, v in
                           enumerate(edge_masks)]
        for i, v in enumerate(self.edge_masks):
            if i == 0:
                em = self.edge_masks[i]
            else:
                em = torch.cat([em, self.edge_masks[i]], 0)
        self.edge_mask = torch.nn.Parameter(em)

        # 初始化特征反事实用到的矩阵
        # 近距离扰动：Ma矩阵
        # perturbation_matrices = torch.FloatTensor(N, N)
        # torch.nn.init.xavier_uniform_(perturbation_matrices.data, gain=5 / 3)
        # self.perturbation_matrices = torch.nn.Parameter(perturbation_matrices)
        ## 正态分布初始化
        # self.perturbation_matrices = torch.nn.Parameter(torch.FloatTensor(N, N))
        # std = torch.nn.init.calculate_gain("relu") * math.sqrt(
        #     2.0 / (N + N)
        # )
        # with torch.no_grad():
        #     self.perturbation_matrices.normal_(1.0, std)

        # perturbation_biases = torch.FloatTensor(N, N)
        # torch.nn.init.xavier_uniform_(perturbation_biases.data, gain=5 / 3)
        # self.perturbation_biases = torch.nn.Parameter(perturbation_biases)
        ## 正态分布初始化
        # self.perturbation_biases = torch.nn.Parameter(torch.FloatTensor(N, N))
        # std = torch.nn.init.calculate_gain("relu") * math.sqrt(
        #     2.0 / (N + N)
        # )
        # with torch.no_grad():
        #     self.perturbation_biases.normal_(1.0, std)

        # 特征遮盖：Mb矩阵
        # masking_matrices = torch.FloatTensor(1, F)
        # torch.nn.init.xavier_uniform_(masking_matrices.data, gain=5 / 3)
        # self.masking_matrices = torch.nn.Parameter(masking_matrices)
        ## 正态分布初始化
        self.masking_matrices = torch.nn.Parameter(torch.FloatTensor(1, F))
        std = torch.nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (1 + F)
        )
        with torch.no_grad():
            self.masking_matrices.normal_(1.0, std)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.node_feat_mask = None
        self.edge_masks = None
        self.edge_mask = None

    @property
    def num_hops(self):
        if self.__num_hops__ is not None:
            return self.__num_hops__

        k = 0
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                k += 1
        return k

    def __flow__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                return module.flow
        return 'source_to_target'

    def __k_hop_subgraph__(self, node_idx, num_hops, edge_index, edge_type, want_type, relabel_nodes=False,
                           num_nodes=None, flow='source_to_target'):

        assert flow in ['source_to_target', 'target_to_source']
        if flow == 'target_to_source':
            row, col = edge_index
        else:
            col, row = edge_index

        node_mask = row.new_empty(self.num, dtype=torch.bool)
        edge_mask = row.new_empty(row.size(0), dtype=torch.bool)
        edge_masks = [edge_mask.clone().fill_(False)] * len(want_type)

        if isinstance(node_idx, (int, list, tuple)):
            node_idx = torch.tensor([node_idx], device=row.device).flatten()
        else:
            node_idx = node_idx.to(row.device)

        subsets = [node_idx]
        # 进行k跳邻居的选择
        for _ in range(num_hops):
            node_mask.fill_(False)
            node_mask[subsets[-1]] = True
            torch.index_select(node_mask, 0, row, out=edge_mask)
            subsets.append(col[edge_mask])

        subset, inv = torch.cat(subsets).unique(return_inverse=True)
        inv = inv[:node_idx.numel()]  # 获取指定的 node_idx

        node_mask.fill_(False)
        node_mask[subset] = True
        edge_mask = node_mask[row] & node_mask[col]  # 重新对 edge_mask 进行赋值，之前的值不管
        # edge_index = edge_index[:, edge_mask]

        # 按边类型
        e = row.new_zeros(row.size(0), dtype=torch.bool)
        for i in list(edge_mask.nonzero()):
            neb_edge_type = edge_type[i.item()]
            neb_edge_type_key = [k for k, v in want_type.items() if neb_edge_type in v]  # 获取这条边的类别在字典里对应的值
            if neb_edge_type_key:
                a = edge_masks[neb_edge_type_key[0]].clone()
                a[i] = True
                edge_masks[neb_edge_type_key[0]] = a
                e[i] = True

        edge_masks = [v for i, v in enumerate(edge_masks) if len(v.nonzero()) > 0]
        edge_index = edge_index[:, e]

        if relabel_nodes:
            node_idx = row.new_full((num_nodes,), -1)
            node_idx[subset] = torch.arange(subset.size(0), device=row.device)
            edge_index = node_idx[edge_index]

        return subset, edge_index, inv, e, edge_masks

    def __subgraph__(self, node_idx, num_hops, x, edge_index, edge_type, want_type, test_idx_dic, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        subset, edge_index, mapping, edge_mask, edge_masks = self.__k_hop_subgraph__(
            node_idx, num_hops, edge_index, edge_type, want_type, relabel_nodes=False,
            num_nodes=num_nodes, flow=self.__flow__())

        s = torch.tensor([test_idx_dic[v.item()] for i, v in enumerate(subset) if
                          v.item() in test_idx_dic.keys() and v.item() < 3976])  # 获得这些节点在字典里的对应序号
        s = torch.tensor([i.item() for i in s if i.item() < 474])
        x = x[s]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, edge_masks, kwargs

    # 提取元路径
    def extract_metapaths(self, graph, start_node, max_depth=3, current_path=[]):
        metapaths = []

        if len(current_path) <= max_depth * 2:
            neighbors = list(graph.neighbors(start_node))
            for neighbor in neighbors:
                if neighbor not in current_path:
                    new_path = current_path + [start_node, neighbor]
                    metapaths.append(new_path)
                    metapaths.extend(self.extract_metapaths(graph, neighbor, max_depth, new_path))

        # 用于检查一个列表是否是另一个列表的子集
        def is_subset(sublist, other):
            return set(sublist).issubset(set(other))

        # 获得最终的metapaths
        final_metapaths = [item for i, item in enumerate(metapaths) if
                           not any(is_subset(item, other) for j, other in enumerate(metapaths) if i != j)]

        return final_metapaths

    def from_meta_paths_to_subgraphs(self, node_idx, x, idx, all_idx, hete_graph, hyper_graph, classifier, pred_label,
                                     max_degree=10, k=3):
        edge_index = copy.deepcopy(hete_graph[0])
        # 统计节点的度
        degrees = defaultdict(int)
        for edge in edge_index:
            degrees[edge[0]] += 1  # 出度
            degrees[edge[1]] += 1  # 入度
        degrees_dict = dict(degrees)
        # 删去对应的边
        edges_to_remove = []
        for index, edge in enumerate(edge_index):
            node1, node2 = edge
            # if node1 in degrees_dict and node2 in degrees_dict:
            # 删去高度数节点对应的边
            if degrees_dict[node1] > max_degree or degrees_dict[node2] > max_degree:
                if node1 != node_idx and node2 != node_idx:
                    edges_to_remove.append(index)
        for index in reversed(edges_to_remove):
            del edge_index[index]
        # 去除重复的边
        unique_edge = set(map(tuple, edge_index))
        unique_edge_index = [list(edge) for edge in unique_edge]
        # 获得目标节点的不同元路径
        G = nx.Graph()  # 构建图
        for edge in unique_edge_index:
            G.add_edge(edge[0], edge[1])
        metapaths = self.extract_metapaths(G, node_idx)
        # 对不同的meta_path进行mask求diff
        # mask之前的结果
        hyp_graph = get_sub_hyp_graph(idx, hyper_graph, 3976)
        company_emb = self.model(hete_graph=hete_graph, hyp_graph=hyp_graph, idx=idx,
                                 x=x.float())
        res = classifier.forward(company_emb).reshape(-1, 2)
        target_res = res[idx.index(node_idx)]

        all_diff = []
        for metapath in metapaths:
            if len(metapath) > 2:  # 起点和终点之间还有其他节点
                masked_edges = [metapath[i:i + 2] for i in range(0, len(metapath), 2)]
            else:
                masked_edges = metapath
            masked_matrix = np.zeros(len(hete_graph[0]), dtype=bool)
            for edge in masked_edges:
                masked_matrix += np.all(np.isin(hete_graph[0], edge), axis=1)  # 只留下了需要masked的边
            maskout_matrix = ~masked_matrix
            # 获得maskout的节点特征
            maskout_idx = [node for node in idx if node not in metapath]
            maskout_idx.append(node_idx)
            maskout_idx_index = [idx.index(node) for node in maskout_idx]
            maskout_x = x[maskout_idx_index]
            # 获得maskout的异质图
            if np.all(maskout_matrix == False):
                maskout_edge_index = [sublist for sublist in hete_graph[0] if node_idx in sublist]
                indices = [index for index, sublist in enumerate(hete_graph[0]) if sublist in maskout_edge_index]
                maskout_matrix[indices] = True
            maskout_edge_index = [edge for edge, mask in zip(hete_graph[0], maskout_matrix) if mask]
            maskout_edge_type = [edge for edge, mask in zip(hete_graph[1], maskout_matrix) if mask]
            maskout_edge_weight = [edge for edge, mask in zip(hete_graph[2], maskout_matrix) if mask]
            maskout_hete_graph = [maskout_edge_index, maskout_edge_type, maskout_edge_weight]
            # 获得maskout的超图
            maskout_hyp_graph = get_sub_hyp_graph(maskout_idx, hyper_graph, 3976)

            maskout_company_emb = self.model(hete_graph=maskout_hete_graph, hyp_graph=maskout_hyp_graph,
                                             idx=maskout_idx, x=maskout_x.float())
            maskout_res = classifier.forward(maskout_company_emb).reshape(-1, 2)
            maskout_target_res = maskout_res[maskout_idx.index(node_idx)]
            # 针对目标节点求 diff
            diff = np.abs(target_res[pred_label].cpu().detach() - maskout_target_res[pred_label].cpu().detach())
            all_diff.append(diff)

        # 选择 diff 位于前 k 的 metapath
        top_k_index = list(np.argsort(all_diff)[::-1][:k])
        top_k_metapaths = [metapaths[i] for i in top_k_index]
        top_k_masked_edges = [[metapath[i:i + 2] for i in range(0, len(metapath), 2)] if len(metapath) > 2 else metapath
                              for metapath in top_k_metapaths]
        top_k_mm = np.zeros(len(hete_graph[0]), dtype=bool)
        for edge in top_k_masked_edges:
            top_k_mm += np.all(np.isin(hete_graph[0], edge), axis=1)

        tkm_edge_index = [edge for edge, mask in zip(hete_graph[0], top_k_mm) if mask]

        final_mm_adjs = torch.zeros((len(all_idx), len(all_idx)))
        for edge in tkm_edge_index:
            if edge[0] in all_idx and edge[1] in all_idx:
                row_idx = all_idx.index(edge[0])
                col_idx = all_idx.index(edge[1])
                final_mm_adjs[row_idx, col_idx] = 1
                final_mm_adjs[col_idx, row_idx] = 1

        return final_mm_adjs

    def get_k_relation_subgraphs(self, hete_graph, k=4):
        subgraph_dict = defaultdict(list)
        for sub_edge_index, edge_type, edge_weight in zip(hete_graph[0], hete_graph[1], hete_graph[2]):
            subgraph_dict[edge_type].append((sub_edge_index, edge_weight))
        subgraphs_dd = {label: {'edge_index': [], 'edge_type': [], 'edge_weight': []} for label in
                        set(hete_graph[1])}
        subgraphs_dl = {label: {} for label in set(hete_graph[1])}
        len_count = dict()
        for edge_type, edges_and_data in subgraph_dict.items():
            for sub_edge_index, edge_weight in edges_and_data:
                subgraphs_dd[edge_type]['edge_index'].append(sub_edge_index)
                subgraphs_dd[edge_type]['edge_type'].append(edge_type)
                subgraphs_dd[edge_type]['edge_weight'].append(edge_weight)
            subgraphs_dl[edge_type] = [subgraphs_dd[edge_type]['edge_index'], subgraphs_dd[edge_type]['edge_type'],
                                       subgraphs_dd[edge_type]['edge_weight']]
            len_count[edge_type] = len(subgraphs_dd[edge_type]['edge_type'])

        tk_keys = sorted(len_count, key=len_count.get, reverse=True)[:k]
        k_subgraphs = {key: subgraphs_dl[key] for key in tk_keys if key in subgraphs_dl}
        return k_subgraphs

    def from_adjs_to_hete_graph(self, sub_idx, adjs):
        row_indices, col_indices = np.where(adjs == 1)  # 找到二进制矩阵中值为1的位置索引
        sub_idx_array = np.array(sub_idx)
        p_ei = np.column_stack((sub_idx_array[row_indices], sub_idx_array[col_indices]))  # 使用这些索引构建节点索引对
        pei = p_ei.tolist()
        # 0：公司到人，1：人到公司，2：公司到公司
        p_edge_index, p_edge_type, p_edge_weight = [], [], []
        for edge in pei:
            if edge[0] < 3976 and edge[1] < 3976:
                p_edge_type.append(2)
            elif edge[0] >= 3976 and edge[1] < 3976:
                p_edge_type.append(1)
            elif edge[0] < 3976 and edge[1] >= 3976:
                p_edge_type.append(0)
            else:
                continue
            p_edge_index.append(edge)
            p_edge_weight.append(1)
        p_hete_graph = [p_edge_index, p_edge_type, p_edge_weight]
        return p_hete_graph

    def from_edge_index_to_adjs(self, idxs, edge_index_list):
        idx = list(np.sort(np.unique(edge_index_list)))
        idx_dic = {node: i for i, node in enumerate(idx)}
        edge_index = torch.LongTensor(edge_index_list).transpose(0, 1)
        transferred_edge_index = torch.zeros_like(edge_index)
        for i in range(edge_index.size(1)):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            if u in idx and v in idx:
                transferred_edge_index[0, i] = idx_dic[u]
                transferred_edge_index[1, i] = idx_dic[v]
        values = torch.Tensor([1 for _ in range(edge_index.size()[1])])
        adjs = torch.sparse_coo_tensor(transferred_edge_index, values, (len(idx), len(idx)), dtype=torch.float)
        adjs_dense = adjs.to_dense()
        adjs_dense = adjs_dense[:len(idxs), :len(idxs)]
        return adjs_dense

    def __loss__(self, node_idx, res, log_logits_p, log_logits_p1, log_logits_p2, log_logits_dis_p, l_recon):
        pre_loss = - res[node_idx]
        pre_loss1 = - log_logits_p[node_idx]  # 获得 node_idx 的损失
        pre_loss2 = - log_logits_p1[node_idx]
        pre_loss3 = - log_logits_p2[node_idx]
        pre_loss4 = - log_logits_dis_p[node_idx]

        ce_func = nn.CrossEntropyLoss()

        factual_loss = ce_func(pre_loss, pre_loss1) + ce_func(pre_loss, pre_loss2) + ce_func(pre_loss, pre_loss3)
        counterfactual_loss = ce_func(1 - pre_loss, pre_loss4)

        parameters = [param.to(self.device) for name, param in self.gcn1.named_parameters()]
        l2_regularization_gcn1 = sum(param.norm(2) ** 2 for param in parameters)
        parameters = [param.to(self.device) for name, param in self.gcn2.named_parameters()]
        l2_regularization_gcn2 = sum(param.norm(2) ** 2 for param in parameters)
        parameters = [param.to(self.device) for name, param in self.gcn3.named_parameters()]
        l2_regularization_gcn3 = sum(param.norm(2) ** 2 for param in parameters)
        parameters = [param.to(self.device) for name, param in self.gcn4.named_parameters()]
        l2_regularization_gcn4 = sum(param.norm(2) ** 2 for param in parameters)
        l2_regularization_gcn = l2_regularization_gcn1 + l2_regularization_gcn2 + \
                                l2_regularization_gcn3 + l2_regularization_gcn4

        parameters = [param.to(self.device) for name, param in self.mlp1.named_parameters()]
        l2_regularization_mlp1 = sum(param.norm(2) ** 2 for param in parameters)
        parameters = [param.to(self.device) for name, param in self.mlp2.named_parameters()]
        l2_regularization_mlp2 = sum(param.norm(2) ** 2 for param in parameters)

        l2_regularization = l2_regularization_gcn + l2_regularization_mlp1 + l2_regularization_mlp2
        l2 = 0.5 * self.weight_decay * l2_regularization

        loss = self.weight_loss * l_recon + self.weight_loss1 * factual_loss + \
               self.weight_loss2 * counterfactual_loss + l2

        return loss

    def explain_node(self, node_idx, x, edge_index, classifier, hete_graph, want_type, test_idx_dic, hyper_graph,
                     target, **kwargs):
        r"""Learns and returns a node feature mask and an edge mask that play a
        crucial role to z_explain_model the prediction made by the GNN for node
        :attr:`node_idx`.

        Args:
            node_idx (int): The node to z_explain_model.
            x (Tensor): The node feature matrix.
            edge_index (LongTensor): The edge indices.
            **kwargs (optional): Additional arguments passed to the GNN module.

        :rtype: (:class:`Tensor`, :class:`Tensor`)
        """

        self.model.eval()
        self.__clear_masks__()

        num_nodes = x.size(0)
        num_edges = edge_index.size(1)
        edge_type = hete_graph[1]

        # Only operate on a k-hop subgraph around `node_idx`.
        # 返回的 hard_edge_mask 是子图边的索引，hard_edge_masks 是每种类别子图边的索引
        sub_x, sub_edge_index, mapping, hard_edge_mask, hard_edge_masks, _ = self.__subgraph__(node_idx,
                                                                                               self.num_hops,
                                                                                               x, edge_index,
                                                                                               edge_type,
                                                                                               want_type,
                                                                                               test_idx_dic,
                                                                                               **kwargs)
        sub_idx, sub_hete_graph, sub_hyp_graph = get_sub_info(node_idx, sub_edge_index, hard_edge_mask, hete_graph,
                                                              hyper_graph, test_idx_dic)

        if sub_hete_graph[0]:
            mapping = torch.tensor([sub_idx.index(node_idx)])
            # mapping1 = torch.tensor([sub_idx_1.index(node_idx)])
            # Get the initial prediction.
            if target is None:
                with torch.no_grad():
                    company_emb = self.model(hete_graph=sub_hete_graph, hyp_graph=sub_hyp_graph, idx=sub_idx,
                                             x=sub_x.float())
                    res = classifier.forward(company_emb).reshape(-1, 2)
                    pred_label = res.argmax(dim=1)[mapping]
            else:
                pred_label = target  # 真实标签

            # 子图的元路径子图
            all_idx = list(np.sort(np.unique(sub_edge_index)))
            mm_adjs = self.from_meta_paths_to_subgraphs(node_idx, sub_x.clone(), copy.deepcopy(sub_idx), all_idx,
                                                        copy.deepcopy(sub_hete_graph), copy.deepcopy(hyper_graph),
                                                        classifier, pred_label)

            threshold = 0.5
            x_mlp1 = self.mlp1(sub_x.float())
            sub_x_binary1 = (torch.sigmoid(x_mlp1) >= threshold).float()
            x_mlp2 = self.mlp2(sub_x.float())
            sub_x_binary2 = (torch.sigmoid(x_mlp2) >= threshold).float()  # 归为0或1

            # 原始子图按关系类型换分为不同的子图，选边数为前4的
            k_subgraphs = self.get_k_relation_subgraphs(sub_hete_graph)
            z_all = []
            z = 0

            gcns = [self.gcn1, self.gcn2, self.gcn3, self.gcn4]

            for i, subgraph in enumerate(k_subgraphs.values()):
                gcn_results = gcns[i].forward(subgraph, sub_idx, sub_x_binary1.float(), sub_x_binary2.float())
                gcn_results = gcn_results[all_idx]
                z_all.append(gcn_results)
                z += gcn_results  ## 此处用的是加，可以考虑其他方法

            #  X` 和 X-X`
            p_x = sub_x * (sub_x_binary1 + sub_x_binary2)
            dis_p_x = sub_x - p_x
            dis_p_x[dis_p_x < 0] = 0

            # A`
            a = 0
            p_threshold = 0.5
            for i, zi in enumerate(z_all):
                zi_h = torch.hstack((zi, z))
                a += torch.matmul(zi_h, zi_h.t())
            a = (a - torch.min(a)) / (torch.max(a) - torch.min(a))
            p_adjs = (a >= p_threshold).float()
            # 加上自连接边
            if torch.all(p_adjs[:len(sub_idx), :len(sub_idx)] == 0).item():
                p_adjs = torch.add(p_adjs, torch.eye(p_adjs.shape[0]))
                p_adjs[p_adjs >= 1] = 1
            p_hete_graph = self.from_adjs_to_hete_graph(all_idx, p_adjs)
            # A-A`
            adjs = self.from_edge_index_to_adjs(all_idx, sub_hete_graph[0])
            adjs[adjs >= 1] = 1
            dis_p_adjs = torch.abs(adjs - p_adjs)
            # dis_p_adjs = adjs - p_adjs
            # is_all_zero = torch.all(dis_p_adjs == 0)
            # if is_all_zero == True:
            #     dis_p_adjs = 1 - p_adjs
            dis_p_hete_graph = self.from_adjs_to_hete_graph(all_idx, dis_p_adjs)

            self.__set_masks__(x, edge_index, edge_masks=hard_edge_masks)  # 设置 edge mask（之前为空，在此处赋值）
            self.to(x.device)

            optimizer = torch.optim.Adam(
                [self.masking_matrices, self.edge_mask, self.weight_loss.cpu().detach(),
                 self.weight_loss2.cpu().detach(), self.weight_loss1.cpu().detach(),
                 self.weight_loss3.cpu().detach(), *self.gcn1.parameters(), *self.gcn2.parameters(),
                 *self.gcn3.parameters(), *self.gcn4.parameters(), *self.mlp1.parameters(), *self.mlp2.parameters()],
                lr=self.lr)

            if self.log:  # pragma: no cover
                pbar = tqdm(total=self.epochs)
                pbar.set_description(f'Explain node {node_idx}')

            # 原始结果，子图作为输入，存在分布漂移
            company_emb = self.model(hete_graph=sub_hete_graph, hyp_graph=sub_hyp_graph, idx=sub_idx,
                                     x=sub_x.float())
            res = classifier.forward(company_emb).reshape(-1, 2)

            # 三部分的扰动：
            ## 1、元路径：mm_idx, mm_x, mm_hete_graph, mm_hyp_graph
            ## 2、GCN：sub_idx, p_x, p_hete_graph, sub_hyp_graph
            ## 3、GCN：sub_idx, dis_p_x, dis_p_hete_graph, sub_hyp_graph
            for epoch in range(1, self.epochs + 1):
                optimizer.zero_grad()
                company_emb_p = self.model(hete_graph=p_hete_graph, hyp_graph=sub_hyp_graph, idx=sub_idx,
                                           x=p_x.float())
                log_logits_p = classifier.forward(company_emb_p).reshape(-1, 2)

                company_emb_p1 = self.model(hete_graph=p_hete_graph, hyp_graph=sub_hyp_graph, idx=sub_idx,
                                            x=sub_x.float())
                log_logits_p1 = classifier.forward(company_emb_p1).reshape(-1, 2)

                company_emb_p2 = self.model(hete_graph=sub_hete_graph, hyp_graph=sub_hyp_graph, idx=sub_idx,
                                            x=p_x.float())
                log_logits_p2 = classifier.forward(company_emb_p2).reshape(-1, 2)

                company_emb_dis_p = self.model(hete_graph=dis_p_hete_graph, hyp_graph=sub_hyp_graph, idx=sub_idx,
                                               x=dis_p_x.float())
                log_logits_dis_p = classifier.forward(company_emb_dis_p).reshape(-1, 2)

                l_recon = nn.functional.mse_loss(adjs, mm_adjs)

                loss = self.__loss__(mapping, res.detach(), log_logits_p.detach(), log_logits_p1.detach(),
                                     log_logits_p2.detach(), log_logits_dis_p.detach(), l_recon)

                # print(loss)
                loss.backward()
                optimizer.step()

                if self.log:  # pragma: no cover
                    pbar.update(1)

            if self.log:  # pragma: no cover
                pbar.close()

            node_feat_mask = self.masking_matrices.detach().sigmoid()
            node_feat_mask = node_feat_mask.max(0)[0]

            edge_mask = self.edge_mask.new_zeros(num_edges)
            edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()

        else:
            node_feat_mask = None
            edge_mask = None

        self.__clear_masks__()

        return node_feat_mask, edge_mask

    def visualize_subgraph(self, test_idx_dic, node_idx, edge_index, edge_mask, y=None, threshold=None, edge_y=None,
                           node_alpha=None,
                           seed=10, **kwargs):
        edge_mask = edge_mask[edge_mask.nonzero()]
        assert edge_mask.size(0) == edge_index.size(1)

        # Only operate on a k-hop subgraph around `node_idx`.
        subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, edge_index, relabel_nodes=True, num_nodes=None, flow=self.__flow__())

        edge_mask = edge_mask[hard_edge_mask]

        if threshold is not None:
            edge_mask = (edge_mask >= threshold).to(torch.float)

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            subset_index = [test_idx_dic[i.item()] for i in subset]
            sub_subset_index = [i for i in subset_index if i < 474]
            person_index = [i for i in subset_index if i > 474]
            y = y[sub_subset_index].to(torch.float) / y.max().item()
            y = torch.cat((y, -torch.ones(len(person_index))), dim=0)

        if edge_y is None:
            edge_color = ['black'] * edge_index.size(1)
        else:
            colors = list(plt.rcParams['axes.prop_cycle'])
            edge_color = [
                colors[i % len(colors)]['color']
                for i in edge_y[hard_edge_mask]
            ]

        data = Data(edge_index=edge_index, att=edge_mask,
                    edge_color=edge_color, y=y, num_nodes=y.size(0)).to('cpu')
        G = to_networkx(data, node_attrs=['y'], edge_attrs=['att', 'edge_color'])
        mapping = {k: i for k, i in enumerate(subset.tolist())}
        G = nx.relabel_nodes(G, mapping)  # 将边由节点索引变为节点id

        # 获得用于画节点的信息
        node_args = set(signature(nx.draw_networkx_nodes).parameters.keys())
        node_kwargs = {k: v for k, v in kwargs.items() if k in node_args}
        node_kwargs['node_size'] = kwargs.get('node_size') or 800
        node_kwargs['cmap'] = kwargs.get('cmap') or 'cool'

        # 获得用于画label的信息
        label_args = set(signature(nx.draw_networkx_labels).parameters.keys())
        label_kwargs = {k: v for k, v in kwargs.items() if k in label_args}
        label_kwargs['font_size'] = kwargs.get('font_size') or 10

        pos = nx.spring_layout(G, seed=seed)
        ax = plt.gca()
        for source, target, data in G.edges(data=True):
            ax.annotate(
                '', xy=pos[target], xycoords='data', xytext=pos[source],
                textcoords='data', arrowprops=dict(
                    arrowstyle="->",
                    alpha=max(data['att'], 0.1),
                    color=data['edge_color'],
                    shrinkA=sqrt(node_kwargs['node_size']) / 2.0,
                    shrinkB=sqrt(node_kwargs['node_size']) / 2.0,
                    connectionstyle="arc3,rad=0.1",
                ))

        if node_alpha is None:
            nx.draw_networkx_nodes(G, pos, node_color=y.tolist(),
                                   **node_kwargs)
        else:
            node_alpha_subset = node_alpha[subset]
            assert ((node_alpha_subset >= 0) & (node_alpha_subset <= 1)).all()
            nx.draw_networkx_nodes(G, pos, alpha=node_alpha_subset.tolist(),
                                   node_color=y.tolist(), **node_kwargs)

        nx.draw_networkx_labels(G, pos, **label_kwargs)

        return ax, G

    def __repr__(self):
        return f'{self.__class__.__name__}()'
