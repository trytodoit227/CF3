"""
除了对异质边信息的处理，
还加入了对节点特征的扰动处理
"""
from math import sqrt
from typing import Optional
from inspect import signature

import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

from utils.model_utils import get_sub_info, gen_attribute_hg

EPS = 1e-15


class HETEGNNExplainer(torch.nn.Module):
    coeffs = {
        'edge_size': 0.005,
        'edge_reduction': 'sum',
        'node_feat_size': 1.0,
        'node_feat_reduction': 'mean',
        'edge_ent': 1.0,
        'node_feat_ent': 0.1,
    }

    def __init__(self, model, num, feat_mask_type: str = "feature", epochs: int = 100, lr: float = 0.01,
                 num_hops: Optional[int] = None, log: bool = True):
        super(HETEGNNExplainer, self).__init__()
        assert feat_mask_type in ["feature", "individual_feature"]
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.__num_hops__ = num_hops
        self.log = log
        self.num = num
        self.feat_mask_type = feat_mask_type
        self.node = True

    def __set_masks__(self, x, edge_index, edge_mask=None):
        (N, F), E = x.size(), edge_index.size(1)  # N：子图节点数量；F：子图特征维度；E：子图边的数量

        if edge_mask is not None:
            for module in self.model.modules():
                if isinstance(module, MessagePassing):
                    module.__explain__ = True
                    module.__edge_mask__ = edge_mask
            return

        # std = 0.1
        # if self.feat_mask_type == "individual_feature":
        #     node_feat_mask = torch.randn(N, F) * std
        # elif self.feat_mask_type == "scalar":
        #     node_feat_mask = torch.randn(N, 1) * std
        # else:
        #     node_feat_mask = torch.randn(1, F) * std
        # # node_feat_mask = torch.mean(node_feat_mask, dim=0, keepdim=False)
        # self.node_feat_mask = torch.nn.Parameter(node_feat_mask)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)

        # 初始化特征反事实用到的矩阵
        perturbation_matrices = torch.FloatTensor(N, N)
        torch.nn.init.xavier_uniform_(perturbation_matrices.data, gain=5 / 3)
        self.perturbation_matrices = torch.nn.Parameter(perturbation_matrices)

        perturbation_biases = torch.FloatTensor(N, N)
        torch.nn.init.xavier_uniform_(perturbation_biases.data, gain=5 / 3)
        self.perturbation_biases = torch.nn.Parameter(perturbation_biases)

        # 特征遮盖：Mb矩阵
        masking_matrices = torch.FloatTensor(1, F)
        torch.nn.init.xavier_uniform_(masking_matrices.data, gain=5 / 3)
        self.masking_matrices = torch.nn.Parameter(masking_matrices)

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

    def __subgraph__(self, node_idx, x, edge_index, test_idx_dic, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        num_nodes = 6381
        subset, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx, self.num_hops, edge_index,
                                                                relabel_nodes=False,
                                                                num_nodes=num_nodes, flow=self.__flow__())

        s = torch.tensor([test_idx_dic[v.item()] for i, v in enumerate(subset) if v.item() in test_idx_dic.keys()])
        s = torch.tensor([i.item() for i in s if i.item() < 474])
        x = x[s]
        for key, item in kwargs.items():
            if torch.is_tensor(item) and item.size(0) == num_nodes:
                item = item[subset]
            elif torch.is_tensor(item) and item.size(0) == num_edges:
                item = item[edge_mask]
            kwargs[key] = item

        return x, edge_index, mapping, edge_mask, kwargs

    def generator(self, node_feature_mask, edge_index):
        # 近距离扰动
        idx = list(np.sort(np.unique(edge_index)))
        idx_dic = {node: i for i, node in enumerate(idx)}
        idx_dic_swapped = {i: node for i, node in enumerate(idx)}
        # 重新对 edge_index 进行编号
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
        adjs_dense = adjs_dense[:node_feature_mask.shape[0], :node_feature_mask.shape[0]]
        perturbation_adjs = torch.mm(self.perturbation_matrices, adjs_dense) + self.perturbation_biases
        perturbation_adjs = torch.sigmoid(perturbation_adjs)
        perturbation_adjs = torch.where(perturbation_adjs <= 0.5, torch.zeros_like(perturbation_adjs),
                                        torch.ones_like(perturbation_adjs))
        perturbation_adjs_sparse = perturbation_adjs.to_sparse()
        if perturbation_adjs_sparse.is_coalesced():
            perturbation_edge_index = perturbation_adjs_sparse.indices()
            # 还原index
            p_edge_index = torch.zeros_like(perturbation_edge_index)
            for i in range(perturbation_edge_index.size(1)):
                u = perturbation_edge_index[0, i].item()
                v = perturbation_edge_index[1, i].item()
                if u in idx_dic_swapped.keys() and v in idx_dic_swapped.keys():
                    p_edge_index[0, i] = idx_dic_swapped[u]
                p_edge_index[1, i] = idx_dic_swapped[v]
        else:
            p_edge_index = edge_index

        # 获得扰动过后的异质图
        p_edge_index = p_edge_index.transpose(0, 1).tolist()
        p_edge_type = [2] * len(p_edge_index)
        p_edge_weight = [1] * len(p_edge_index)
        p_hete_graph = [p_edge_index, p_edge_type, p_edge_weight]

        # 特征遮盖
        masking_matrices = self.masking_matrices.sigmoid()
        masked_attrs = torch.mul(masking_matrices, node_feature_mask)

        return adjs_dense, perturbation_adjs, p_hete_graph, masking_matrices, masked_attrs

    def similarity_loss(self, adjs_dense, perturbation_adjs, masking_matrix):
        diff = adjs_dense - perturbation_adjs
        diff_norm = torch.linalg.matrix_norm(diff, ord=1) / torch.ones_like(diff).sum()  # 计算F范数
        masking_matrix_norm = torch.linalg.matrix_norm(masking_matrix, ord=1) / torch.ones_like(masking_matrix).sum()
        return diff_norm - masking_matrix_norm

    def kl_div(self, predicted_results, perturbation_predicted_results, masking_predicted_results):
        predicted_results = predicted_results.softmax(dim=1)
        perturbation_predicted_results = perturbation_predicted_results.softmax(dim=1)
        masking_predicted_results = masking_predicted_results.softmax(dim=1)
        loss_func = torch.nn.KLDivLoss(reduction='batchmean')
        kl_loss_1 = loss_func(perturbation_predicted_results, predicted_results)
        kl_loss_2 = loss_func(masking_predicted_results, predicted_results)
        return kl_loss_1 + kl_loss_2

    def __loss__(self, node_idx, res, log_logits_p, log_logits_m, pred_label, adjs_dense, perturbation_adjs,
                 masking_matrix):
        loss = -log_logits_p[node_idx, pred_label] - log_logits_m[node_idx, pred_label]  # 获得 node_idx 的损失

        # 求边的loss，原始代码的方法
        m = self.edge_mask.sigmoid()
        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])
        loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['edge_ent'] * ent.mean()

        # 求节点的loss，反事实
        sim_loss = self.similarity_loss(adjs_dense, perturbation_adjs, masking_matrix)
        kl_loss = self.kl_div(res, log_logits_p, log_logits_m)
        loss = loss + (sim_loss - kl_loss)

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
        x, edge_index, mapping, hard_edge_mask, kwargs = self.__subgraph__(node_idx, x, edge_index, test_idx_dic,
                                                                           **kwargs)

        sub_idx, sub_hete_graph, sub_hyp_graph = get_sub_info(node_idx, edge_index, hard_edge_mask, hete_graph,
                                                              hyper_graph, test_idx_dic)
        if sub_hete_graph[0]:
            mapping = torch.tensor([sub_idx.index(node_idx)])
            # Get the initial prediction.
            if target is None:
                with torch.no_grad():
                    company_emb = self.model(hete_graph=sub_hete_graph, hyp_graph=sub_hyp_graph, idx=sub_idx,
                                             x=x.float())
                    res = classifier.forward(company_emb).reshape(-1, 2)
                    pred_label = res.argmax(dim=1)[mapping]
            else:
                pred_label = target

            self.__set_masks__(x, edge_index)  # 设置 edge mask（之前为空，在此处赋值）
            self.to(x.device)

            optimizer = torch.optim.Adam([self.masking_matrices, self.edge_mask], lr=self.lr)

            if self.log:  # pragma: no cover
                pbar = tqdm(total=self.epochs)
                pbar.set_description(f'Explain node {node_idx}')

            # 原始结果
            company_emb = self.model(hete_graph=sub_hete_graph, hyp_graph=sub_hyp_graph, idx=sub_idx,
                                     x=x.float())
            res = classifier.forward(company_emb).reshape(-1, 2)

            for epoch in range(1, self.epochs + 1):
                optimizer.zero_grad()
                # 扰动后的结果
                adjs_dense, perturbation_adjs, p_hete_graph, masking_matrices, masked_attrs = self.generator(x,
                                                                                                             edge_index)
                company_emb_p = self.model(hete_graph=p_hete_graph, hyp_graph=sub_hyp_graph, idx=sub_idx,
                                           x=x.float())
                log_logits_p = classifier.forward(company_emb_p).reshape(-1, 2)
                company_emb_m = self.model(hete_graph=sub_hete_graph, hyp_graph=sub_hyp_graph, idx=sub_idx,
                                           x=masked_attrs.float())
                log_logits_m = classifier.forward(company_emb_m).reshape(-1, 2)
                loss = self.__loss__(mapping, res, log_logits_p, log_logits_m, pred_label, adjs_dense,
                                     perturbation_adjs, masked_attrs)
                print(loss)
                loss.backward(retain_graph=True)
                optimizer.step()

                if self.log:  # pragma: no cover
                    pbar.update(1)

            if self.log:  # pragma: no cover
                pbar.close()

            node_feat_mask = self.masking_matrices.detach().sigmoid()
            node_feat_mask = node_feat_mask.max(0)[0]
            # nfm = self.node_feat_mask.detach().sigmoid().squeeze()
            # node_feat_mask = torch.max(nfm, node_feat_mask)

            edge_mask = self.edge_mask.new_zeros(num_edges)
            edge_mask[hard_edge_mask] = self.edge_mask.detach().sigmoid()

        else:
            node_feat_mask = None
            edge_mask = None

        self.__clear_masks__()

        return node_feat_mask, edge_mask

    def __repr__(self):
        return f'{self.__class__.__name__}()'
