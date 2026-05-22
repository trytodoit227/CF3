import torch
import math
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from scipy.sparse import coo_matrix

from utils.model_utils import get_sub_hyp_graph


class NodeExplainerEdgeMulti(torch.nn.Module):

    def __init__(self, model, idx, x, edge_index, classifier, hete_graph, hard_edge_mask, hyp_graph, hyper_graph,
                 epochs, device, args, fix_exp=None):
        super(NodeExplainerEdgeMulti, self).__init__()
        self.model = model
        self.model.eval()
        self.idx = idx
        self.x = x
        self.edge_index = edge_index
        self.classifier = classifier
        self.hete_graph = hete_graph
        self.hard_edge_mask = hard_edge_mask
        self.hyp_graph = hyp_graph
        self.hyper_graph = hyper_graph
        # self.test_idx_dic = test_idx_dic
        self.num_epochs = epochs
        self.device = device
        self.args = args
        # self.model.to(self.device)
        self.num_nodes = self.x.shape[0]

        if fix_exp:  # fix_exp：是否固定阈值
            self.fix_exp = fix_exp * 2
        else:
            self.fix_exp = None
        self.gam = 0.5
        self.lam = 500
        self.alp = 0.6

        # std = torch.nn.init.calculate_gain('relu') * math.sqrt(2.0 / (2 * self.x.size(0)))
        # self.edge_mask = torch.nn.Parameter(torch.randn(len(self.hard_edge_mask.nonzero())) * std)

        self.edge_mask = torch.nn.Parameter(torch.randn(len(self.hard_edge_mask.nonzero())))

    def explain_nodes_gnn_stats(self, new_idx, target, node_idx):
        # 获得子图的初始预测概率
        company_emb = self.model(hete_graph=self.hete_graph, hyp_graph=self.hyp_graph, idx=self.idx,
                                 x=self.x.float())
        res = self.classifier.forward(company_emb).reshape(-1, 2)
        pred_label = res.argmax(dim=1)[new_idx]
        # 获得真实标签
        ori_label = target
        edge_mask = self.explain(node_idx, new_idx, pred_label, self.device, self.hyper_graph)
        return edge_mask

    def explain(self, node_idx, new_idx, pred_label, device, hyper_graph):
        # ExplainModelNodeMulti：解释单个节点
        #
        explainer = ExplainModelNodeMulti(
            idx=self.idx,
            x=self.x,
            hete_graph=self.hete_graph,
            hyp_graph=self.hyp_graph,  # 子图节点的超图
            hyper_graph=hyper_graph,  # 用于获取扰动后的超图
            classifier=self.classifier,
            base_model=self.model,
            device=device,
        )
        # explainer.to(self.device)
        optimizer = torch.optim.Adam(explainer.parameters(), lr=self.args.lr, weight_decay=0)
        explainer.train()

        pbar = tqdm(total=self.num_epochs)
        pbar.set_description(f'Explain node {node_idx}')

        for epoch in range(self.num_epochs):
            explainer.zero_grad()
            pred1, pred2 = explainer()  # pred1：事实推理, pred2：反事实推理
            bpr1, bpr2, l1, loss = explainer.loss(
                pred1[new_idx], pred2[new_idx], pred_label, self.gam, self.lam, self.alp)
            # if epoch % 201 == 0:
            #     print('bpr1: ', self.args.lam * self.args.alp * bpr1,
            #           'bpr2:', self.args.lam * (1 - self.args.alp) * bpr2,
            #           'l1', l1,
            #           'loss', loss)
            loss.backward()
            optimizer.step()

            pbar.update(1)

        pbar.close()

        edge_mask = self.edge_mask.new_zeros(self.hard_edge_mask.shape[0])
        masked_adj = explainer.get_masked_adj()
        masked_adj_sparse = masked_adj.to_sparse()
        if masked_adj_sparse.is_coalesced():
            adj = torch.where(masked_adj == 0, torch.zeros_like(masked_adj), torch.ones_like(masked_adj))
            raw_adj = explainer.get_raw_adj()
            edge_adj = raw_adj * adj
            filled_edge_adj = torch.cat((edge_adj, torch.zeros(edge_mask.size(0) - edge_adj.size(0))), dim=0)
            filled_edge_adj_mask = filled_edge_adj.long() != 0
            edge_mask[filled_edge_adj_mask] = masked_adj[masked_adj.nonzero()].squeeze(dim=1)
        else:
            edge_mask[self.hard_edge_mask] = self.edge_mask.detach().sigmoid()
        return edge_mask


class ExplainModelNodeMulti(torch.nn.Module):

    def __init__(self, idx, x, hete_graph, hyper_graph, hyp_graph, classifier, base_model, device):
        super(ExplainModelNodeMulti, self).__init__()
        self.idx = idx
        self.x = x
        self.hete_graph = hete_graph
        self.hyp_graph = hyp_graph
        self.hyper_graph = hyper_graph
        self.classifier = classifier
        self.base_model = base_model
        self.device = device

        self.num_nodes = self.x.size(0)

        self.adj_mask = self.construct_adj_mask()  # 正态分布初始化的adj掩码矩阵
        self.diag_mask = torch.ones(self.num_nodes, self.num_nodes) - torch.eye(self.num_nodes)
        if self.device != "cpu":
            self.diag_mask = self.diag_mask.cuda()

        self.idx_dic = None
        self.idx_dic_swapped = None

    def forward(self):
        masked_adj = self.get_masked_adj()
        # 修改
        p_adj = self.get_raw_adj() - masked_adj  # pred2
        p_adj = p_adj.reshape(self.num_nodes, self.num_nodes)
        p_adj_sparse = p_adj.to_sparse()
        if p_adj_sparse.is_coalesced():
            perturbation_edge_index = p_adj_sparse.indices()
            # 还原index
            p_edge_index = torch.zeros_like(perturbation_edge_index)
            for i in range(perturbation_edge_index.size(1)):
                u = perturbation_edge_index[0, i].item()
                v = perturbation_edge_index[1, i].item()
                if u in self.idx_dic_swapped.keys() and v in self.idx_dic_swapped.keys():
                    p_edge_index[0, i] = self.idx_dic_swapped[u]
                p_edge_index[1, i] = self.idx_dic_swapped[v]
            # 获得扰动过后的异质图
            p_edge_index = p_edge_index.transpose(0, 1).tolist()
            p_edge_type = [2] * len(p_edge_index)  # 这里要改，加入人节点之后，不能直接笼统设置边类型
            p_edge_weight = [1] * len(p_edge_index)
            p_hete_graph = [p_edge_index, p_edge_type, p_edge_weight]
            # 获得扰动过后的超图
            idx = list(np.sort(np.unique(p_edge_index)))
            p_hyp_graph = get_sub_hyp_graph(idx, self.hyper_graph, 3976)
        else:
            p_hete_graph = self.hete_graph
            p_hyp_graph = self.hyp_graph

        company_emb_1 = self.base_model(hete_graph=self.hete_graph, hyp_graph=self.hyp_graph, idx=self.idx,
                                        x=self.x.float())
        pred1 = self.classifier.forward(company_emb_1).reshape(-1, 2)

        company_emb_2 = self.base_model(hete_graph=p_hete_graph, hyp_graph=p_hyp_graph, idx=self.idx,
                                        x=self.x.float())
        pred2 = self.classifier.forward(company_emb_2).reshape(-1, 2)
        return pred1, pred2

    def loss(self, pred1, pred2, pred_label, gam, lam, alp):
        relu = torch.nn.ReLU()
        f_next = torch.max(torch.cat((pred1[:pred_label],
                                      pred1[pred_label + 1:])))
        cf_next = torch.max(torch.cat((pred2[:pred_label],
                                       pred2[pred_label + 1:])))
        bpr1 = relu(gam + f_next - pred1[pred_label])
        bpr2 = relu(gam + pred2[pred_label] - cf_next)
        masked_adj = self.get_masked_adj()
        L1 = torch.linalg.norm(masked_adj, ord=1)
        loss = L1 + lam * (alp * bpr1 + (1 - alp) * bpr2)
        return bpr1, bpr2, L1, loss

    # 目的是生成一个与邻接矩阵具有相同形状的掩码矩阵，其中的元素经过正态分布始化，并可以用于影响邻接矩阵中对应位置的值。
    def construct_adj_mask(self):
        mask = torch.nn.Parameter(torch.FloatTensor(self.num_nodes, self.num_nodes))
        std = torch.nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (self.num_nodes + self.num_nodes)
        )
        with torch.no_grad():
            mask.normal_(1.0, std)
        return mask

    # 获得子图原始的adj
    def get_raw_adj(self):
        edge_index = torch.LongTensor(self.hete_graph[0]).transpose(0, 1)
        idx = list(np.sort(np.unique(edge_index)))
        self.idx_dic = {node: i for i, node in enumerate(idx)}
        self.idx_dic_swapped = {i: node for i, node in enumerate(idx)}
        transferred_edge_index = torch.zeros_like(edge_index)
        for i in range(edge_index.size(1)):
            u = edge_index[0, i].item()
            v = edge_index[1, i].item()
            if u in idx and v in idx:
                transferred_edge_index[0, i] = self.idx_dic[u]
                transferred_edge_index[1, i] = self.idx_dic[v]
        values = torch.Tensor([1 for _ in range(edge_index.size()[1])])
        adj = torch.sparse_coo_tensor(transferred_edge_index, values, (len(idx), len(idx)), dtype=torch.float)
        adj_dense = adj.to_dense()
        adj_dense = adj_dense[:len(self.idx), :len(self.idx)].reshape(1, -1)[0]

        return adj_dense

    # 根据给定的邻接矩阵权重和掩码，生成一个经过掩码调整后的邻接矩阵，用于调整图中节点之间的连接强度或权重。
    def get_masked_adj(self):
        sym_mask = torch.sigmoid(self.adj_mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        flatten_sym_mask = torch.reshape(sym_mask, (-1,))
        masked_adj = self.get_raw_adj() * flatten_sym_mask
        return masked_adj
