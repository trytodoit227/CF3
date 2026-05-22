from math import sqrt
from typing import Optional
from inspect import signature

import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, to_networkx

from utils.model_utils import get_sub_info, gen_attribute_hg

EPS = 1e-15


# 计算节点v_i的传入信息和传出信息的函数
def agg(mask):
    result = torch.tensor(1.)
    mask = mask.view(-1)
    weight = mask.shape[0]
    zeros = 0
    for m in mask:
        if m == 0.:
            zeros += 1
            continue
        result *= m
    weight = weight - zeros
    result = torch.pow(result, 1. / weight) if weight != 0 else torch.tensor(0.)
    if weight != 0:
        result = torch.pow(result, (weight + zeros) / weight)
    return result


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
                 num_hops: Optional[int] = None, log: bool = True, agg1="agg", agg2="max"):
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

        if agg1 == "mean":
            self.agg1 = torch.mean
        elif agg1 == "min":
            self.agg1 = torch.min
        elif agg1 == "max":
            self.agg1 = torch.max
        elif agg1 == "sum":
            self.agg1 = torch.sum
        else:
            self.agg1 = agg

        if agg2 == "mean":
            self.agg2 = torch.mean
        elif agg2 == "min":
            self.agg2 = torch.min
        elif agg2 == "max":
            self.agg2 = torch.max
        elif agg2 == "sum":
            self.agg2 = torch.sum
        else:
            self.agg2 = agg

    def __set_masks__(self, x, edge_index, edge_masks=None, edge_mask=None):
        (N, F), E = x.size(), edge_index.size(1)  # N：子图节点数量；F：子图特征维度；E：子图边的数量

        if edge_mask is not None:
            for module in self.model.modules():
                if isinstance(module, MessagePassing):
                    module.__explain__ = True
                    module.__edge_mask__ = edge_mask
            return

        std = 0.1
        if self.feat_mask_type == "individual_feature":
            node_feat_mask = torch.randn(N, F) * std
        elif self.feat_mask_type == "scalar":
            node_feat_mask = torch.randn(N, 1) * std
        else:
            node_feat_mask = torch.randn(1, F) * std  # 对节点特征矩阵进行随机初始赋值，默认为此
        # node_feat_mask = torch.mean(node_feat_mask, dim=0, keepdim=False)
        self.node_feat_mask = torch.nn.Parameter(node_feat_mask)

        std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
        # torch.nn.init.calculate_gain('relu') = √2；sqrt(2.0 / N)：初始化权重
        # self.edge_mask = torch.nn.Parameter(torch.randn(E) * std)  # 对子图的边进行初始化（相同节点数量初始化的值相同）
        self.edge_masks = [torch.nn.Parameter(torch.randn(len(v.nonzero())) * std) for i, v in
                           enumerate(edge_masks)]  # 初始化
        for i, v in enumerate(self.edge_masks):
            if i == 0:
                em = self.edge_masks[i]
            else:
                em = torch.cat([em, self.edge_masks[i]], 0)
        self.edge_mask = torch.nn.Parameter(em)

        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = True
                module.__edge_mask__ = self.edge_mask

    def __clear_masks__(self):
        for module in self.model.modules():
            if isinstance(module, MessagePassing):
                module.__explain__ = False
                module.__edge_mask__ = None
        self.out_degree = None
        self.in_degree = None
        self.out_edge_mask = None
        self.in_edge_mask = None
        self.self_loop_mask = None
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

    def __subgraph__(self, node_idx, x, edge_index, edge_type, want_type, test_idx_dic, **kwargs):
        num_nodes, num_edges = x.size(0), edge_index.size(1)
        subset, edge_index, mapping, edge_mask, edge_masks = self.__k_hop_subgraph__(
            node_idx, self.num_hops, edge_index, edge_type, want_type, relabel_nodes=False,
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

    def __refine_mask__(self, mask, beta=1., training=True):  # 伯努利分布
        if training:
            random_noise = torch.rand(mask.shape)
            random_noise = torch.log(random_noise) - torch.log(1 - random_noise)
            s = (random_noise + mask) / beta  # 公式4
            s = s.sigmoid()
            z = s * 1.5 - 0.25
            z = z.clamp(0, 1)
        else:
            z = (mask / beta).sigmoid()

        return z

    def __get_indices__(self, x, edge_index, subset):
        (N, F), E = x.size(), edge_index.size(1)

        self.out_edge_mask = torch.zeros(N, E, dtype=torch.bool)
        self.in_edge_mask = torch.zeros(N, E, dtype=torch.bool)
        for i, n in enumerate(subset):
            self.out_edge_mask[i] = (edge_index[0] == n) & (edge_index[1] != n)
            self.in_edge_mask[i] = (edge_index[1] == n) & (edge_index[0] != n)

        self.out_degree = torch.zeros(N, dtype=torch.int)  # 子图节点的传出信息
        self.in_degree = torch.zeros(N, dtype=torch.int)  # 子图节点的传入信息
        for n in range(N):
            in_num = torch.sum(self.in_edge_mask[n], dtype=torch.int)
            out_num = torch.sum(self.out_edge_mask[n], dtype=torch.int)
            self.in_degree[n] = in_num
            self.out_degree[n] = out_num

        self.self_loop_mask = torch.zeros(N, dtype=torch.long)
        for e in range(E):
            if edge_index[0, e] == edge_index[1, e]:  # 添加自回归边
                self.self_loop_mask[edge_index[0, e]] = e
        if self.self_loop_mask.sum() == 0:
            self.self_loop_mask = None

    def __loss__(self, node_idx, log_logits, pred_label):
        loss = -log_logits[node_idx, pred_label]  # 获得 node_idx 的损失

        # 求边的loss，原始代码的方法
        # m = self.edge_mask.sigmoid()  # m：边缘掩码；sigmod：转换每个值到 [0, 1] 之间；公式 σ(M)
        # edge_reduce = getattr(torch, self.coeffs['edge_reduction'])  # sum
        # loss = loss + self.coeffs['edge_size'] * edge_reduce(m)
        # ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)  # 交叉熵公式
        # loss = loss + self.coeffs['edge_ent'] * ent.mean()  # ent.mean()：平均损失

        # 增加了边的类型,求边的loss
        ms = [i.sigmoid() for i in self.edge_masks]
        edge_reduce = getattr(torch, self.coeffs['edge_reduction'])  # sum
        loss = sum([loss + self.coeffs['edge_size'] * edge_reduce(i) for i in ms])
        ents = [-i * torch.log(i + EPS) - (1 - i) * torch.log(1 - i + EPS) for i in ms]
        loss = loss + sum([self.coeffs['edge_ent'] * i.mean() for i in ents])

        # 求节点的loss
        m = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])  # mean
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + EPS) - (1 - m) * torch.log(1 - m + EPS)
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

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
        x, edge_index, mapping, hard_edge_mask, hard_edge_masks, kwargs = self.__subgraph__(node_idx, x, edge_index,
                                                                                            edge_type, want_type,
                                                                                            test_idx_dic, **kwargs)
        # 返回的 hard_edge_mask 是子图边的索引，hard_edge_masks 是每种类别子图边的索引
        # hyp_graph = []
        # for i in ['industry', 'area', 'qualify']:
        #     hyp_graph += [gen_attribute_hg(3976, hyper_graph[i], X=None)]
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

            self.__get_indices__(x, edge_index, sub_idx)
            self.__set_masks__(x, edge_index, edge_masks=hard_edge_masks)  # 设置 edge mask（之前为空，在此处赋值）
            self.to(x.device)

            optimizer = torch.optim.Adam([self.node_feat_mask, self.edge_mask], lr=self.lr)

            if self.log:  # pragma: no cover
                pbar = tqdm(total=self.epochs)
                pbar.set_description(f'Explain node {node_idx}')

            for epoch in range(1, self.epochs + 1):
                optimizer.zero_grad()
                # node_feat_mask = self.__refine_mask__(
                #     self.node_feat_mask, beta=epoch / self.epochs) if self.node else self.node_feat_mask.sigmoid()
                h = x * self.node_feat_mask.sigmoid()
                company_emb = self.model(hete_graph=sub_hete_graph, hyp_graph=sub_hyp_graph, idx=sub_idx, x=h.float())
                log_logits = classifier.forward(company_emb).reshape(-1, 2)
                loss = self.__loss__(mapping, log_logits, pred_label)
                # l1 = sum([torch.sum(abs(param)) for param in self.model.parameters()])
                # loss = loss + 1e-3 * l1  # 加正则
                loss.backward()
                optimizer.step()

                if self.log:  # pragma: no cover
                    pbar.update(1)

            if self.log:  # pragma: no cover
                pbar.close()

            node_feat_mask = self.__refine_mask__(self.node_feat_mask,
                                                  training=False) if self.node else self.node_feat_mask.detach().sigmoid()
            sub_nodes_index = [test_idx_dic[i] for i in sub_idx]
            if self.feat_mask_type == "individual_feature":
                new_mask = x.new_zeros(num_nodes, x.size(-1)).float()
                new_mask[sub_nodes_index] = node_feat_mask
                node_feat_mask = new_mask
            elif self.feat_mask_type == "scalar":
                new_mask = x.new_zeros(num_nodes, 1).float()
                new_mask[sub_nodes_index] = node_feat_mask
                node_feat_mask = new_mask
            node_feat_mask = node_feat_mask.squeeze()
            em = self.__refine_mask__(self.edge_mask,
                                      training=False) if self.node else self.edge_mask.detach().sigmoid()
            edge_mask = self.edge_mask.new_zeros(num_edges)
            edge_mask[hard_edge_mask] = em

            """ 算节点的mask """
            node_mask = torch.zeros(num_nodes)
            if self.node:
                node_feat_msg = torch.sum(node_feat_mask * x, dim=-1).view(-1)
                # node_feat_msg = (node_feat_msg - torch.min(node_feat_msg)) / (
                #             torch.max(node_feat_msg) - torch.min(node_feat_msg))
                x = x.clone()
                x[x > 0.] = 1.
                for i, n in enumerate(sub_idx):
                    if self.out_degree[i] > 0 or self.in_degree[i] > 0:
                        out_masks = torch.zeros(1)
                        if self.out_degree[i] > 0:
                            out_masks = em[self.out_edge_mask[i]]
                        node_mask_out = out_masks * node_feat_msg[i]
                        node_mask_out = self.agg1(node_mask_out)
                        in_masks = em[
                            self.self_loop_mask[i]] if self.self_loop_mask is not None else torch.zeros(1)
                        node_mask_in = in_masks * node_feat_msg[i]
                        if self.in_degree[i] > 0:
                            in_nodes = edge_index[0, self.in_edge_mask[i]]
                            in_nodes = list(i.item() for i in in_nodes if test_idx_dic[i.item()] < 474)
                            in_nodes_index = list(sub_idx.index(j) for j in in_nodes)
                            in_masks = em[self.in_edge_mask[i]]
                            if self.self_loop_mask is not None:
                                in_masks = torch.cat((in_masks.view(-1), em[self.self_loop_mask[i]].view(-1)))
                            if in_nodes_index != []:
                                node_mask_in = in_masks * max(node_feat_msg[in_nodes_index])
                            else:
                                node_mask_in = in_masks
                        node_mask_in = self.agg1(node_mask_in)  # self.agg1：max
                        node_mask[test_idx_dic[n]] = self.agg2(
                            torch.cat((node_mask_in.view(-1), node_mask_out.view(-1))))  # self.agg2：max

            else:
                for i, n in enumerate(sub_idx):
                    out_max = torch.tensor(0, dtype=torch.float)
                    in_max = torch.tensor(0, dtype=torch.float)
                    if self.out_degree[i] > 0:
                        out_max = torch.max(em[self.out_edge_mask[i]])
                    if self.in_degree[i] > 0:
                        in_max = torch.max(em[self.in_edge_mask[i]])
                    node_mask[test_idx_dic[n]] = torch.max(out_max, in_max)

            # 考虑如何将 node_mask 融入 node_feat_mask 或者是 edge_mask
            # 将 node_mask 融入 node_feat_mask
            node_mask_copy = torch.mean(node_mask.detach().sigmoid(), dim=0, keepdim=False)
            node_feat_mask = (node_feat_mask + node_mask_copy).detach().sigmoid()
            # 将 node_mask 融入 edge_mask


        else:
            node_feat_mask = None
            edge_mask = None

        self.__clear_masks__()

        return node_feat_mask, edge_mask

    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None, threshold=None, edge_y=None, node_alpha=None,
                           seed=10, **kwargs):

        assert edge_mask.size(0) == edge_index.size(1)

        num_nodes = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
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
            y = y[subset].to(torch.float) / y.max().item()

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
