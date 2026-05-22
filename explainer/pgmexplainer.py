import torch
import numpy as np
import pandas as pd
from scipy.special import softmax
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import MessagePassing
from pgmpy.estimators.CITests import chi_square
from utils.model_utils import gen_attribute_hg, set_random_seed

class PGMExplainer:
    def __init__(self, model, features, edge_index, classifier, hete_graph, hyper_graph,
                 num_hops, test_company_num, mode=0, print_result=1):
        self.model = model
        self.model.eval()
        self.x = features
        self.edge_index = edge_index
        self.classifier = classifier
        self.hete_graph = hete_graph
        self.num_hops = num_hops
        self.mode = mode
        self.print_result = print_result
        self.test_company_num = test_company_num

        hyp_graph = []
        for i in ['industry', 'area', 'qualify']:
            hyp_graph += [gen_attribute_hg(3976, hyper_graph[i], X=None)]
        self.hyp_graph = hyp_graph

    def perturb_features_on_node(self, feature_matrix, node_idx, test_idx_dic, random=0, mode=0):
        # return a random perturbed feature matrix
        # random = 0 for nothing, 1 for random.
        # mode = 0 for random 0-1, 1 for scaling with original feature

        X_perturb = feature_matrix
        if mode == 0:
            if random == 0:
                perturb_array = X_perturb[node_idx]
            elif random == 1:
                perturb_array = np.random.randint(2, size=X_perturb[test_idx_dic[node_idx]].shape[0])
            X_perturb[test_idx_dic[node_idx]] = perturb_array
        elif mode == 1:
            if random == 0:
                perturb_array = X_perturb[node_idx]
            elif random == 1:
                # 将节点特征设置为所有节点的平均值
                # np.random.uniform：从一个均匀分布[low,high)中随机采样
                perturb_array = np.multiply(
                    X_perturb[test_idx_dic[node_idx]],
                    np.random.uniform(low=0.0, high=2.0, size=X_perturb[test_idx_dic[node_idx]].shape[0])
                )
            X_perturb[test_idx_dic[node_idx]] = perturb_array
        return X_perturb

    def explain(self, node_idx, target, test_idx_dic, num_samples=100, top_node=None, p_threshold=0.05,
                pred_threshold=0.1):

        idx = [k for k, v in test_idx_dic.items() if v < self.test_company_num]

        neighbors, edge_index, mapping, edge_mask = k_hop_subgraph(node_idx, self.num_hops, self.edge_index)

        neighbors = neighbors.cpu().detach().numpy()
        neighbors = np.array([i for i in neighbors if test_idx_dic[i] < self.test_company_num])
        if node_idx not in neighbors:
            neighbors = np.append(neighbors, node_idx)

        company_emb = self.model(hete_graph=self.hete_graph, hyp_graph=self.hyp_graph, idx=idx, x=self.x.float())
        pred_torch = self.classifier.forward(company_emb)
        # pred_torch = res.argmax(dim=1)
        soft_pred = np.asarray([softmax(np.asarray(pred_torch[node_].data.cpu())) for node_ in range(len(idx))])

        # 获得要解释的节点的预测值
        pred_node = np.asarray(pred_torch[test_idx_dic[node_idx]].data.cpu())
        label_node = np.argmax(pred_node)
        soft_pred_node = softmax(pred_node)

        Samples = []
        Pred_Samples = []

        for iteration in range(num_samples):
            # 进行数据生成
            X_perturb = self.x.cpu().detach().numpy()  # 记录节点是否被扰动了
            sample = []
            # 进行扰动
            for node in neighbors:
                seed = np.random.randint(2)
                if seed == 1:
                    latent = 1
                    X_perturb = self.perturb_features_on_node(X_perturb, node, test_idx_dic, random=seed)
                else:
                    latent = 0
                sample.append(latent)
            # X_perturb_torch = torch.tensor(X_perturb, dtype=torch.float).to(self.device)
            X_perturb_torch = torch.tensor(X_perturb, dtype=torch.float)
            company_emb = self.model(hete_graph=self.hete_graph, hyp_graph=self.hyp_graph, idx=idx,
                                     x=X_perturb_torch.float())
            pred_perturb_torch = self.classifier.forward(company_emb)
            # pred_perturb_torch = res.argmax(dim=1)
            soft_pred_perturb = np.asarray(
                [softmax(np.asarray(pred_perturb_torch[node_].data.cpu())) for node_ in range(len(idx))]
            )

            # 进行变量选择
            sample_bool = []
            if target is None:
                target = np.argmax(soft_pred[test_idx_dic[node]])

            for node in neighbors:
                # if (soft_pred_perturb[
                #         test_idx_dic[node], np.argmax(soft_pred[test_idx_dic[node]])] + pred_threshold) < np.max(
                #         soft_pred[test_idx_dic[node]]):
                if (soft_pred_perturb[test_idx_dic[node], target] + pred_threshold) < soft_pred[
                    test_idx_dic[node], target]:
                    sample_bool.append(1)
                else:
                    sample_bool.append(0)

            Samples.append(sample)
            Pred_Samples.append(sample_bool)

        Samples = np.asarray(Samples)
        Pred_Samples = np.asarray(Pred_Samples)
        Combine_Samples = Samples - Samples
        # 依次对每一行进行处理
        for s in range(Samples.shape[0]):
            Combine_Samples[s] = np.asarray(
                [Samples[s, i] * 10 + Pred_Samples[s, i] + 1 for i in range(Samples.shape[1])]
            )

        data_pgm = pd.DataFrame(Combine_Samples)
        data_pgm = data_pgm.rename(columns={0: "A", 1: "B"})  # Trick to use chi_square test on first two data columns
        ind_ori_to_sub = dict(zip(neighbors, list(data_pgm.columns)))

        p_values = []
        dependent_neighbors = []
        dependent_neighbors_p_values = []
        for node in neighbors:
            if node == node_idx:
                p = 0  # p<0.05 => we are confident that we can reject the null hypothesis (i.e. the prediction is the same after perturbing the neighbouring node
                # => this neighbour has no influence on the prediction - should not be in the explanation)
            else:
                chi2, p, _ = chi_square(
                    ind_ori_to_sub[node], ind_ori_to_sub[node_idx], [], data_pgm, boolean=False,
                    significance_level=0.05
                )
            # p_values.append(p)
            p_values.append(1 - p)
            if p < p_threshold:
                dependent_neighbors.append(node)
                dependent_neighbors_p_values.append(p)

        # pgm_stats = dict(zip(neighbors, p_values))  # 概率矩阵

        # pgm_stats = torch.ones(self.x.size(0), dtype=torch.float)
        # neighbors_index = np.array([test_idx_dic[i] for i in neighbors])
        # pgm_stats[neighbors_index] = torch.tensor(p_values, dtype=torch.float)

        # p_values.append(1 - p)
        pgm_stats = torch.zeros(self.x.size(0), dtype=torch.float)
        neighbors_index = np.array([test_idx_dic[i] for i in neighbors])
        pgm_stats[neighbors_index] = torch.tensor(p_values, dtype=torch.float)

        node_mask = torch.zeros(self.x.size(), dtype=torch.int)
        dependent_neighbors_index = [test_idx_dic[i] for i in dependent_neighbors]
        node_mask[dependent_neighbors_index] = 1

        # set_random_seed(14)
        return node_mask, pgm_stats
