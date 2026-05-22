""" evaluate_metrics.py
    Compute Fidelity+ and Fidelity- scores.

    fid+ =
    fid- = 1 - 1/N *
"""
import numpy as np
import pandas as pd
from scipy.special import softmax
from sympy import re
# sympy 是一个Python的科学计算库，用一套强大的符号计算体系完成诸如多项式求值、求极限、解方程、求积分、微分方程、级数展开、矩阵运算等等计算问题。
import torch
import torch.nn.functional as F

from utils.model_utils import get_sub_info, gen_attribute_hg


def list_to_dict(preds):
    preds_dict = pd.DataFrame(preds).to_dict("list")
    for key in preds_dict.keys():
        preds_dict[key] = np.array(preds_dict[key])
    return preds_dict


def get_proba(ypred):
    ypred = ypred.reshape(-1, 2)
    yprob = softmax(ypred, axis=1)
    return yprob


def eval_related_pred_nc(model, data, edge_masks, node_feat_masks, list_node_idx, args):
    """ Evaluate related predictions for a single node.

    Args:
        model: trained GNN model
        data: initial data object
        edge_masks: edge masks for the testing node
        node_feat_masks: node features masks for the testing node
        list_node_idx: list of testing nodes

    Returns:
        related_pred: dictionary of related predictions with masked and maskout predictions
    """
    related_preds = []
    x = data['features']
    y = data['labels']
    edge_index = data['edge_index']
    classifier = data['classifier']
    hete_graph = data['hete_graph']
    hyper_graph = data['hyp_graph']
    idx = data['idx']
    test_idx_dic = data['test_idx_dic']

    hyp_graph = []
    for i in ['industry', 'area', 'qualify']:
        hyp_graph += [gen_attribute_hg(3976, hyper_graph[i], X=None)]
    company_emb = model(hete_graph=hete_graph, hyp_graph=hyp_graph, idx=idx, x=x.float())
    ori_ypred = classifier.forward(company_emb).cpu().detach().numpy()
    ori_yprob = get_proba(ori_ypred)

    num_test = args.num_test_final if args.E else 1

    for i in range(num_test):

        node_idx = list_node_idx[i]

        if not args.NF:
            x_masked = x
            x_maskout = x
        else:
            node_feat_mask = torch.Tensor(node_feat_masks[i])
            # if node_feat_mask.dim() == 2:
            #     x_masked = node_feat_mask
            #     x_maskout = (1 - node_feat_mask)
            # else:
            x_masked = x * node_feat_mask
            x_maskout = x * (1 - node_feat_mask)

        if not args.E:
            company_emb = model(hete_graph=hete_graph, hyp_graph=hyp_graph, idx=idx, x=x_masked.float())
            masked_ypred = classifier.forward(company_emb).cpu().detach().numpy()

            company_emb = model(hete_graph=hete_graph, hyp_graph=hyp_graph, idx=idx, x=x_maskout.float())
            maskout_ypred = classifier.forward(company_emb).cpu().detach().numpy()

        else:
            edge_mask = torch.Tensor(edge_masks[i])
            out_edge_mask = edge_mask <= 0
            masked_edge_index = edge_index[:, edge_mask > 0]  # 留下的边
            maskout_edge_index = edge_index[:, out_edge_mask > 0]  # 移除的边
            sub_idx, sub_hete_graph, sub_hyp_graph = get_sub_info(node_idx, masked_edge_index, edge_mask, hete_graph,
                                                                  hyper_graph, test_idx_dic, args.test_company_num)
            sub_idx1 = [test_idx_dic[i] for i in sub_idx]
            x_masked = x_masked[sub_idx1]
            company_emb = model(hete_graph=sub_hete_graph, hyp_graph=hyp_graph, idx=sub_idx, x=x_masked.float())
            masked_ypred = classifier.forward(company_emb).cpu().detach().numpy()  # 得到的是与要解释的节点相关的边

            sub_idx_out, sub_hete_graph_out, sub_hyp_graph_out = get_sub_info(node_idx, maskout_edge_index,
                                                                              out_edge_mask, hete_graph, hyper_graph,
                                                                              test_idx_dic, args.test_company_num)
            sub_idx_out1 = [test_idx_dic[i] for i in sub_idx_out]
            x_maskout = x_maskout[sub_idx_out1]
            # company_emb = model(hete_graph=sub_hete_graph_out, hyp_graph=sub_hyp_graph_out, idx=sub_idx_out,
            #                     x=x_maskout.float())
            company_emb = model(hete_graph=sub_hete_graph_out, hyp_graph=hyp_graph, idx=sub_idx_out,
                                x=x_maskout.float())
            maskout_ypred = classifier.forward(company_emb).cpu().detach().numpy()

        ori_probs = ori_yprob[test_idx_dic[node_idx]]

        masked_yprob = get_proba(masked_ypred)
        masked_probs = masked_yprob[sub_idx.index(node_idx)]

        maskout_yprob = get_proba(maskout_ypred)
        maskout_probs = maskout_yprob[sub_idx_out.index(node_idx)]

        true_label = y[test_idx_dic[node_idx]].cpu().numpy()
        pred_label = np.argmax(ori_probs)

        # assert true_label == pred_label, "The label predicted by the GCN does not match the true label."\
        related_preds.append(
            {
                "node_idx": node_idx,
                "masked": masked_probs,
                "maskout": maskout_probs,
                "origin": ori_probs,
                "true_label": true_label,
                "pred_label": pred_label,
            }
        )

    related_preds = list_to_dict(related_preds)
    return related_preds


def fidelity_acc(related_preds):
    labels = related_preds["true_label"]  # 真实的标签
    ori_labels = np.argmax(related_preds["origin"], axis=1)  # 初始预测的标签
    unimportant_labels = np.argmax(related_preds["maskout"], axis=1)  # maskout：删去该子图
    p_1 = np.array(ori_labels == labels).astype(int)  # 初始预测的标签与真实标签之间的差距
    p_2 = np.array(unimportant_labels == labels).astype(int)  #
    drop_probability = np.abs(p_1 - p_2)
    return drop_probability.mean().item()


def fidelity_acc_inv(related_preds):
    labels = related_preds["true_label"]
    ori_labels = np.argmax(related_preds["origin"], axis=1)
    important_labels = np.argmax(related_preds["masked"], axis=1)
    p_1 = np.array([ori_labels == labels]).astype(int)
    p_2 = np.array([important_labels == labels]).astype(int)
    drop_probability = np.abs(p_1 - p_2)
    return drop_probability.mean().item()


def fidelity_gnn_acc(related_preds):
    labels = related_preds["pred_label"]
    ori_labels = np.argmax(related_preds["origin"], axis=1)
    unimportant_labels = np.argmax(related_preds["maskout"], axis=1)
    p_1 = np.array(ori_labels == labels).astype(int)
    p_2 = np.array(unimportant_labels == labels).astype(int)
    drop_probability = np.abs(p_1 - p_2)
    return drop_probability.mean().item()


def fidelity_gnn_acc_inv(related_preds):
    labels = related_preds["pred_label"]
    ori_labels = np.argmax(related_preds["origin"], axis=1)
    important_labels = np.argmax(related_preds["masked"], axis=1)
    p_1 = np.array([ori_labels == labels]).astype(int)
    p_2 = np.array([important_labels == labels]).astype(int)
    drop_probability = np.abs(p_1 - p_2)
    return drop_probability.mean().item()


# Fidelity+  metric  studies  the  prediction  change  by
# removing  important  nodes/edges/node  features.
# Higher fidelity+ value indicates good explanations -->1
def fidelity_prob(related_preds):
    labels = related_preds["true_label"]
    ori_probs = np.choose(labels, related_preds["origin"].T)
    unimportant_probs = np.choose(labels, related_preds["maskout"].T)
    drop_probability = np.abs(ori_probs - unimportant_probs)
    return drop_probability.mean().item()


# Fidelity-  metric  studies  the  prediction  change  by
# removing  unimportant  nodes/edges/node  features.
# Lower fidelity- value indicates good explanations -->0
def fidelity_prob_inv(related_preds):
    labels = related_preds["true_label"]
    ori_probs = np.choose(labels, related_preds["origin"].T)
    important_probs = np.choose(labels, related_preds["masked"].T)
    drop_probability = np.abs(ori_probs - important_probs)
    return drop_probability.mean().item()


# Fidelity+  metric  studies  the  prediction  change  by
# removing  important  nodes/edges/node  features.
# Higher fidelity+ value indicates good explanations -->1
def fidelity_gnn_prob(related_preds):
    labels = related_preds["pred_label"]  # 获得预测的标签
    ori_probs = np.choose(labels, related_preds["origin"].T)  # 根据预测的标签来获得对应的初始预测的prob
    unimportant_probs = np.choose(labels, related_preds["maskout"].T)  # 根据预测的标签来获得对应的maskout预测的prob
    drop_probability = np.abs(ori_probs - unimportant_probs)  # 相减获得prob
    return drop_probability.mean().item()


# Fidelity-  metric  studies  the  prediction  change  by
# removing  unimportant  nodes/edges/node  features.
# Lower fidelity- value indicates good explanations -->0
def fidelity_gnn_prob_inv(related_preds):
    labels = related_preds["pred_label"]  # 获得预测的标签
    ori_probs = np.choose(labels, related_preds["origin"].T)  # 根据预测的标签来获得对应的初始预测的prob
    important_probs = np.choose(labels, related_preds["masked"].T)  # 根据预测的标签来获得对应的masked预测的prob
    drop_probability = np.abs(ori_probs - important_probs)
    return drop_probability.mean().item()


def characterization_score(pos_fidelity, neg_fidelity):
    pos_weight = 0.5
    neg_weight = 0.5
    if pos_fidelity == 0 or neg_fidelity == 1:
        result = None
    else:
        denom = (pos_weight / pos_fidelity) + (neg_weight / (1. - neg_fidelity))
        result = 1. / denom
    return result


def unfaithfulness(related_preds):
    ori_probs = torch.tensor(related_preds["origin"])
    important_probs = torch.tensor(related_preds["masked"])
    result = 1 - torch.exp(-F.kl_div(ori_probs.log(), important_probs, reduction='batchmean')).item()
    return result


def eval_fidelity(related_preds, args):
    if eval(args.true_label_as_target):
        fidelity_acc_result = fidelity_acc(related_preds)
        fidelity_acc_inv_result = fidelity_acc_inv(related_preds)
        fidelity_prob_result = fidelity_prob(related_preds)
        fidelity_prob_inv_result = fidelity_prob_inv(related_preds)
    else:
        fidelity_acc_result = fidelity_gnn_acc(related_preds)
        fidelity_acc_inv_result = fidelity_gnn_acc_inv(related_preds)
        fidelity_prob_result = fidelity_gnn_prob(related_preds)
        fidelity_prob_inv_result = fidelity_gnn_prob_inv(related_preds)
    fidelity_scores = {
        "fidelity_gnn_acc+": round(fidelity_acc_result, 4),
        "fidelity_gnn_acc-": round(fidelity_acc_inv_result, 4),
        "fidelity_gnn_acc_cs": characterization_score(fidelity_acc_result, fidelity_acc_inv_result),
        "fidelity_gnn_prob+": round(fidelity_prob_result, 4),
        "fidelity_gnn_prob-": round(fidelity_prob_inv_result, 4),
        "fidelity_gnn_prob_cs": characterization_score(fidelity_prob_result, fidelity_prob_inv_result),
    }
    return fidelity_scores


def eval_unfaithfulness(related_preds):
    unfaithfulness_result = unfaithfulness(related_preds)
    unfaithfulness_scores = {"unfaithfulness": round(unfaithfulness_result, 4)}
    return unfaithfulness_scores

# def fidelity_curve_auc(pos_fidelity, neg_fidelity):
#     if torch.any(neg_fidelity == 1):
#         result = None
#     else:
#         result = pos_fidelity / (1. - neg_fidelity)
#     return result
