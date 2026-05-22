import numpy as np
import pandas as pd
import json
from scipy.stats import entropy, gaussian_kde
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
from scipy.sparse.csgraph import connected_components
from torch_geometric.utils import to_scipy_sparse_matrix

from utils.evaluate_metrics import eval_related_pred_nc, eval_fidelity, eval_unfaithfulness
from utils.model_utils import set_random_seed

set_random_seed(14)


def get_labels(ypred):
    ylabels = np.argmax(ypred, axis=1)
    return ylabels


def get_test_nodes(num, list_node_idx):
    list_test_nodes = [x.item() for x in
                       np.random.choice(list_node_idx, size=min(num, len(list_node_idx)), replace=False)]
    # if x.item() not in [3898]
    return list_test_nodes


def normalize_mask(x):
    if (np.nanmax(x) - np.nanmin(x)) == 0:
        return x
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def clean_masks(masks):
    """Clean masks by removing NaN, inf and too small values and normalizing"""
    for i in range(len(masks)):
        if masks[i] is not None:
            masks[i] = np.nan_to_num(masks[i], copy=True, nan=0.0, posinf=10, neginf=-10)
            masks[i] = np.clip(masks[i], -10, 10)
            masks[i] = normalize_mask(masks[i])
            masks[i] = np.where(masks[i] < 0.001, 0, masks[i])
    return masks


def get_sparsity(masks):
    sparsity = 0
    for i in range(len(masks)):
        sparsity += 1.0 - (masks[i] != 0).sum() / len(masks[i])
    return sparsity / len(masks)


def get_size(masks):
    size = 0
    for i in range(len(masks)):
        size += (masks[i] != 0).sum()  # 总的不为0的边的数量（子图的边的数量之和）
    return size / len(masks)  # 平均下来每个子图的边的数量


def get_entropy(masks):
    ent = 0
    k = 0
    for i in range(len(masks)):
        pos_mask = masks[i][masks[i] > 0]
        if len(pos_mask) == 0:
            continue
        ent += entropy(pos_mask)
        k += 1
    if k == 0:
        return -1
    return ent / k


def get_avg_max(masks):
    max_avg = 0
    k = 0
    for i in range(len(masks)):
        pos_mask = masks[i][masks[i] > 0]
        if len(pos_mask) == 0:
            continue
        # kde = gaussian_kde(np.array(pos_mask))
        # density = kde(pos_mask)
        # index = np.argmax(density)
        ys, xs, _ = plt.hist(pos_mask, bins=100)
        index = np.argmax(ys)
        max_avg += xs[index]
        k += 1
    if k == 0:
        return -1
    return max_avg / k


def get_ratio_connected_components(edge_masks, edge_index):
    """Compute connected components ratio of the edge mask."""
    cc_ratio = []
    for i in range(len(edge_masks)):
        edge_mask = edge_masks[i]
        masked_edge_index = edge_index[:, edge_mask > 0]
        if masked_edge_index.size(1) == 0:
            return None
        masked_edge_index = masked_edge_index.cpu().numpy()
        lst = np.sort(np.unique(masked_edge_index))
        d = {lst[i]: i for i in range(len(lst))}
        indexer = np.array([d.get(i, -1) for i in range(masked_edge_index.min(), masked_edge_index.max() + 1)])
        relabel_masked_edge_index = indexer[(masked_edge_index - masked_edge_index.min())]
        sparse_adj = to_scipy_sparse_matrix(torch.LongTensor(relabel_masked_edge_index)).toarray()
        n_components, labels = connected_components(csgraph=sparse_adj, directed=False, return_labels=True)
        cc_ratio.append(n_components / len(labels))
    return np.mean(cc_ratio)


def get_mask_info(masks, edge_index):
    mask_info = {'mask_size': get_size(masks), 'mask_entropy': get_entropy(masks), 'max_avg': get_avg_max(masks),
                 'cc_ratio': get_ratio_connected_components(masks, edge_index)}
    return mask_info


def topk_edges_unique(edge_mask, edge_index, num_top_edges):
    """Return the indices of the top-k edges in the mask.

    Args:
        edge_mask (Tensor): edge mask of shape (num_edges,).
        edge_index (Tensor): edge index tensor of shape (2, num_edges)
        num_top_edges (int): number of top edges to be kept
    """
    indices = (-edge_mask).argsort()
    top = np.array([], dtype="int")
    i = 0
    list_edges = np.sort(edge_index.cpu().T, axis=1)
    while len(top) < num_top_edges:
        subset = indices[num_top_edges * i: num_top_edges * (i + 1)]
        topk_edges = list_edges[subset]
        u, idx = np.unique(topk_edges, return_index=True, axis=0)
        top = np.concatenate([top, subset[idx]])
        i += 1
    return top[:num_top_edges]


def mask_to_shape(mask, edge_index, num_top_edges):
    """Modify the mask by selecting only the num_top_edges edges with the highest mask value."""
    indices = topk_edges_unique(mask, edge_index, num_top_edges)
    unimportant_indices = [i for i in range(len(mask)) if i not in indices]
    new_mask = mask.clone()
    new_mask[unimportant_indices] = 0
    return new_mask


def control_sparsity(mask, sparsity):
    r"""
    :param edge_mask: mask that need to transform
    :param sparsity: sparsity we need to control i.e. 0.7, 0.5
    :return: transformed mask where top 1 - sparsity values are set to inf.
    """
    mask_len = len(mask)
    split_point = int((1 - sparsity) * mask_len)
    unimportant_indices = (-mask).argsort()[split_point:]
    mask[unimportant_indices] = 0
    return mask


def transform_mask(masks, data, param, args):
    """Transform masks according to the given strategy (topk, threshold, sparsity) and level."""
    new_masks = []
    for mask_ori in masks:
        mask = mask_ori.copy()
        if args.strategy == 'topk':
            if eval(args.directed):
                unimportant_indices = (-mask).argsort()[param + 1:]
                mask[unimportant_indices] = 0
            else:
                mask = mask_to_shape(mask, data.edge_index, param)
                # indices = np.where(mask > 0)[0]
        if args.strategy == 'sparsity':
            mask = control_sparsity(mask, param)
        if args.strategy == 'threshold':
            mask = np.where(mask > param, mask, 0)
        new_masks.append(mask)
    return np.array(new_masks, dtype=np.float64)


def plot_masks_density(masks, args, type="edge"):
    pal = sns.color_palette("tab10")
    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)

    # Plotting avg density
    # edge_values = np.array(edge_masks).reshape(-1)
    # positive_edge_values = edge_values[edge_values>0]
    max_len = 0
    for i in range(5):
        mask = masks[i]
        pos_mask = mask[mask > 0]
        max_len = max_len if max_len > len(pos_mask) else len(pos_mask)
        sns.histplot(pos_mask, kde=True, color=pal[i], alpha=0.4, ax=ax)
    plt.xlim(0, 1)
    plt.ylim(0, max_len)
    plt.title(
        f"Density of 5 {type} masks for {args.explainer_name}, target as true label = {args.true_label_as_target}, sparsity = {args.sparsity}"
    )
    plt.xlabel(f"{type} importance")
    # print(gen_mask_density_plt_name(args, type))
    # plt.savefig(gen_mask_density_plt_name(args, type), dpi=600)
    plt.close()
    matplotlib.style.use("default")


def plot_feat_importance(node_feat_masks, args):
    pal = sns.color_palette("tab10")
    matplotlib.style.use("seaborn")
    plt.switch_backend("agg")
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5)

    for i in range(5):
        node_feat_mask = node_feat_masks[i]

    sns.barplot(range(0, len(node_feat_mask)), node_feat_mask, ax=ax)
    plt.xlim(0, 1)
    plt.title(f"Node feature importance for {args.explainer_name}")
    plt.xlabel("Node feature importance")
    # plt.savefig(gen_feat_importance_plt_name(args), dpi=600)
    plt.close()
    matplotlib.style.use("default")


def evaluate_explain(model, data, edge_masks, node_feat_masks, list_test_nodes, want_type, args):
    args.E = False if edge_masks[0] is None else True
    args.NF = False if node_feat_masks[0] is None else True
    args.num_test_final = len(edge_masks) if args.E else None
    # args.NF = False

    infos = {
        "train_model_single": args.conv_name,
        "explainer": args.explainer_name,
        "number_of_edges": data['edge_index'].size(1),
        "num_test_final": args.num_test_final, }

    if args.E:
        ### Mask normalisation and cleaning ###
        edge_masks = [edge_mask.astype("float") for edge_mask in edge_masks]
        edge_masks = clean_masks(edge_masks)
        print("__initial_edge_mask_infos:" + json.dumps(get_mask_info(edge_masks, data['edge_index'])))

        infos["edge_mask_sparsity_init"] = get_sparsity(edge_masks)
        infos["edge_mask_size_init"] = get_size(edge_masks)
        infos['edge_mask_entropy_init'] = get_entropy(edge_masks)
        infos['edge_mask_avg_max_init'] = get_avg_max(edge_masks)
        infos["edge_mask_connected_init"] = get_ratio_connected_components(edge_masks, data['edge_index'])

    if args.NF:
        ### Mask normalisation and cleaning ###
        node_feat_masks = [node_feat_mask.astype("float") for node_feat_mask in node_feat_masks]
        node_feat_masks = clean_masks(node_feat_masks)
        # infos["node_feat_mask_sparsity_init"] = get_sparsity(node_feat_masks)
        # infos["node_feat_mask_size_init"] = get_size(node_feat_masks)
        if eval(args.hard_mask) == False:
            plot_masks_density(node_feat_masks, args, type="node_feat")
            plot_feat_importance(node_feat_masks, args)

    print("__infos:" + json.dumps(infos))

    args.true_label_as_target = 'True'
    print("Masks are transformed with strategy: " + args.strategy)
    params_lst = [i for i in args.params_list.split(',')]
    if '%' in params_lst[0]:
        params_lst = [eval(i.split('%')[0]) / 100 for i in params_lst]
    else:
        params_lst = [eval(i) for i in params_lst]
    edge_masks_ori = edge_masks.copy()

    for param in params_lst:  # k=10条边是一个合适的解释规模
        if type(param) == float:
            param_values = int(int(infos["edge_mask_size_init"]) * param)
        else:
            param_values = param
        params_transf = {args.strategy: param, args.strategy + "_values": param_values}
        ### Mask transformation ###
        edge_masks = transform_mask(edge_masks_ori, data, param_values, args)  # 进行边的数量的选择：param
        if not eval(args.hard_mask):
            plot_masks_density(edge_masks, args, type="edge")
        # transformed_mask_infos = {key: value for key, value in sorted(
        #     get_mask_info(edge_masks, data['edge_index']).items() | params_transf.items())}
        # print("__transformed_mask_infos:" + json.dumps(transformed_mask_infos))
        related_preds = eval_related_pred_nc(model, data, edge_masks, node_feat_masks, list_test_nodes, args)
        ### Fidelity ###
        fidelity = eval_fidelity(related_preds, args)
        fidelity_scores = {key: value for key, value in sorted(fidelity.items() | params_transf.items())}
        print("__fidelity:" + json.dumps(fidelity_scores))
        ### unfaithfulness ###
        unfaithfulness = eval_unfaithfulness(related_preds)
        unfaithfulness_scores = {key: value for key, value in sorted(unfaithfulness.items())}
        print("__unfaithfulness:" + json.dumps(unfaithfulness_scores))

        others = {"seed": args.seed, "lr": args.lr}
        print("__others:" + json.dumps(others))
        scores = {
            key: value
            for key, value in sorted(
                infos.items()
                | fidelity.items()
                | unfaithfulness.items()
                | params_transf.items()
                | others.items()
            )
        }
        results = list(scores.values())

    ### Save results ###
    save_path = os.path.join("./explain_results", args.conv_name)
    os.makedirs(save_path, exist_ok=True)
    result_file = os.path.join(
        save_path,
        "results_{}_{}_type{}_batch{}_params{}.csv".format(
            args.conv_name,
            args.explainer_name,
            str(len(want_type)),
            str(args.batch_size),
            str(args.params_list),
        ),
    )
    if not os.path.exists(result_file):
        title = pd.DataFrame(columns=(scores.keys()))
        title.to_csv(result_file, index=False, mode='a', encoding='utf8')

    results_pd = pd.DataFrame(columns=(results))
    results_pd.to_csv(result_file, index=False, mode='a', encoding='utf8')
    #
    # return fidelity_scores, unfaithfulness_scores
