
# from ablation_hetegnnexplainer_no_feature import HETEGNNExplainer     # individual_feature
# from ablation_hetegnnexplainer_no_edge import HETEGNNExplainer
# from hetegnnexplainer import HETEGNNExplainer

# from ablation_hetegnnexplainer_no_feature import HETEGNNExplainer     # individual_feature
# from ablation_hetegnnexplainer_no_edge_1 import HETEGNNExplainer
from hetegnnexplainer_1 import HETEGNNExplainer


def explain_hetegnnexplainer_node(model, data, node_idx, want_type, args):
    target = data['labels'][data['test_idx_dic'][node_idx]]
    # target = None
    gnn_exlainer = HETEGNNExplainer(model=model,
                                    epochs=args.explain_epoch,
                                    lr=args.lr,
                                    num_hops=2,
                                    feat_mask_type='feature',
                                    num=args.num)
    # 此处的node_feat_mask是该节点加上邻居节点的结果
    node_feat_mask, edge_mask = gnn_exlainer.explain_node(int(node_idx), data['features'], data['edge_index'],
                                                          data['classifier'], data['hete_graph'], want_type,
                                                          data['test_idx_dic'], data['hyp_graph'], target)
    if edge_mask is not None:
        edge_mask = edge_mask.cpu().detach().numpy()
    if node_feat_mask is not None:
        node_feat_mask = node_feat_mask.cpu().detach().numpy()
    return edge_mask.astype("float"), node_feat_mask.astype("float")
