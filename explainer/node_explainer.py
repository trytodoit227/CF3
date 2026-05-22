import os
import numpy as np
import torch
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, k_hop_subgraph, to_dense_adj, subgraph

from explainer.cf2 import NodeExplainerEdgeMulti
from explainer.gnnexplainer import GNNExplainer
# from explainer.pgmexplainer import PGMExplainer
from explainer.pgexplainer import PGExplainer
from explainer.subgraphx import SubgraphX
from explainer.cfgnnexplainer import CFExplainer
from explainer.dnx import sgc_train, dnx_explain

# from explainer.cf3 import CF3
# from explainer.cf3_batch import CF3
from explainer.cf3_batch_1 import CF3

from utils.model_utils import get_sub_info


def node_attr_to_edge(edge_index, node_mask, t):
    edge_mask = np.zeros(edge_index.shape[1])
    nm = np.zeros(len(t))
    # nonzero = np.array([t[i] for i in node_mask.nonzero()[0]])  # pagerank
    nonzero = np.array(node_mask.nonzero())  # pgmexplainer
    nm[nonzero] = node_mask[node_mask.nonzero()]
    e0 = np.array([t[i] for i in edge_index[0].cpu().numpy()])
    e1 = np.array([t[i] for i in edge_index[1].cpu().numpy()])
    edge_mask += nm[e0]
    edge_mask += nm[e1]
    return edge_mask


def filter_existing_edges(perturb_edges, edge_index):
    """Returns a mask of edges that are perturbed by CF-GNNExplainer and also exist in the original graph
    Args:
        perturb_edges (_type_): counterfactual explanations, i.e. edges that are perturbed by CF-GNNExplainer
        edge_index (_type_): edge index of the original graph
    Returns:
        _type_: edge mask with value 1 if the edge exists.
    """
    edge_mask = np.zeros(edge_index.shape[1])
    list_tuples = zip(*perturb_edges)
    for i in range(edge_index.shape[1]):
        # if include_edges is not None and not include_edges[i].item():
        # continue
        u, v = list(edge_index[:, i])
        if (u, v) in list_tuples:
            edge_mask[i] = 1
    return edge_mask


def get_neighbourhood(node_idx, edge_index, n_hops, features, labels, test_idx_dic, test_company_num):
    edge_subset = k_hop_subgraph(node_idx, n_hops, edge_index)  # Get all nodes involved
    edge_subset_relabel = subgraph(edge_subset[0], edge_index, edge_attr=None,
                                   relabel_nodes=False)  # Get relabelled subset of edges
    sub_adj = to_dense_adj(edge_subset_relabel[0]).squeeze()
    sa_index = torch.tensor(
        [i.item() for i in edge_subset[0] if i.item() < 3976 and test_idx_dic[i.item()] < test_company_num])
    sa_index_row = sa_index.unsqueeze(1).expand(-1, sa_index.shape[0])
    sa_index_col = sa_index.unsqueeze(0).expand(sa_index.shape[0], -1)
    sub_adj = sub_adj[sa_index_row, sa_index_col]
    es = torch.tensor(
        [test_idx_dic[i.item()] for i in edge_subset[0] if
         i.item() < 3976 and test_idx_dic[i.item()] < test_company_num])
    sub_feat = features[es, :]
    sub_labels = labels[es]
    new_index = np.array([i for i in range(len(es))])
    node_dict = dict(zip(edge_subset[0].numpy(), new_index))  # Maps orig labels to new
    # print("Num nodes in subgraph: {}".format(len(edge_subset[0])))
    # if model_name == 'cfgnnexplainer':
    # edge_mask = (edge_subset[1] < 3976).all(dim=0)
    # edge_index = edge_subset[1][:, edge_mask]
    edge_mask = edge_subset[3]
    edge_index = edge_subset[1]

    s = torch.tensor([i.item() for i in edge_subset[0] if
                      i.item() < 3976 and i.item() in test_idx_dic.keys() and test_idx_dic[
                          i.item()] < test_company_num])
    mapping = {int(v): k for k, v in enumerate(s)}
    return sub_adj, sub_feat, sub_labels, node_dict, edge_index, edge_mask, mapping


"""---------------------methods--------------------------------"""


def explain_random_node(model, data, node_idx, want_type, args):
    edge_mask = np.random.uniform(size=data['edge_index'].shape[1])
    node_feat_mask = np.random.uniform(size=data['features'].shape[1])
    return edge_mask.astype("float"), node_feat_mask.astype("float")


def explain_distance_node(model, data, node_idx, want_type, args):
    data = Data(x=data['features'], edge_index=data['edge_index'])
    g = to_networkx(data)
    length = nx.shortest_path_length(g, target=node_idx)

    def get_attr(node):
        if node in length:
            return 1 / (length[node] + 1)
        return 0

    edge_sources = data.edge_index[1].cpu().numpy()
    return np.array([get_attr(node) for node in edge_sources]).astype("float"), None


def explain_pagerank_node(model, data, node_idx, want_type, args):
    test_idx_dic = data['test_idx_dic']
    data = Data(x=data['features'], edge_index=data['edge_index'])
    g = to_networkx(data)
    pagerank = nx.pagerank(g, personalization={node_idx: 1})

    node_attr = np.zeros(max(pagerank.keys()) + 1)
    for node, value in pagerank.items():
        node_attr[node] = value
    edge_mask = node_attr_to_edge(data.edge_index, node_attr, test_idx_dic)
    return edge_mask.astype("float"), None


def explain_gnnexplainer_node(model, data, node_idx, want_type, args):
    target = data['labels'][data['test_idx_dic'][node_idx]]
    # target = None
    gnn_exlainer = GNNExplainer(model=model,
                                epochs=args.explain_epoch,
                                lr=args.lr,
                                num_hops=2,
                                feat_mask_type='feature',
                                test_company_num=args.test_company_num)
    node_feat_mask, edge_mask = gnn_exlainer.explain_node(int(node_idx), data['features'], data['edge_index'],
                                                          data['classifier'], data['hete_graph'], data['test_idx_dic'],
                                                          data['hyp_graph'], target)
    edge_mask = edge_mask.cpu().detach().numpy()
    node_feat_mask = node_feat_mask.cpu().detach().numpy()
    return edge_mask.astype("float"), node_feat_mask.astype("float")


def explain_pgmexplainer_node(model, data, node_idx, want_type, args):
    target = data['labels'][data['test_idx_dic'][node_idx]]
    explainer = PGMExplainer(
        model,
        data['features'],
        data['edge_index'],
        data['classifier'],
        data['hete_graph'],
        data['hyp_graph'],
        num_hops=2,
        print_result=0,
        test_company_num=args.test_company_num
    )
    explanation = explainer.explain(node_idx, target, data['test_idx_dic'], num_samples=200, top_node=None,
                                    p_threshold=0.05, pred_threshold=0.1)
    node_mask = explanation[0]
    node_attr = explanation[1].cpu().detach().numpy()
    edge_mask = node_attr_to_edge(data['edge_index'], node_attr, data['test_idx_dic'])
    if node_mask is not None:
        node_mask = node_mask.cpu().detach().numpy()
    # return edge_mask.astype("float"), node_mask.astype("float")
    return edge_mask.astype("float"), None


def explain_pgexplainer_node(model, data, node_idx, want_type, args):
    pgexplainer = PGExplainer(
        model=model,
        in_channels=12 * 3,
        device=args.device,
        num_hops=2,
        test_company_num=args.test_company_num
    )
    save_explain_model_path = './explain_model_save'
    if not os.path.exists(save_explain_model_path):
        os.makedirs(save_explain_model_path)
    nodes_len = len(data['list_test_nodes'])
    pgexplainer_saving_path = os.path.join(save_explain_model_path,
                                           f"pgexplainer_nodes_len_{nodes_len}_model_{args.conv_name}.pth")
    if os.path.isfile(pgexplainer_saving_path):
        state_dict = torch.load(pgexplainer_saving_path)
        pgexplainer.load_state_dict(state_dict)
    else:
        pgexplainer.train_explanation_network(data)
        torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
        state_dict = torch.load(pgexplainer_saving_path)
        pgexplainer.load_state_dict(state_dict)
    edge_mask = pgexplainer.explain_node(int(node_idx), data['features'], data['edge_index'], data['classifier'],
                                         data['hete_graph'], data['test_idx_dic'], data['hyp_graph'], data['labels'])
    edge_mask = edge_mask.cpu().detach().numpy()
    return edge_mask.astype("float"), None


def explain_subgraphx_node(model, data, node_idx, want_type, args):
    target = data['labels'][data['test_idx_dic'][node_idx]]
    subgraphx = SubgraphX(
        model=model,
        num_classes=2,
        device=args.device,
        num_hops=2,
        explain_graph=False,
        rollout=20,
        min_atoms=4,
        expand_atoms=14,
        high2low=True,
        sample_num=50,
        reward_method="mc_shapley",
        subgraph_building_method="zero_filling",
        local_radius=4,
        test_company_num=args.test_company_num
    )
    edge_mask = subgraphx.explain(data['features'], data['edge_index'], data['classifier'], data['hete_graph'],
                                  data['test_idx_dic'], data['hyp_graph'], max_nodes=10, label=target,
                                  node_idx=int(node_idx))
    return edge_mask.astype("float"), None


def explain_cfgnnexplainer_node(model, data, node_idx, want_type, args):
    target = data['labels'][data['test_idx_dic'][node_idx]]
    n_momentum, beta, optimizer, lr = 0.9, 0.5, "SGD", 0.01
    features, labels = data['features'], data['labels']
    sub_adj, sub_feat, sub_labels, node_dict, sub_edge_index, edge_mask, mapping = get_neighbourhood(int(node_idx),
                                                                                                     data['edge_index'],
                                                                                                     3, features,
                                                                                                     labels, data[
                                                                                                         'test_idx_dic'],
                                                                                                     args.test_company_num)
    sub_idx, sub_hete_graph, sub_hyp_graph = get_sub_info(node_idx, sub_edge_index, edge_mask, data['hete_graph'],
                                                          data['hyp_graph'], data['test_idx_dic'],
                                                          args.test_company_num)
    new_idx = sub_idx.index(node_idx)
    # Need to instantitate new cf model every time because size of P changes based on size of sub_adj
    explainer = CFExplainer(model=model, cf_model_name='model', adj=sub_adj, feat=sub_feat,
                            hete_graph=sub_hete_graph, hyp_graph=sub_hyp_graph, classifier=data['classifier'],
                            n_hid=12, dropout=0.0, readout='identity', edge_dim=1, num_layers=2, labels=sub_labels,
                            y_pred_orig=target, beta=beta, device=args.device, mapping=mapping)
    cf_example = explainer.explain_node(node_idx=node_idx, cf_optimizer=optimizer, new_idx=new_idx, lr=lr,
                                        n_momentum=n_momentum, num_epochs=args.explain_epoch, )
    if cf_example == []:
        return None, None
    else:
        perturb_edges = cf_example[0][2]
        node_dict_inv = {int(v): int(k) for k, v in node_dict.items()}
        perturb_edges_ori = np.vectorize(node_dict_inv.get)(perturb_edges)
        edge_mask = filter_existing_edges(perturb_edges_ori, data.edge_index)
        return edge_mask.astype("float"), None


def explain_cf2_node(model, data, node_idx, want_type, args):
    target = data['labels'][data['test_idx_dic'][node_idx]]
    features, labels = data['features'], data['labels']
    sub_adj, sub_feat, sub_labels, node_dict, sub_edge_index, \
    hard_edge_mask, mapping = get_neighbourhood(int(node_idx), data['edge_index'], 3, features, labels,
                                                data['test_idx_dic'], args.test_company_num)
    sub_idx, sub_hete_graph, sub_hyp_graph = get_sub_info(node_idx, sub_edge_index, hard_edge_mask, data['hete_graph'],
                                                          data['hyp_graph'], data['test_idx_dic'],
                                                          args.test_company_num)
    new_idx = sub_idx.index(node_idx)
    cf2_expalainer = NodeExplainerEdgeMulti(model=model, idx=sub_idx, x=sub_feat, edge_index=sub_edge_index,
                                            classifier=data['classifier'], hete_graph=sub_hete_graph,
                                            hard_edge_mask=hard_edge_mask, hyp_graph=sub_hyp_graph,
                                            hyper_graph=data['hyp_graph'], epochs=args.explain_epoch, args=args,
                                            device=args.device)
    edge_mask = cf2_expalainer.explain_nodes_gnn_stats(int(new_idx), target, node_idx)
    if edge_mask is not None:
        edge_mask = edge_mask.cpu().detach().numpy()
    return edge_mask.astype("float"), None


def explain_dnx_node(model, data, node_idx, want_type, args):
    save_explain_model_path = './explain_model_save'
    if not os.path.exists(save_explain_model_path):
        os.makedirs(save_explain_model_path)
    dnx_saving_path = os.path.join(save_explain_model_path, f"dnx_sgc_model_model_{args.conv_name}.pth")
    if not os.path.isfile(dnx_saving_path):
        sgc_train(data, model, args)
    sgc_model = torch.load(dnx_saving_path)
    # z_explain_model
    explanation = dnx_explain(node_idx, data, model, sgc_model, args.test_company_num)
    node_attr = explanation[1].cpu().detach().numpy()
    edge_mask = node_attr_to_edge(data['edge_index'], node_attr, data['test_idx_dic'])
    return edge_mask.astype("float"), None


# def explain_hetegnnexplainer_node(model, data, node_idx, want_type, args):
#     target = data['labels'][data['test_idx_dic'][node_idx]]
#     # target = None
#     gnn_explainer = CF3(model=model,
#                         epochs=args.explain_epoch,
#                         temp=args.temp,
#                         lr=args.lr,
#                         num_hops=2,  # 消融实验，改变不同的子图阶数
#                         feat_mask_type='feature',
#                         num=args.num,
#                         x_shape=data['features'].shape[1],
#                         device=args.device,
#                         test_company_num=args.test_company_num)
#     # 此处的node_feat_mask是该节点加上邻居节点的结果
#     node_feat_mask, edge_mask = gnn_explainer.explain_node(int(node_idx), data['features'],
#                                                            data['edge_index'], data['classifier'],
#                                                            data['hete_graph'], want_type,
#                                                            data['test_idx_dic'], data['hyp_graph'],
#                                                            target)
#
#     if edge_mask is not None:
#         edge_mask = edge_mask.cpu().detach().numpy()
#     if node_feat_mask is not None:
#         node_feat_mask = node_feat_mask.cpu().detach().numpy()
#
#     return edge_mask.astype("float"), node_feat_mask.astype("float")


def explain_cf3_node(model, data, node_idx_list, want_type, args):
    target_list = [data['labels'][data['test_idx_dic'][node_idx]] for node_idx in node_idx_list]

    save_explain_model_path = './explain_model_save'
    if not os.path.exists(save_explain_model_path):
        os.makedirs(save_explain_model_path)
    cf3_saving_path = os.path.join(save_explain_model_path, f"cf3_train_%s.pkl" % args.conv_name)
    if not os.path.isfile(cf3_saving_path):
        gnn_explainer = CF3(model=model,
                            epochs=args.explain_train_epoch,
                            temp=args.temp,
                            lr=args.lr,
                            num_hops=2,  # 消融实验，改变不同的子图阶数
                            feat_mask_type='feature',
                            num=args.num,
                            x_shape=data['features'].shape[1],
                            batch_size=args.batch_size,
                            device=args.device,
                            test_company_num=args.test_company_num)
        gnn_explainer.train()
        is_training = True
    else:
        gnn_explainer = torch.load(cf3_saving_path)
        gnn_explainer.eval()
        is_training = False
    # 此处的node_feat_mask是该节点加上邻居节点的结果
    node_feat_mask, edge_mask = gnn_explainer.explain_node(node_idx_list, data['features'],
                                                           data['edge_index'], data['classifier'],
                                                           data['hete_graph'], want_type,
                                                           data['test_idx_dic'], data['hyp_graph'],
                                                           target_list, is_training)
    # 保存模型
    if is_training:
        torch.save(gnn_explainer, cf3_saving_path)

    if None not in edge_mask:
        edge_mask = [i.cpu().detach().numpy() for i in edge_mask]
        edge_mask = [i.astype("float") for i in edge_mask]

    if None not in node_feat_mask:
        node_feat_mask = [i.cpu().detach().numpy() for i in node_feat_mask]
        node_feat_mask = [i.astype("float") for i in node_feat_mask]

    return edge_mask, node_feat_mask
