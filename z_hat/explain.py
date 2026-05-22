# coding: utf-8
import argparse
import pickle, os
# from torch_geometric.utils import softmax


from utils.parser_utils import arg_parse
from utils.model_utils import *
from utils.explain_utils import *
from explainer.node_explainer import *

import warnings

warnings.filterwarnings("ignore")


def explain_by_explainer(list_test_nodes, explainer_name, want_type, args):
    '''
        Explain
    '''
    args.explain_epoch = 200  # 200
    args.explainer_name = explainer_name
    non_count_nodes = []

    explain_function = eval("explain_" + args.explainer_name + "_node")
    edge_masks, node_feat_masks = [], []
    print(
        'Train model: {0}, Explainer: {1}, Explain node: {2}'.format(args.conv_name, args.explainer_name,
                                                                     list_test_nodes))
    for node_idx in list_test_nodes:
        edge_mask, node_feat_mask = explain_function(gnn, data, node_idx, want_type, args)
        if edge_mask is None:
            non_count_nodes.append(node_idx)
            # break
        edge_masks.append(edge_mask)
        node_feat_masks.append(node_feat_mask)
    # print(edge_masks)
    # results_path = "./result_masks/" + args.conv_name
    # if not os.path.exists(results_path):
    #     os.makedirs(results_path)
    # with open("./result_masks/" + args.conv_name + "/" + args.explainer_name + "_type" + str(
    #         len(want_type)) + "_params" + str(args.params_list) + "_result_masks.pkl", 'wb') as f:
    #     pickle.dump([edge_masks, node_feat_masks], f)
    '''
        Evaluate Explain
    '''
    count_nodes = [item for item in list_test_nodes if item not in non_count_nodes]
    edge_masks = [edge_mask for edge_mask in edge_masks if edge_mask is not None]
    print("edge masks' length: {0}, nodes' length: {1}".format(len(edge_masks), len(count_nodes)))
    if len(count_nodes) > 0:
        evaluate_explain(gnn, data, edge_masks, node_feat_masks, count_nodes, want_type, args)


if __name__ == "__main__":

    args = arg_parse()
    args.lr = 2e-3
    args.clip = 1e-5  # 1e-5
    args.conv_name = 'hat'
    args.seed = 123
    set_random_seed(args.seed)  # 14

    device = torch.device("cpu")
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    total_company_num = 3976
    person_num = 2405

    court_type = 4
    category = 4
    cause_type_num = 11

    test_company_num = 474
    args.num = total_company_num + person_num

    data_path = '../data/'
    test_data = pd.read_pickle('%stest_data.pkl' % data_path)
    split_data_idx = pd.read_pickle('%ssplit_data_idx.pkl' % data_path)

    test_sxjl_data, test_xzxk_data, test_zdgz_data, test_risk_data, test_company_attr, test_hete_graph, test_hyper_graph, test_label = test_data
    _, _, test_idx = split_data_idx
    x_test = initialize_company_info(test_sxjl_data, test_zdgz_data, test_xzxk_data, test_risk_data,
                                     test_company_attr, test_company_num, cause_type_num, court_type, category,
                                     test_idx)
    test_edge_index = get_edge_index(test_hete_graph)

    test_idx_dic = {v: i for i, v in enumerate(test_idx)}
    idx = list(np.sort(np.unique(test_edge_index)))
    for i in idx:
        if i not in test_idx_dic.keys():
            test_idx_dic[i] = len(test_idx_dic)

    '''
        Explain
    '''

    data = dict()
    gnn, classifier = torch.load('../model_save/%s.pkl' % args.conv_name)
    data['classifier'] = classifier
    data['labels'] = torch.LongTensor(test_label)
    data['features'] = torch.LongTensor(x_test)
    data['edge_index'] = test_edge_index
    data['hete_graph'] = test_hete_graph
    data['hyp_graph'] = test_hyper_graph
    data['idx'] = test_idx
    data['test_idx_dic'] = test_idx_dic

    nums = [50]  # 20
    # nums = [10]  # 20

    # 20种关系
    # want_type = {0: [0, 2, 4, 9], 1: [1, 3, 5, 8], 2: [6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}
    want_type = {0: [0, 1, 2, 3, 4, 5, 8, 9], 1: [6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}
    # want_type = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9], 5: [10, 11], 6: [12, 16], 7: [13, 17],
    #              8: [14, 18], 9: [15, 19]}

    args.params_list = '30'

    # explainer_list = ['hetegnnexplainer','gnnexplainer','pgmexplainer',
    # 'pgexplainer','subgraphx','dnx','cfgnnexplainer','cf2']
    # 最好的是 pgmexplainer 和 subgraphx 单独运行
    # cfgnnexplainer 最后没有得到反事实解释的结果，故不考虑此模型
    # explainer_list = ['cfgnnexplainer', 'gnnexplainer']
    explainer_list = ['hetegnnexplainer']

    nodes = test_idx
    for n in nums:
        list_test_nodes = get_test_nodes(n, nodes)
        # list_test_nodes = [3378]
        # list_test_nodes = [492]
        # list_test_nodes = [589]
        # list_test_nodes = [492]
        args.all_len = len(list_test_nodes)
        data['list_test_nodes'] = list_test_nodes
        for j in explainer_list:
            explain_by_explainer(list_test_nodes, j, want_type, args)
