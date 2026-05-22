# coding: utf-8
import argparse
import pickle,time

from utils.parser_utils import arg_parse
from utils.model_utils import *
from utils.explain_utils import *
from explainer.node_explainer import *

import warnings

warnings.filterwarnings("ignore")


def explain_by_explainer(list_test_nodes, explainer_name, want_type, args, mode):
    '''
        Explain
    '''
    # set_random_seed(14)
    # args.explain_epoch = 200
    args.explainer_name = explainer_name
    non_count_nodes = []
    edge_masks, node_feat_masks = [], []

    explain_function = eval("explain_" + args.explainer_name + "_node")
    print('Train model: {0}, Explainer: {1}, Explain node: {2}'.format(args.conv_name, args.explainer_name,
                                                                       list_test_nodes))
    if mode == 'train':
        if os.path.exists('./explain_model_save/cf3_train_%s.pkl' % args.conv_name):
            os.remove('./explain_model_save/cf3_train_%s.pkl' % args.conv_name)
    # 以batch的形式跑
    # for node in list_test_nodes:
    #     edge_mask, node_feat_mask = explain_function(gnn, data, [node], want_type, args)
    #     edge_masks.append(edge_mask)
    #     node_feat_masks.append(node_feat_mask)

    # 记录开始时间
    start_time = time.time()
    edge_masks, node_feat_masks = explain_function(gnn, data, list_test_nodes, want_type, args)

    # 记录结束时间
    end_time = time.time()
    # 计算运行时间
    alltime = (end_time - start_time)
    elapsed_time = (end_time - start_time) / len(list_test_nodes)

    print(f"Algorithm took {elapsed_time} seconds to run for per instance and took {alltime} seconds.")

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
    args.clip = 1e-5
    args.conv_name = 'model'
    args.seed = 123
    set_random_seed(args.seed)

    # args.device = torch.device("cpu")
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    total_company_num = 3976
    person_num = 2405

    court_type = 4
    category = 4
    cause_type_num = 11

    args.test_company_num = 2816
    # test_company_num = 474
    args.num = total_company_num + person_num

    data_path = './data/'
    test_data = pd.read_pickle('%strain_data.pkl' % data_path)
    # test_data = pd.read_pickle('%stest_data.pkl' % data_path)
    split_data_idx = pd.read_pickle('%ssplit_data_idx.pkl' % data_path)

    test_sxjl_data, test_xzxk_data, test_zdgz_data, test_risk_data, test_company_attr, test_hete_graph, test_hyper_graph, test_label = test_data
    train_idx, _, _ = split_data_idx
    test_idx = train_idx
    x_test = initialize_company_info(test_sxjl_data, test_zdgz_data, test_xzxk_data, test_risk_data,
                                     test_company_attr, args.test_company_num, cause_type_num, court_type, category,
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
    gnn, classifier = torch.load('./model_save/%s.pkl' % args.conv_name)
    data['classifier'] = classifier
    data['labels'] = torch.LongTensor(test_label)
    data['features'] = torch.LongTensor(x_test)
    data['edge_index'] = test_edge_index
    data['hete_graph'] = test_hete_graph
    data['hyp_graph'] = test_hyper_graph
    data['idx'] = test_idx
    data['test_idx_dic'] = test_idx_dic

    # 20种关系
    # want_type = {0: [0, 2, 4, 9], 1: [1, 3, 5, 8], 2: [6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}

    want_type = {0: [0, 1], 1: [2, 3], 2: [4, 5], 3: [6, 7], 4: [8, 9], 5: [10, 11], 6: [12, 13], 7: [14, 15],
                 8: [16, 17], 9: [18, 19]}
    #
    # want_type = {0: [0, 1, 2, 3, 4, 5, 8, 9], 1: [6, 7, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]}
    #
    # want_type = {i: [i] for i in range(20)}

    args.params_list = '25'
    explainer_name = 'cf3'
    args.explain_train_epoch = 20
    args.lr = 0.012

    # 按照 8:2 划分训练测试集
    random.shuffle(test_idx)
    ttrain_samples = int(args.test_company_num * 0.8)
    ttrain_nodes = test_idx[:ttrain_samples]
    ttest_nodes = test_idx[ttrain_samples:]
    args.batch_size = 128
    # train
    explain_by_explainer(ttrain_nodes, explainer_name, want_type, args, 'train')
    # test
    explain_by_explainer(ttest_nodes, explainer_name, want_type, args, 'test')
    print('ok')
