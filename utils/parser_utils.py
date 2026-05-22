import argparse
# import numpy as np
# import torch


def arg_parse():
    parser = argparse.ArgumentParser(description='Training GNN')
    '''
        Dataset arguments
    '''
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='The address of preprocessed graph.')
    parser.add_argument('--model_dir', type=str, default='.\model_save',
                        help='The address for storing the models and optimization results.')
    parser.add_argument('--cuda', type=int, default=0,
                        help='Avaiable GPU ID')

    '''
       Model arguments
    '''
    parser.add_argument('--conv_name', type=str, default='riskgnn',
                        choices=['gat', 'gcn', 'rgcn', 'hat', 'han', 'hetgnn', 'hgnn', 'hgcn', 'hwnn', 'lr', 'svm',
                                 'dt', 'gbdt', 'riskgnn'],
                        help='The name of GNN filter.')
    parser.add_argument('--input_dim', type=int, default=16,
                        help='Number of input dimension')
    parser.add_argument('--output_dim', type=int, default=12,
                        help='Number of output dimension')
    parser.add_argument('--n_heads', type=int, default=1,
                        help='Number of attention head')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of HeteGAT layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout ratio')

    '''
        Optimization arguments
    '''
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adamw', 'adam', 'sgd', 'adagrad'],
                        help='optimizer to use.')

    parser.add_argument('--n_epoch', type=int, default=200,
                        help='Number of epoch to run')
    parser.add_argument('--mlp_epoch', type=int, default=10,
                        help='Number of mlp model epoch to run')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='Gradient Norm Clipping')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay of adamw')
    parser.add_argument('--temp', type=float, default=1,
                        help='temperature of contrastive learning')
    '''
        Evaluate arguments
    '''
    # 硬掩码：通过设置每个正值为1将掩码转换成硬掩码来对一个非加权的子图进行解释
    parser.add_argument("--hard_mask", type=str, default="True", help="Soft or hard mask")
    # 定义了三种减少解释大小的策略：稀疏度、阈值和topk，它们将解释子图G_{S}转化为更稀疏的版本G_{S}^{t}
    # topk策略：是唯一一个独立于图的大小和解释器方法的最大边数k的策略，即 解释子图中的边的数量
    parser.add_argument("--strategy", type=str, default="topk",
                        help="strategy for mask transformation")  # ["topk", "sparsity", "threshold"]
    parser.add_argument("--params_list", type=str, default="10,15,20,25,30,35,40,45,50",
                        help="list of transformation degrees")
    parser.add_argument("--directed", type=str, default="True",
                        help="if directed, choose the topk directed edges; otherwise topk undirected (no double counting)")
    parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')

    args = parser.parse_args()

    return args
