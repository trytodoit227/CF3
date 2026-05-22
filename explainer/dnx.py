import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm
import numpy as np
import os

from utils.model_utils import gen_attribute_hg, get_sub_info


# 蒸馏模型
class SGC(nn.Module):
    def __init__(self, company_num, person_num, k, input_dim, output_dim, device):
        super(SGC, self).__init__()

        self.company_num = company_num
        self.person_num = person_num

        self.att_dim = 32
        self.proj = nn.Linear(input_dim + self.att_dim, input_dim, bias=False)

        self.company_emb = nn.Embedding(company_num, input_dim)
        self.person_emb = nn.Embedding(person_num, input_dim)
        self.conv = SGConv(input_dim, output_dim, k)

        self.device = device

    def forward(self, hete_graph, idx, x):
        edge_index, edge_type, edge_weight = hete_graph
        edge_index = torch.LongTensor(edge_index).transpose(0, 1)
        company_emb = self.company_emb(torch.LongTensor([i for i in range(self.company_num)]))
        person_emb = self.person_emb(torch.LongTensor([i for i in range(self.person_num)]))
        company_attr_cal = torch.zeros((self.company_num, self.att_dim))
        company_attr_cal[idx] = torch.FloatTensor(x)
        company_emb = self.proj(torch.cat((company_emb, company_attr_cal), dim=1))
        emb = torch.cat((company_emb, person_emb), dim=0)

        emb = self.conv(emb, edge_index)

        return emb[idx].to(self.device)


# 解释模型
class DnX(nn.Module):
    def __init__(self, model, n_input):
        super(DnX, self).__init__()
        self.E = nn.Parameter(torch.randn(n_input, 1, dtype=torch.float32))
        self.model = model

    def forward(self, x, model, hete_graph, idx, t=5):
        node_weight = torch.softmax(self.E / t, 0)
        y_expl = model(hete_graph, idx, node_weight * x)
        return node_weight, y_expl


def sgc_train(data, model, args, company_num=3976, person_num=2405):
    avg = []
    x = data['features']
    hyp_graph = []
    for i in ['industry', 'area', 'qualify']:
        hyp_graph += [gen_attribute_hg(3976, data['hyp_graph'][i], X=None)]
    company_emb = model(hete_graph=data['hete_graph'], hyp_graph=hyp_graph, idx=data['idx'],
                        x=x.float())  # 获得全图的预测值
    pred = data['classifier'].forward(company_emb)

    sgc_model = SGC(company_num, person_num, k=2, input_dim=x.shape[1], output_dim=12, device=args.device)
    optimizer = torch.optim.Adam(sgc_model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    train_acc_list = []
    best_acc = 0

    mask = np.ones((x.shape[0], x.shape[0]))

    for epoch in tqdm(range(1, 201)):
        # train
        sgc_model.train()
        optimizer.zero_grad()
        company_emb = sgc_model(data['hete_graph'], data['idx'], x.numpy())
        res = data['classifier'].forward(company_emb)
        loss = criterion(res, pred)
        loss.backward(retain_graph=True)
        optimizer.step()
        del loss
        # test
        sgc_model.eval()
        company_emb = sgc_model(data['hete_graph'], data['idx'], x.numpy())
        logits = data['classifier'].forward(company_emb)
        for mask in [mask]:
            mask_pred = torch.argmax(logits[mask], dim=1)
            train_acc = mask_pred.eq(torch.argmax(pred[mask], dim=1)).sum().item() / mask.sum().item()
        if train_acc > best_acc:
            best_acc = train_acc
            best_model = sgc_model
        train_acc_list.append(train_acc)

    avg.append(train_acc)
    print('Mean Accuracy:{} |  Std Accuracy:{}'.format(np.mean(avg), np.std(avg)))

    # save
    save_explain_model_path = './explain_model_save'
    if not os.path.exists(save_explain_model_path):
        os.makedirs(save_explain_model_path)
    dnx_saving_path = os.path.join(save_explain_model_path, f"dnx_sgc_model_model_{args.conv_name}.pth")
    torch.save(best_model, dnx_saving_path)


def dnx_explain(node_idx, data, model, sgc_model, test_company_num, epochs=200):
    test_idx_dic = data['test_idx_dic']
    company_emb = sgc_model(data['hete_graph'], data['idx'], data['features'].numpy())
    pred = data['classifier'].forward(company_emb)
    nodes_explanations = {}
    t = 5
    neighbors, sub_edge_index, _, edge_mask = k_hop_subgraph(int(node_idx), 3, data['edge_index'],
                                                             relabel_nodes=False)
    neighbors_index = torch.tensor(
        [test_idx_dic[v.item()] for i, v in enumerate(neighbors) if v.item() in test_idx_dic.keys()])
    neighbors_index = torch.tensor([i.item() for i in neighbors_index if i.item() < len(data['idx'])])
    sub_x = data['features'][neighbors_index]

    explainer = DnX(model=model, n_input=len(neighbors_index))
    opt = torch.optim.Adam(explainer.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, min_lr=1e-5, patience=20,
                                                           verbose=True)
    loss = nn.MSELoss()

    pbar = tqdm(total=epochs)
    pbar.set_description(f'Explain node {node_idx}')

    for epoch in range(1, epochs + 1):
        explainer.train()
        opt.zero_grad()
        sub_idx, sub_hete_graph, _ = get_sub_info(node_idx, sub_edge_index, edge_mask, data['hete_graph'],
                                                  data['hyp_graph'], test_idx_dic, test_company_num)
        _, company_emb_ex = explainer(sub_x, sgc_model, sub_hete_graph, sub_idx, t)
        pred_ex = data['classifier'].forward(company_emb_ex)
        l = loss(pred_ex[sub_idx.index(node_idx)], pred[test_idx_dic[node_idx]])
        l.backward(retain_graph=True)
        opt.step()
        scheduler.step(l)

        # generating_explanations
        explainer.eval()
        # expl是解释的节点的权重
        expl, _ = explainer(sub_x, sgc_model, sub_hete_graph, sub_idx, t)
        values, nodes = torch.topk(expl.squeeze(-1), dim=-1, k=expl.shape[0])
        explic = neighbors[nodes]
        stats = {v.item(): values[i].detach().numpy() for i, v in enumerate(explic)}
        if len(nodes_explanations.keys()) == 0:
            nodes_explanations.update(stats)
        else:
            result_dict = {}
            for key in nodes_explanations:
                if key in stats:
                    # 两个字典中都存在相同的键，将其值相加
                    result_dict[key] = stats[key] + nodes_explanations[key]
            nodes_explanations = result_dict

        pbar.update(1)
    pbar.close()

    # 去掉不在test_idx范围内的节点
    node_add_index = [i.item() for i in explic if test_idx_dic[i.item()] >= len(data['idx'])]
    for delete_key in node_add_index:
        if delete_key in node_add_index:
            del nodes_explanations[delete_key]

    for key in nodes_explanations:
        nodes_explanations[key] = nodes_explanations[key] / epochs

    values = list(nodes_explanations.values())
    nodes_explanations_stats = torch.zeros(data['features'].shape[0], dtype=torch.float)

    neighbors_index = [test_idx_dic[i] for i in nodes_explanations.keys()]
    nodes_explanations_stats[neighbors_index] = torch.tensor(values, dtype=torch.float)

    node_mask = torch.zeros(data['features'].shape[0], dtype=torch.int)
    node_mask[neighbors_index] = 1

    return node_mask, nodes_explanations_stats
