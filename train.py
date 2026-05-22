# coding: utf-8
import sys

import torch.nn.functional as F

from model.model_old import Model
from model.baseline import Baseline
from model.baselines.HOGGCN import HOGGCN_MLP
from model.baselines.BMGCN import BMGCN_MLP
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
# from xgboost import XGBClassifier

from utils.model_utils import *

from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import recall_score as rec
from sklearn.metrics import precision_score as pre
from sklearn.metrics import f1_score as f1
from sklearn.metrics import roc_auc_score as roc

from utils.parser_utils import arg_parse

import warnings

warnings.filterwarnings('ignore')


def MLP_train(optimizer_mlp):
    ## MLP pre-train
    best_acc = 0
    for i in range(args.mlp_epoch):
        model_MLP.train()
        optimizer_mlp.zero_grad()
        res_train = model_MLP(torch.FloatTensor(x_train))
        output_train = res_train.argmax(axis=1)
        res_valid = model_MLP(torch.FloatTensor(x_valid))
        output_valid = res_valid.argmax(axis=1)
        res_test = model_MLP(torch.FloatTensor(x_test))
        output_test = res_test.argmax(axis=1)
        loss = criterion(res_train, torch.LongTensor(train_label))
        acc_train = acc(output_train.detach().numpy(), train_label)
        loss.backward()
        optimizer_mlp.step()

        model_MLP.eval()
        acc_val = acc(output_valid.detach().numpy(), valid_label)
        acc_test = acc(output_test.detach().numpy(), test_label)

        if acc_val >= best_acc:
            best_acc = acc_val
            torch.save(model_MLP, './model_save/%s.pkl' % (args.conv_name + '_MLP'))
            print('MLP Model UPDATE!!!')

        # sys.stdout.flush()
        # sys.stdout.write('\r')
        # sys.stdout.write(
        #     "Epoch #{:4d}\tTrain Loss: {:.4f} | Train Acc: {:.4f} | Val Acc: {:.4f} | Test Acc: {:.4f}".format(i + 1, loss.item(), acc_train, acc_val, acc_test))

        print('epoch:{}'.format(i + 1),
              'loss: {:.4f}'.format(loss.item()),
              'acc: {:.4f}'.format(acc_train),
              'val: {:.4f}'.format(acc_val),
              'test: {:.4f}'.format(acc_test))

    labels_for_lp_train = one_hot_embedding(max(train_label) + 1, res_train).type(torch.FloatTensor)
    labels_for_lp_valid = one_hot_embedding(max(valid_label) + 1, res_valid).type(torch.FloatTensor)
    labels_for_lp_test = one_hot_embedding(max(test_label) + 1, res_test).type(torch.FloatTensor)
    return labels_for_lp_train, labels_for_lp_valid, labels_for_lp_test


def train():
    up = 0

    best_acc = 0
    best_f1 = 0

    for epoch in np.arange(args.n_epoch):
        st = time.time()
        '''
            Train 
        '''
        model.train()
        train_losses = []

        company_emb = gnn.forward(train_hete_graph, train_hyp, train_idx, x_train)
        res = classifier.forward(company_emb)
        loss = criterion(res, torch.LongTensor(train_label).to(device))

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        train_losses += [loss.cpu().detach().tolist()]
        scheduler.step()
        del res, loss

        '''
            Valid 
        '''
        model.eval()
        with torch.no_grad():
            company_emb = gnn.forward(valid_hete_graph, valid_hyp, valid_idx, x_valid)
            res = classifier.forward(company_emb).cpu().detach()
            loss = criterion(res, torch.LongTensor(valid_label))

            pred = res.argmax(dim=1)
            ac = acc(valid_label, pred)
            pr = pre(valid_label, pred)
            re = rec(valid_label, pred)
            f = f1(valid_label, pred)
            rc = roc(valid_label, res[:, 1])

            # if ac > best_acc and f > best_f1:
            if ac > best_acc and f > best_f1:
                best_acc = ac
                best_f1 = f
                # torch.save(model.state_dict(), './model_save/%s.pkl' % args.conv_name)
                torch.save(model, './model_save/%s.pkl' % args.conv_name)
                print('UPDATE!!!')
                up = 1

            et = time.time()
            print((
                      "Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid Acc: %.4f Valid Pre: %.4f  Valid Recall: %.4f Valid F1: %.4f  Valid Roc: %.4f") % (
                      epoch, (et - st), optimizer.param_groups[0]['lr'], np.average(train_losses),
                      loss.cpu().detach().tolist(), ac, pr, re, f, rc))

            if up == 1:
                test()
                up = 0

            del res, loss

            if epoch + 1 == args.n_epoch:
                company_emb = gnn.forward(test_hete_graph, test_hyp, test_idx, x_test)

                res = classifier.forward(company_emb).cpu().detach()

                pred = res.argmax(dim=1)
                ac = acc(test_label, pred)
                pr = pre(test_label, pred)
                re = rec(test_label, pred)
                f = f1(test_label, pred)
                rc = roc(test_label, res[:, 1])

                print(
                    'Last Test Acc: %.4f Last Test Pre: %.4f Last Test Recall: %.4f Last Test F1: %.4f Last Test ROC: %.4f' % (
                        ac, pr, re, f, rc))
                print(sum(pred))


def test():
    best_model = torch.load('./model_save/%s.pkl' % args.conv_name)
    best_model.eval()
    gnn, classifier = best_model
    with torch.no_grad():
        # company_index,label=test_label
        company_emb = gnn.forward(test_hete_graph, test_hyp, test_idx, x_test)
        res = classifier.forward(company_emb).cpu().detach()

        pred = res.argmax(dim=1)
        ac = acc(test_label, pred)
        pr = pre(test_label, pred)
        re = rec(test_label, pred)
        f = f1(test_label, pred)
        rc = roc(test_label, res[:, 1])

        pred_probs = np.abs(np.choose(test_label, res.numpy().T))
        prob = pred_probs.mean().item()

        print(
            'Best Test Prob: %.4f Best Test Acc: %.4f Best Test Pre: %.4f Best Test Recall: %.4f Best Test F1: %.4f Best Test ROC: %.4f' % (
                prob, ac, pr, re, f, rc))
        print(sum(pred))


def HOGGCN_train():
    up = 0

    best_acc = 0
    best_f1 = 0
    best_model_MLP = torch.load('./model_save/%s.pkl' % (args.conv_name + '_MLP'))

    for epoch in np.arange(args.n_epoch):
        st = time.time()
        '''
            Train 
        '''
        model.train()
        best_model_MLP.train()
        train_losses = []

        # mlp loss
        res_train = best_model_MLP(torch.FloatTensor(x_train))
        output_train = res_train.argmax(axis=1)
        loss_mlp = criterion(res_train, torch.LongTensor(train_label))
        # lp loss、gcn loss
        y_hat, company_emb = gnn.forward(train_hete_graph, train_hyp, train_idx, x_train,
                                         **{"output": output_train, "labels_for_lp": labels_for_lp_train})
        res = classifier.forward(company_emb)
        loss_gcn = criterion(res, torch.LongTensor(train_label))
        loss_lp = criterion(y_hat, torch.LongTensor(train_label))
        loss = loss_mlp + loss_gcn + 1 * loss_lp  # 与 loss = loss_gcn 的结果一样

        optimizer.zero_grad()
        optimizer_mlp.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        optimizer_mlp.step()

        train_losses += [loss.cpu().detach().tolist()]
        scheduler.step()
        del res, loss, loss_mlp, loss_gcn, loss_lp

        '''
            Valid 
        '''
        model.eval()
        best_model_MLP.eval()
        with torch.no_grad():
            res_valid = best_model_MLP(torch.FloatTensor(x_valid))
            output_valid = res_valid.argmax(axis=1)
            loss_mlp = criterion(res_valid, torch.LongTensor(valid_label))
            y_hat, company_emb = gnn.forward(valid_hete_graph, valid_hyp, valid_idx, x_valid,
                                             **{"output": output_valid, "labels_for_lp": labels_for_lp_valid})
            res = classifier.forward(company_emb)
            loss_gcn = criterion(res, torch.LongTensor(valid_label))
            loss_lp = criterion(y_hat, torch.LongTensor(valid_label))
            loss = loss_mlp + loss_gcn + 1 * loss_lp  # 与 loss = loss_gcn 的结果一样

            pred = res.argmax(dim=1)
            ac = acc(valid_label, pred)
            pr = pre(valid_label, pred)
            re = rec(valid_label, pred)
            f = f1(valid_label, pred)
            rc = roc(valid_label, res[:, 1])

            # if ac > best_acc and f > best_f1:
            if ac > best_acc and f > best_f1:
                best_acc = ac
                best_f1 = f
                torch.save(model, './model_save/%s.pkl' % args.conv_name)
                print('UPDATE!!!')
                up = 1

            et = time.time()
            print((
                      "Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid Acc: %.4f Valid Pre: %.4f  Valid Recall: %.4f Valid F1: %.4f  Valid Roc: %.4f") % (
                      epoch, (et - st), optimizer.param_groups[0]['lr'], np.average(train_losses),
                      loss.cpu().detach().tolist(), ac, pr, re, f, rc))

            if up == 1:
                HOGGCN_test()
                up = 0

            del res, loss, loss_mlp, loss_gcn, loss_lp

            if epoch + 1 == args.n_epoch:
                res_test = best_model_MLP(torch.FloatTensor(x_test))
                output_test = res_test.argmax(axis=1)
                _, company_emb = gnn.forward(test_hete_graph, test_hyp, test_idx, x_test,
                                             **{"output": output_test, "labels_for_lp": labels_for_lp_test})
                res = classifier.forward(company_emb)
                pred = res.argmax(dim=1)
                ac = acc(test_label, pred)
                pr = pre(test_label, pred)
                re = rec(test_label, pred)
                f = f1(test_label, pred)
                rc = roc(test_label, res[:, 1])

                print(
                    'Last Test Acc: %.4f Last Test Pre: %.4f Last Test Recall: %.4f Last Test F1: %.4f Last Test ROC: %.4f' % (
                        ac, pr, re, f, rc))
                print(sum(pred))


def HOGGCN_test():
    best_model = torch.load('./model_save/%s.pkl' % args.conv_name)
    best_model_MLP = torch.load('./model_save/%s.pkl' % (args.conv_name + '_MLP'))
    best_model.eval()
    best_model_MLP.eval()
    gnn, classifier = best_model

    with torch.no_grad():
        res_test = best_model_MLP(torch.FloatTensor(x_test))
        output_test = res_test.argmax(axis=1)
        _, company_emb = gnn.forward(test_hete_graph, test_hyp, test_idx, x_test,
                                     **{"output": output_test, "labels_for_lp": labels_for_lp_test})
        res = classifier.forward(company_emb)

        pred = res.argmax(dim=1)
        ac = acc(test_label, pred)
        pr = pre(test_label, pred)
        re = rec(test_label, pred)
        f = f1(test_label, pred)
        rc = roc(test_label, res[:, 1])

        pred_probs = np.abs(np.choose(test_label, res.numpy().T))
        prob = pred_probs.mean().item()

        print(
            'Best Test Prob: %.4f Best Test Acc: %.4f Best Test Pre: %.4f Best Test Recall: %.4f Best Test F1: %.4f Best Test ROC: %.4f' % (
                prob, ac, pr, re, f, rc))
        print(sum(pred))


def BMGCN_train():
    up = 0

    best_acc = 0
    best_f1 = 0
    best_model_MLP = torch.load('./model_save/%s.pkl' % (args.conv_name + '_MLP'))

    for epoch in np.arange(args.n_epoch):
        st = time.time()
        '''
            Train 
        '''
        model.train()
        best_model_MLP.train()
        train_losses = []

        # mlp loss
        res_train = best_model_MLP(torch.FloatTensor(x_train))
        loss_mlp = criterion(res_train, torch.LongTensor(train_label))
        # lp loss、gcn loss
        company_emb = gnn.forward(train_hete_graph, train_hyp, train_idx, x_train,
                                  **{"output": res_train, "labels_oneHot": labels_oneHot_train})
        res = classifier.forward(company_emb)
        loss_gcn = criterion(res, torch.LongTensor(train_label))
        loss = loss_mlp + loss_gcn

        optimizer.zero_grad()
        optimizer_mlp.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        optimizer_mlp.step()

        train_losses += [loss.cpu().detach().tolist()]
        scheduler.step()
        del res, loss, loss_mlp, loss_gcn

        '''
            Valid 
        '''
        model.eval()
        best_model_MLP.eval()
        with torch.no_grad():
            res_valid = best_model_MLP(torch.FloatTensor(x_valid))
            loss_mlp = criterion(res_valid, torch.LongTensor(valid_label))
            company_emb = gnn.forward(valid_hete_graph, valid_hyp, valid_idx, x_valid,
                                      **{"output": res_valid, "labels_oneHot": labels_oneHot_valid})
            res = classifier.forward(company_emb)
            loss_gcn = criterion(res, torch.LongTensor(valid_label))
            loss = loss_mlp + loss_gcn  # 与 loss = loss_gcn 的结果一样

            pred = res.argmax(dim=1)
            ac = acc(valid_label, pred)
            pr = pre(valid_label, pred)
            re = rec(valid_label, pred)
            f = f1(valid_label, pred)
            rc = roc(valid_label, res[:, 1])

            # if ac > best_acc and f > best_f1:
            if ac > best_acc and f > best_f1:
                best_acc = ac
                best_f1 = f
                torch.save(model, './model_save/%s.pkl' % args.conv_name)
                print('UPDATE!!!')
                up = 1

            et = time.time()
            print((
                      "Epoch: %d (%.1fs)  LR: %.5f Train Loss: %.2f  Valid Loss: %.2f  Valid Acc: %.4f Valid Pre: %.4f  Valid Recall: %.4f Valid F1: %.4f  Valid Roc: %.4f") % (
                      epoch, (et - st), optimizer.param_groups[0]['lr'], np.average(train_losses),
                      loss.cpu().detach().tolist(), ac, pr, re, f, rc))

            if up == 1:
                BMGCN_test()
                up = 0

            del res, loss, loss_mlp, loss_gcn

            if epoch + 1 == args.n_epoch:
                res_test = best_model_MLP(torch.FloatTensor(x_test))
                company_emb = gnn.forward(test_hete_graph, test_hyp, test_idx, x_test,
                                          **{"output": res_test, "labels_oneHot": labels_oneHot_test})
                res = classifier.forward(company_emb)
                pred = res.argmax(dim=1)
                ac = acc(test_label, pred)
                pr = pre(test_label, pred)
                re = rec(test_label, pred)
                f = f1(test_label, pred)
                rc = roc(test_label, res[:, 1])

                print(
                    'Last Test Acc: %.4f Last Test Pre: %.4f Last Test Recall: %.4f Last Test F1: %.4f Last Test ROC: %.4f' % (
                        ac, pr, re, f, rc))
                print(sum(pred))


def BMGCN_test():
    best_model = torch.load('./model_save/%s.pkl' % args.conv_name)
    best_model_MLP = torch.load('./model_save/%s.pkl' % (args.conv_name + '_MLP'))
    best_model.eval()
    best_model_MLP.eval()
    gnn, classifier = best_model

    with torch.no_grad():
        res_test = best_model_MLP(torch.FloatTensor(x_test))
        company_emb = gnn.forward(test_hete_graph, test_hyp, test_idx, x_test,
                                  **{"output": res_test, "labels_oneHot": labels_oneHot_test})
        res = classifier.forward(company_emb)

        pred = res.argmax(dim=1)
        ac = acc(test_label, pred)
        pr = pre(test_label, pred)
        re = rec(test_label, pred)
        f = f1(test_label, pred)
        rc = roc(test_label, res[:, 1])

        pred_probs = np.abs(np.choose(test_label, res.numpy().T))
        prob = pred_probs.mean().item()

        print(
            'Best Test Prob: %.4f Best Test Acc: %.4f Best Test Pre: %.4f Best Test Recall: %.4f Best Test F1: %.4f Best Test ROC: %.4f' % (
                prob, ac, pr, re, f, rc))
        print(sum(pred))


if __name__ == '__main__':
    # 设置参数
    args = arg_parse()
    args.input_dim = 16
    args.output_dim = 12
    args.dropout = 0.2
    args.n_epoch = 200
    args.clip = 1e-5
    args.conv_name = 'model'

    # hetgnn的参数设置，与其他模型的不同之处
    # args.input_dim = 64
    # args.clip = 1e-6
    # args.conv_name = 'hetgnn'

    # hgnn的参数设置，与其他模型的不同之处
    # args.input_dim = 64
    # args.n_epoch = 100
    # args.conv_name = 'hgnn'

    # hwnn的参数设置，与其他模型的不同之处
    # args.n_epoch = 500
    # args.conv_name = 'hwnn'

    # device = torch.device("cpu")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    set_random_seed(14)

    criterion = torch.nn.CrossEntropyLoss()

    ###justification dict
    total_company_num = 3976
    person_num = 2405
    court_type = 4
    category = 4

    train_company_num = 2816
    valid_company_num = 686
    test_company_num = 474
    args.num = total_company_num + person_num

    cause_type_num = 11
    rel_num = 20

    data_path = './data/'

    train_data = pd.read_pickle('%strain_data.pkl' % data_path)
    valid_data = pd.read_pickle('%svalidate_data.pkl' % data_path)
    test_data = pd.read_pickle('%stest_data.pkl' % data_path)
    split_data_idx = pd.read_pickle('%ssplit_data_idx.pkl' % data_path)

    train_sxjl_data, train_xzxk_data, train_zdgz_data, train_risk_data, train_company_attr, train_hete_graph, train_hyper_graph, train_label = train_data
    valid_sxjl_data, valid_xzxk_data, valid_zdgz_data, valid_risk_data, valid_company_attr, valid_hete_graph, valid_hyper_graph, valid_label = valid_data
    test_sxjl_data, test_xzxk_data, test_zdgz_data, test_risk_data, test_company_attr, test_hete_graph, test_hyper_graph, test_label = test_data
    train_idx, valid_idx, test_idx = split_data_idx

    x_train = initialize_company_info(train_sxjl_data, train_zdgz_data, train_xzxk_data, train_risk_data,
                                      train_company_attr, train_company_num, cause_type_num, court_type, category,
                                      train_idx)
    x_valid = initialize_company_info(valid_sxjl_data, valid_zdgz_data, valid_xzxk_data, valid_risk_data,
                                      valid_company_attr, valid_company_num, cause_type_num, court_type, category,
                                      valid_idx)
    x_test = initialize_company_info(test_sxjl_data, test_zdgz_data, test_xzxk_data, test_risk_data,
                                     test_company_attr, test_company_num, cause_type_num, court_type, category,
                                     test_idx)

    # ML model
    if args.conv_name in ['lr', 'knn', 'svm', 'dt', 'gbdt', 'xgb']:
        if args.conv_name == 'lr':
            model = LogisticRegression(C=1.5, random_state=42)
            model.fit(x_train, train_label)
            prob = model.predict_proba(x_test)
            res = [np.argmax(i) for i in prob]
            print('Test Acc: %.4f Test precision: %.4f Test recall: %.4f Test f1: %.4f Test roc: %.4f' %
                  (acc(test_label, res), pre(test_label, res), rec(test_label, res), f1(test_label, res),
                   roc(test_label, np.array(prob)[:, 1])))

        if args.conv_name == 'knn':
            model = KNeighborsClassifier(p=1)
            model.fit(x_train, train_label)
            res = model.predict(x_test)
            print('Test Acc: %.4f Test precision: %.4f Test recall: %.4f Test f1: %.4f Test roc: %.4f' %
                  (acc(test_label, res), pre(test_label, res), rec(test_label, res), f1(test_label, res),
                   roc(test_label, res)))

        if args.conv_name == 'svm':
            model = svm.SVC(kernel='linear', random_state=42)
            model.fit(x_train, train_label)
            res = model.predict(x_test)
            print('Test Acc: %.4f Test precision: %.4f Test recall: %.4f Test f1: %.4f Test roc: %.4f' %
                  (acc(test_label, res), pre(test_label, res), rec(test_label, res), f1(test_label, res),
                   roc(test_label, res)))

        if args.conv_name == 'dt':
            model = DecisionTreeClassifier(max_features=30, max_depth=3, random_state=42)
            model.fit(x_train, train_label)
            res = model.predict(x_test)
            print('Test Acc: %.4f Test precision: %.4f Test recall: %.4f Test f1: %.4f Test roc: %.4f' %
                  (acc(test_label, res), pre(test_label, res), rec(test_label, res), f1(test_label, res),
                   roc(test_label, res)))

        if args.conv_name == 'gbdt':
            model = GradientBoostingClassifier(learning_rate=0.05, min_samples_leaf=30, max_features='sqrt',
                                               random_state=42)
            model.fit(x_train, train_label)
            res = model.predict(x_test)
            pred_prob = model.predict_proba(x_test)[:, 1]
            print('Test Acc: %.4f Test precision: %.4f Test recall: %.4f Test f1: %.4f Test roc: %.4f' %
                  (acc(test_label, res), pre(test_label, res), rec(test_label, res), f1(test_label, res),
                   roc(test_label, pred_prob)))

        # if args.conv_name == 'xgb':
        #     model = XGBClassifier(learning_rate=0.05, random_state=42)
        #     model.fit(x_train, train_label)
        #     res = model.predict(x_test)
        #     pred_prob = model.predict_proba(x_test)[:, 1]
        #     print('Test Acc: %.4f Test precision: %.4f Test recall: %.4f Test f1: %.4f Test roc: %.4f' %
        #           (acc(test_label, res), pre(test_label, res), rec(test_label, res), f1(test_label, res),
        #            roc(test_label, pred_prob)))
        exit()

    # other gnn model
    if args.conv_name in ['gat', 'gat2', 'gcn', 'gcn2', 'rgcn', 'hat', 'han', 'hetgnn', 'hgnn', 'hgnnp', 'hgcn', 'hwnn',
                          'simplehgn', 'ognn', 'hoggcn', 'bmgcn']:
        if args.conv_name in ['hoggcn']:
            args.mlp_epoch = 10
            model_MLP = HOGGCN_MLP(n_feat=x_train.shape[1], n_hid=16, nclass=2)
            optimizer_mlp = torch.optim.Adam(model_MLP.parameters(), lr=0.001, weight_decay=0.02)
            labels_for_lp_train, labels_for_lp_valid, labels_for_lp_test = MLP_train(optimizer_mlp)
        if args.conv_name in ['bmgcn']:
            args.mlp_epoch = 400
            labels_oneHot_train = F.one_hot(torch.LongTensor(train_label), num_classes=max(train_label) + 1).float()
            labels_oneHot_valid = F.one_hot(torch.LongTensor(valid_label), num_classes=max(valid_label) + 1).float()
            labels_oneHot_test = F.one_hot(torch.LongTensor(test_label), num_classes=max(test_label) + 1).float()

            model_MLP = BMGCN_MLP(in_size=x_train.shape[1], hidden_size=16, out_size=labels_oneHot_train.shape[1],
                                  num_layers=2)
            optimizer_mlp = torch.optim.Adam(model_MLP.parameters(), lr=0.01, weight_decay=5e-4)
            _, _, _ = MLP_train(optimizer_mlp)
        gnn = Baseline(args.input_dim, args.output_dim, total_company_num, person_num, rel_num,
                       conv_name=args.conv_name, device=device)

    # my gnn model
    if args.conv_name == 'model':
        gnn = Model(args.input_dim, args.output_dim, total_company_num, person_num, rel_num, device)
    classifier = Classifier(args.output_dim, 2).to(device)
    model = nn.Sequential(gnn, classifier)

    if args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters())

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=1e-6)

    train_hyp, valid_hyp, test_hyp = [], [], []
    for i in ['industry', 'area', 'qualify']:
        train_hyp += [gen_attribute_hg(total_company_num, train_hyper_graph[i], X=None)]
        valid_hyp += [gen_attribute_hg(total_company_num, valid_hyper_graph[i], X=None)]
        test_hyp += [gen_attribute_hg(total_company_num, test_hyper_graph[i], X=None)]

    '''
        Train、Valid
    '''
    if args.conv_name == 'hoggcn':
        HOGGCN_train()
    elif args.conv_name == 'bmgcn':
        BMGCN_train()
    else:
        train()

    '''
        Evaluate 
    '''
    if args.conv_name == 'hoggcn':
        HOGGCN_test()
    elif args.conv_name == 'bmgcn':
        BMGCN_test()
    else:
        test()
