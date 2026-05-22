# GraduationDesign

此为硕士毕业设计的模型与实验

实验设计思路见：https://docs.qq.com/mind/DYk5xc2VvRkdZSE1k?&u=b9a7c15a507e4b578643638e56686de5

参考文献见：https://docs.qq.com/sheet/DYnBMd3dsRUh4UFFp?tab=BB08J2&u=b9a7c15a507e4b578643638e56686de5

总体的实验相关文档见：https://docs.qq.com/desktop/mydoc/folder/bkFzXPpcOgeH?u=0a9302808bcd4fe1905012972a94a424&_t=1671605634809

实验环境要求参见 requirement.txt

## 1、构建数据集：
数据集的构建说明见：https://docs.qq.com/doc/DYkh3VmtGZ3hvdGt0?u=0a9302808bcd4fe1905012972a94a424


## 2、训练模型：
### 2.1 模型设计
hete gnn(SimpleHGN) + hyper gnn
### 2.2 对比模型
机器学习模型：LR、SVM、GBDT

同质图模型：GAT、GCN、GATv2、GCNII

异质图模型：HAN、ie-HGCN、HAT

同时处理同质图和异质图的模型：HOG-GCN、BM-GCN、OGNN

超图模型：HGNN、HGNN+
### 2.3 实验结果
训练模型的测试结果见：https://docs.qq.com/sheet/DYmlHSnpDUkZKbm5S?tab=BB08J2&u=0a9302808bcd4fe1905012972a94a424

### 2.4 其他
训练模型的代码位于`train.py`文件，模型代码位于`model`文件夹，训练好的模型位于`model_save`文件夹

## 3、解释模型：
解释模型输出指标的含义见：https://docs.qq.com/doc/DYmFzekZNb3J1amRw
### 3.1 模型设计
求edge_mask时加入异质图的边类型的处理 + node_mask与edge_mask或者node_feat_mask进行结合
### 3.2 对比模型
扰动模型：原始的GNNExplainer、PGExplainer、SubgraphX

代理模型：PGM-Explainer、DnX
### 3.3 实验结果
解释结果位于`explain_results`文件夹下
### 3.4 其他
解释模型的代码位于`explain.py`文件，模型代码位于`explainer`文件夹，训练好的pgexplainer、dnx模型位于`explain_model_save`文件夹