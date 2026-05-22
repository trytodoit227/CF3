import torch.nn as nn

from model.baselines.GAT import GAT
from model.baselines.GATv2 import GATv2
from model.baselines.GCN import GCN
from model.baselines.GCNII import GCNII
from model.baselines.HAN import HAN
from model.baselines.HAT import HAT
from model.baselines.HETGNN import HETGNN
from model.baselines.HGCN import HGCN
from model.baselines.HGNN import HGNN
from model.baselines.HGNNP import HGNNP
from model.baselines.HWNN import HWNN
from model.baselines.RGCN import RGCN
from model.baselines.SimpleHGN import SimpleHGN
from model.baselines.HOGGCN import HOGGCN
from model.baselines.BMGCN import BMGCN
from model.baselines.OGNN import OGNN


class Baseline(nn.Module):
    def __init__(self, input_dim, output_dim, company_num, person_num, rel_num, device, conv_name='gat'):
        super(Baseline, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'gcn':
            self.base_conv = GCN(input_dim, output_dim, company_num, person_num)
        elif self.conv_name == 'gcn2':
            self.base_conv = GCNII(input_dim, output_dim, company_num, person_num)
        elif self.conv_name == 'gat':
            self.base_conv = GAT(input_dim, output_dim, company_num, person_num)
        elif self.conv_name == 'gat2':
            self.base_conv = GATv2(input_dim, output_dim, company_num, person_num)
        elif self.conv_name == 'rgcn':
            self.base_conv = RGCN(input_dim, output_dim, company_num, person_num, rel_num)
        elif self.conv_name == 'han':
            self.base_conv = HAN(input_dim, output_dim, company_num, person_num, rel_num)
        elif self.conv_name == 'hetgnn':
            self.base_conv = HETGNN(input_dim, output_dim, company_num, person_num, rel_num)
        elif self.conv_name == 'hgnn':
            self.base_conv = HGNN(input_dim, output_dim, company_num)
        elif self.conv_name == 'hgnnp':
            self.base_conv = HGNNP(input_dim, output_dim, company_num)
        elif self.conv_name == 'hwnn':
            self.base_conv = HWNN(input_dim, output_dim, company_num)
        elif self.conv_name == 'hgcn':
            self.base_conv = HGCN(input_dim, output_dim, company_num, person_num)
        elif self.conv_name == 'hat':
            self.base_conv = HAT(input_dim, output_dim, company_num, person_num, rel_num, device)
        elif self.conv_name == 'simplehgn':
            self.base_conv = SimpleHGN(input_dim, output_dim, company_num, person_num)
        elif self.conv_name == 'ognn':
            self.base_conv = OGNN(input_dim, output_dim, company_num, person_num)
        elif self.conv_name == 'hoggcn':
            self.base_conv = HOGGCN(input_dim, output_dim, company_num, person_num)
        elif self.conv_name == 'bmgcn':
            self.base_conv = BMGCN(input_dim, output_dim, company_num, person_num)

    def forward(self, hete_graph, hyp_graph, idx, x, **kwargs):
        if self.conv_name == 'gcn':
            return self.base_conv(hete_graph, idx, x)
        elif self.conv_name == 'gcn2':
            return self.base_conv(hete_graph, idx, x)
        elif self.conv_name == 'gat':
            return self.base_conv(hete_graph, idx, x)
        elif self.conv_name == 'gat2':
            return self.base_conv(hete_graph, idx, x)
        elif self.conv_name == 'rgcn':
            return self.base_conv(hete_graph, idx, x)
        elif self.conv_name == 'han':
            return self.base_conv(hete_graph, idx, x)
        elif self.conv_name == 'hetgnn':
            return self.base_conv(hete_graph, idx, x)
        elif self.conv_name == 'hgnn':
            return self.base_conv(hyp_graph, idx, x)
        elif self.conv_name == 'hgnnp':
            return self.base_conv(hyp_graph, idx, x)
        elif self.conv_name == 'hwnn':
            return self.base_conv(hyp_graph, idx, x)
        elif self.conv_name == 'hgcn':
            return self.base_conv(hete_graph, idx, x)
        elif self.conv_name == 'hat':
            return self.base_conv(hete_graph, idx, x)
        elif self.conv_name == 'simplehgn':
            return self.base_conv(hete_graph, idx, x)
        elif self.conv_name == 'ognn':
            return self.base_conv(hete_graph, idx, x)
        elif self.conv_name == 'hoggcn':
            return self.base_conv(hete_graph, idx, x, **kwargs)
        elif self.conv_name == 'bmgcn':
            return self.base_conv(hete_graph, idx, x, **kwargs)
