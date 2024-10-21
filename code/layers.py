import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import os
init = nn.init.xavier_uniform_
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()
        self.act = nn.LeakyReLU(negative_slope=0.5)     #args.leaky)

    def forward(self, adj, embeds):
        return self.act(torch.spmm(adj, embeds))


class DM_gcn(nn.Module):
    def __init__(self, Diagnum, mednum, pronum, featuredim):
        super(DM_gcn, self).__init__()
        self.dEmbed = nn.Parameter(init(torch.empty(Diagnum, featuredim)))
        self.mEmbed = nn.Parameter(init(torch.empty(mednum, featuredim)))
        self.pEmbed = nn.Parameter(init(torch.empty(pronum, featuredim)))
        self.gcnLayer1 = GCNLayer()     # LightGCN
        self.gcnLayer2 = GCNLayer()     #

        #self.HANlayers = HANLayer(num_meta_paths, featuredim, nhid, num_heads[0], dropout)
        self.Diagnum = Diagnum
        self.pronum = pronum
        self.inter = nn.Parameter(torch.FloatTensor(1))

    def forward(self, adj1, adj2):
        embeds1 = torch.cat([self.dEmbed, self.mEmbed], dim=0)
        embeds2 = torch.cat([self.pEmbed, self.mEmbed], dim=0)
        gnnLats1, gnnLats2 = [[] for _ in range(2)]    # EmbedsList
        lats1, lats2 = [embeds1], [embeds2]
        for i in range(4):     # 异构信息
            tem1 = self.gcnLayer1(adj1, lats1[-1])
            tem1 = F.relu(tem1)
            gnnLats1.append(tem1)
            tem2 = self.gcnLayer2(adj2, lats2[-1])
            tem2 = F.relu(tem2)
            gnnLats2.append(tem2)
        gnnEmbeds1 = sum(gnnLats1)      # torch.Size([15161, 64])
        gnnEmbeds2 = sum(gnnLats2)      # torch.Size([16974, 64])
        #pEmbed_gcn1, pEmbed_gcn2 = gnnEmbeds1[:args.patient], gnnEmbeds2[:args.patient]     # torch.Size([15016, 64])
        mEmbed_gcn_1 = gnnEmbeds1[self.Diagnum:]     # torch.Size([145, 64])
        dEmbed_gcn = gnnEmbeds1[:self.Diagnum]
        mEmbed_gcn_2 = gnnEmbeds2[self.pronum:]  # torch.Size([145, 64])
        pEmbed_gcn = gnnEmbeds2[:self.pronum]
        mEmbed = self.inter * mEmbed_gcn_1 + (1 - self.inter) * mEmbed_gcn_2
        return mEmbed, dEmbed_gcn, pEmbed_gcn

class DDI_gcn(nn.Module):
    def __init__(self, mednum, featuredim):
        super(DDI_gcn, self).__init__()
        self.mEmbed = nn.Parameter(init(torch.empty(mednum, featuredim)))
        #self.m2Embed = nn.Parameter(init(torch.empty(mednum, featuredim)))
        self.gcnLayer1 = GCNLayer()     # LightGCN
        self.mednum = mednum
        self.inter = nn.Parameter(torch.FloatTensor(1))

    def forward(self, adj1):
        embeds1 = torch.cat([self.mEmbed, self.mEmbed], dim=0)
        gnnLats1 = []    # EmbedsList
        lats1 = [embeds1]
        for i in range(2):     # 异构信息
            tem1 = self.gcnLayer1(adj1, lats1[-1])
            tem1 = F.relu(tem1)
            gnnLats1.append(tem1)
        gnnEmbeds1 = sum(gnnLats1)      # torch.Size([15161, 64])
        m1Embed_gcn = gnnEmbeds1[:self.mednum]
        m2Embed_gcn = gnnEmbeds1[self.mednum:]
        mEmbed = self.inter * m1Embed_gcn + (1 - self.inter) * m2Embed_gcn
        return mEmbed

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'