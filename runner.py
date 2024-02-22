#Original
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        # print(rel_names)
        # print('out feat {}'.format(out_feats))
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv3 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv4 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv5 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv6 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs is features of nodes
        # print('graph {}\nimputs {}'.format(graph, inputs))
        h = self.conv1(graph, inputs)
        # print(' 1 h shape {}'.format(h.keys()))
        # h = {k: F.relu(v) for k, v in h.items()}
        # h = self.conv2(graph, h)
        # h = {k: F.relu(v) for k, v in h.items()}
        # h = self.conv3(graph, h)
        # h = {k: F.relu(v) for k, v in h.items()}
        # h = self.conv4(graph, h)
        # h = {k: F.relu(v) for k, v in h.items()}
        # h = self.conv5(graph, h)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv6(graph, h)
        return h
    
class HeteroRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        h = g.ndata['feat']
        # print('h feat {}'.format(h))
        h = self.rgcn(g, h)
        # print('damn h for g {} \n ge herre {}'.format(g, h))
        # print('h shape {}'.format(h))
        # input('bbbb')
        # if len(list(h2.keys()))>0:
        #     h=h2
        with g.local_scope():
            g.ndata['h'] = h
            hg = 0

            for ntype in g.ntypes:
                # print('ntyoe {}'.format(ntype))
                hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
                # pass
            # print('type hg {} {} {}'.format(type(hg),hg.shape,hg))
            # input('aaaa')
            return self.regressor(hg)


data=torch.load("data/syn1.pth")
print("\n-------------------------- Syn 1 Data -------------------\n")
print(data)

data=torch.load("data/bestModel.pt")
print("\n-------------------------- Best Model Data -------------------\n")
print(data)

data=torch.load("data/bestModel_2.pt")
print("\n-------------------------- Best Model 2 Data -------------------\n")
print(data)

data=torch.load("data/a_saveGraph.pt")
print("\n-------------------------- a_saveGraph.pt Data -------------------\n")
print(data)
