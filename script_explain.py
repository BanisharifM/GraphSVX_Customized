""" script_explain.py

    Derive explanations using GraphSVX 
"""

import argparse
import random
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore")

import configs
from utils.io_utils import fix_seed
from src.data import prepare_data
from src.explainers import GraphSVX
from src.train import evaluate, test

# import torch.nn as nn

# class RGCN(nn.Module):
#     def __init__(self, in_feats, hid_feats, out_feats, rel_names):
#         super().__init__()

#         self.conv1 = dglnn.HeteroGraphConv({
#             rel: dglnn.GraphConv(in_feats, hid_feats)
#             for rel in rel_names}, aggregate='sum')
#         self.conv2 = dglnn.HeteroGraphConv({
#             rel: dglnn.GraphConv(hid_feats, hid_feats)
#             for rel in rel_names}, aggregate='sum')
#         self.conv3 = dglnn.HeteroGraphConv({
#             rel: dglnn.GraphConv(hid_feats, hid_feats)
#             for rel in rel_names}, aggregate='sum')
#         self.conv4 = dglnn.HeteroGraphConv({
#             rel: dglnn.GraphConv(hid_feats, hid_feats)
#             for rel in rel_names}, aggregate='sum')
#         self.conv5 = dglnn.HeteroGraphConv({
#             rel: dglnn.GraphConv(hid_feats, hid_feats)
#             for rel in rel_names}, aggregate='sum')
#         self.conv6 = dglnn.HeteroGraphConv({
#             rel: dglnn.GraphConv(hid_feats, out_feats)
#             for rel in rel_names}, aggregate='sum')

#     def forward(self, g, h):
#         h = self.conv1(g, h)
#         h = self.conv2(g, h)
#         h = self.conv3(g, h)
#         h = self.conv4(g, h)
#         h = self.conv5(g, h)
#         h = self.conv6(g, h)

#         return h


# class HeteroRegressor(nn.Module):
#     def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
#         super().__init__()

#         self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
#         self.regressor = nn.Linear(hidden_dim, 1)

#     def forward(self, g):
#         h = g.ndata['feat']
#         h = self.rgcn(g, h)

#         with g.local_scope():
#             g.ndata['x'] = h  # Add 'x' attribute to the nodes
#             hg = 0

#             for ntype in g.ntypes:
#                 hg = hg + dgl.mean_nodes(g, 'x', ntype=ntype)

#             return self.regressor(hg)


import torch.nn as nn
import dgl.nn.pytorch as dglnn

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()

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

    def forward(self, g, x):
        h = x
        h = self.conv1(g, h)
        h = self.conv2(g, h)
        h = self.conv3(g, h)
        h = self.conv4(g, h)
        h = self.conv5(g, h)
        h = self.conv6(g, h)

        return h

    
class HeteroRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, g, x):
        h = x  # Use 'x' as input
        h = self.rgcn(g, h)

        with g.local_scope():
            g.ndata['x'] = h  # Add 'x' attribute to the nodes
            hg = 0

            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, 'x', ntype=ntype)

            return self.regressor(hg)




def main():

    args = configs.arg_parse()
    fix_seed(args.seed)

    # Load the dataset
    data = prepare_data(args.dataset, args.train_ratio,
                        args.input_dim, args.seed)

    # Load the model
    # model_path = 'models/{}_model_{}.pth'.format(args.model, args.dataset)
    model_path = 'models/bestModel.pt'
    model = torch.load(model_path)
    
    # Evaluate the model 
    if args.dataset in ['Cora', 'PubMed']:
        _, test_acc = evaluate(data, model, data.test_mask)
    else: 
        test_acc = test(data, model, data.test_mask)
    print('Test accuracy is {:.4f}'.format(test_acc))

    # Explain it with GraphSVX
    explainer = GraphSVX(data, model, args.gpu)

    # Distinguish graph classfication from node classification
    if args.dataset in ['Mutagenicity', 'syn6']:
        explanations = explainer.explain_graphs(args.indexes,
                                         args.hops,
                                         args.num_samples,
                                         args.info,
                                         args.multiclass,
                                         args.fullempty,
                                         args.S,
                                         'graph_classification',
                                         args.feat,
                                         args.coal,
                                         args.g,
                                         args.regu,
                                         True)
    else: 
        explanations = explainer.explain(args.indexes,
                                        args.hops,
                                        args.num_samples,
                                        args.info,
                                        args.multiclass,
                                        args.fullempty,
                                        args.S,
                                        args.hv,
                                        args.feat,
                                        args.coal,
                                        args.g,
                                        args.regu,
                                        True)

    print('Sum explanations: ', [np.sum(explanation) for explanation in explanations])
    print('Base value: ', explainer.base_values)

if __name__ == "__main__":
    main()
