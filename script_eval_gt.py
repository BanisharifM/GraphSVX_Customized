# The whole frame from GNN Explainer to get data and model
""" explainer_main.py

     Main user interface for the explainer module.
"""
import argparse
import os
import warnings
import torch
warnings.filterwarnings("ignore", category=FutureWarning)

import configs 
from src.data import prepare_data, selected_data
from src.eval import eval_Mutagenicity, eval_syn, eval_syn6
from utils.io_utils import fix_seed

# import torch.nn as nn

# class RGCN(nn.Module):
#     def __init__(self, in_feats, hid_feats, out_feats, rel_names,x):
#         super().__init__()
#         # print(rel_names)
#         # print('out feat {}'.format(out_feats))
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

# class HeteroRegressor(nn.Module):
#     def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
#         super().__init__()

#         self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, len(rel_names), num_bases=30)
#         self.regressor = nn.Linear(hidden_dim, 1)

#     def forward(self, g, x):  # Add 'x' as an input
#         # Get node features from the graph
#         h = g.ndata['feat']

#         # Perform RGCN propagation
#         h = self.rgcn(g, h)

#         # Compute global representation by averaging node features
#         with g.local_scope():
#             g.ndata['h'] = h
#             hg = dgl.mean_nodes(g, 'h')

#             # Apply linear regression to predict the target variable
#             return self.regressor(hg)

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

# #         return h


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



import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super(RGCN, self).__init__()
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feats, hid_feats)
            for rel in rel_names}, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(hid_feats, out_feats)
            for rel in rel_names}, aggregate='sum')

    def forward(self, graph, inputs):
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h

class HeteroRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super(HeteroRegressor, self).__init__()
        self.rgcn = RGCN(in_dim, hidden_dim, n_classes, rel_names)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, g):
        if 'feat' in g.ndata:
            h = g.ndata['feat']
        elif 'x' in g.ndata:
            h = g.ndata['x']
        else:
            raise ValueError("Graph does not contain 'feat' or 'x' node features.")

        h = self.rgcn(g, h)
        with g.local_scope():
            g.ndata['h'] = h
            hg = 0
            for ntype in g.ntypes:
                hg += dgl.mean_nodes(g, 'h', ntype=ntype)
            return self.regressor(hg)




def main(): 

    # Load a configuration
    args = configs.arg_parse()
    fix_seed(args.seed)

    # GPU or CPU
    if args.gpu:
        print("CUDA")
    else:
        print("Using CPU")

    # Load dataset
    data = prepare_data(args.dataset, args.train_ratio, args.input_dim, args.seed)
    
    # Load model 
    model_path = '/models/bestModel.pt'.format(args.dataset)
    model = torch.load(model_path)

    # Evaluate GraphSVX 
    if args.dataset == 'Mutagenicity':
        data = selected_data(data, args.dataset)
        eval_Mutagenicity(data, model, args)
    elif args.dataset == 'syn6': 
        eval_syn6(data, model, args)
    else: 
        eval_syn(data, model, args)
    
if __name__ == "__main__":
    main()
