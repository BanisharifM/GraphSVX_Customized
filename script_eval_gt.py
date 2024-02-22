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

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import dgl 

class RGCN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names,fp_tensor,index_run):
        super().__init__()
        # print(rel_names)
        # print('out feat {}'.format(out_feats))
        self.fpTensor=fp_tensor
        self.indexRun=index_run
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

        # print('g {}'.format(graph))
        if self.indexRun < 1:
            print('h {}'.format(h))
            torch.save(h, self.fpTensor)
        self.indexRun += 1
        # input('aaaa')
        # input('aaaa')
        return h


class HeteroRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names,fpSaveTensor):
        super().__init__()

        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names,fpSaveTensor,0)
        self.regressor = nn.Linear(hidden_dim, 1)
        # self.fpSaveTensor=fpSaveTensor
        # self.indexRun=0

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
            # print('type g {} {}'.format(type(g), g))

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
    model_path = 'models/a_saveGraph.pt'.format(args.dataset)
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
