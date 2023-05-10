#!/usr/bin/env python
# encoding: utf-8
'''
@author: Eason
@software: Pycharm
@file: tem.py
@time: 2023/5/9 08:18
@desc:
'''
import torch
import torch.nn.functional as F
import  torch.nn as nn
from torch_geometric.datasets import Planetoid
import argparse
from  model.models import  RHGATBase
import torchstat


parser = argparse.ArgumentParser(description="Parser For Arguments",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter, )
parser.add_argument("--data_dir", dest="data_dir", default="./data", help="Dataset dir", )
parser.add_argument("--output_dir", dest="output_dir", default="./output", help="Output dir", )
parser.add_argument("--ckpt_dir", dest="ckpt_dir", default="./checkpoint", help="Checkpoint dir", )
parser.add_argument("--opn", dest="opn", default="corr", help="Composition Operation to be used in CompGCN", )
parser.add_argument("--batch", dest="batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--gpu", type=int, default=0, help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0", )
parser.add_argument("--epoch", dest="max_epochs", type=int, default=1000, help="Number of epochs", )
parser.add_argument("--l2", type=float, default=0.0, help="L2 Regularization for Optimizer")
parser.add_argument("--lr", type=float, default=0.01, help="Starting Learning Rate")
parser.add_argument("--num_workers", type=int, default=2, help="Number of processes to construct batches", )
parser.add_argument("--seed", dest="seed", default=41504, type=int, help="Seed for randomization", )
parser.add_argument("--num_bases", dest="num_bases", default=-1, type=int,
                    help="Number of basis relation vectors to use", )
parser.add_argument("--init_dim", dest="init_dim", default=50, type=int,
                    help="Initial dimension size for entities and relations", )
parser.add_argument('--embed_dim', dest="embed_dim", default=200, type=int,
                    help="Embedding dimension to give as input to score function")
parser.add_argument('--gcn_layer', dest='gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
parser.add_argument("--gcn_dim", dest="gcn_dim", default=200, type=int, help="Number of hidden uints of GCN layer")
parser.add_argument("--gcn_drop", dest="dropout", default=0.1, type=float, help="Dropout to use in GCN Layer", )
parser.add_argument('--hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')
parser.add_argument("--layer_dropout", nargs="?", default="[0.3]",
                    help="List of dropout value after each compGCN layer", )
parser.add_argument('--model_name', dest='model_name', default='rhgat', help='Gnn model as  encoder ')
parser.add_argument('--score_func', dest='score_func', default='conve', help='Score Function for Link prediction')
parser.add_argument('--bias', dest='bias', action='store_true', help='Whether to use bias in the model')
parser.add_argument('--cache', dest='cache', action='store_true', help='Whether to use cache  in the gcn model')
parser.add_argument('--num_neg', dest='num_neg', default=1, type=int, help='Number of Negative sample')

# ConvE specific hyperparameters
parser.add_argument('--hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
parser.add_argument('--feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
parser.add_argument('--k_w', dest='k_w', default=10, type=int, help='ConvE: k_w')
parser.add_argument('--k_h', dest='k_h', default=20, type=int, help='ConvE: k_h')
parser.add_argument('--num_filt', dest='num_filt', default=200, type=int,
                    help='ConvE: Number of filters in convolution')
parser.add_argument('--ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

# rGAT specific hyperparameters
parser.add_argument('--k_kernel', dest='k_kernel', default=2, type=int, help='k Kernel to use for ent')
parser.add_argument('--d_q', dest='d_q', default=200, type=int,
                    help='Embedding dimension to give as input to score function')
### RHGAT specific hyperparameters
parser.add_argument('--heads', dest='heads', default=8, type=int, help='multi heads for attention')
parser.add_argument('--combine', dest='combine', default='add', type=str, help='combination method ')

args = parser.parse_args()
args.num_ent = 299888
args.num_rel = 112

src = torch.LongTensor([1,2,3,4,5,6,3,3,4])
dst = torch.LongTensor([0,0,0,0,1,1,1,2,2])
edge_index  = torch.stack([src,dst],dim=0)
edge_type = torch.LongTensor([0,0,0,1,1,2,2,3,3])
ent_feature = torch.randint(0,2,(7,3))


class M(RHGATBase):
    def __init__(self, edge_index, edge_type, ent_feature, num_rel, params=None):
        super(M, self).__init__(edge_index, edge_type, ent_feature, num_rel, params)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
    def forward(self,sub,rel,obj=None):
        sub_emb, rel_emb, x = self.forward_base(sub,rel,self.hidden_drop,self.hidden_drop2)
        return  sub_emb



model = M(edge_index,edge_type,ent_feature,4,args)
for name,parameter in model.named_parameters():
    print(name,parameter.size())

# out = model(src,edge_type)







