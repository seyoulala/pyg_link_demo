#!/usr/bin/env python
# encoding: utf-8
'''
@author: Eason
@software: Pycharm
@file: rgat_conv.py
@time: 2023/4/27 21:16
@desc:
'''
import torch
import torch as th
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import  softmax,remove_self_loops,add_self_loops
from torch_geometric.typing import Tensor,Adj

class RGATConv(MessagePassing):
    def __init__(self, in_channel, out_channel, num_rels, k,concat=True,dropout= 0.1,bias=True,
                 negative_slope=0.2,add_self_loops=True, params=None,**kwargs):
        """
        """
        kwargs.setdefault('aggr', 'add')
        super(RGATConv, self).__init__(**kwargs)

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_rels = num_rels
        self.k = k
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.p = params
        self.device = params.device

        # k entity weight aspect weight
        self.ent_wk = nn.Linear(self.in_channel,self.out_channel,bias=False)
        # k rel weight aspect weight
        self.rel_wk = nn.Linear(self.in_channel,self.out_channel,bias=False)
        self.attn_w = nn.Parameter(th.Tensor(1,self.k,(self.out_channel//self.k)*3))
        self.loop_rel = nn.Parameter(th.Tensor(1, self.in_channel))  # self loop 这条边的embedding
        if bias:
            self.bias = nn.Parameter(th.Tensor((self.out_channel//k)*self.k))
        else:
            self.register_parameter('bias',None)

        self.drop = nn.Dropout(self.dropout)
        self.bn = nn.BatchNorm1d(self.out_channel)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.ent_wk)
        glorot(self.rel_wk)
        glorot(self.attn_w)
        glorot(self.loop_rel)
        zeros(self.bias)

    def forward(self,x,edge_index,edge_type,rel_emb=None,size=None):
        # [num_ent,self.out_channel]
        num_ent = x.size(0)
        x = self.ent_wk(x)
        # r = th.index_select(rel_emb,0,edge_type)
        # [N_edge,self.out_channel]
        rel_emb = torch.cat([rel_emb,self.loop_rel],dim=0)
        r = self.rel_wk(rel_emb)
        self.loop_type = torch.full((num_ent,), rel_emb.size(0) - 1, dtype=torch.long).to(self.device)
        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x.size(0)
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, edge_attr = remove_self_loops(edge_index)
                edge_index, edge_attr = add_self_loops(edge_index, edge_attr,num_nodes=num_nodes)
            edge_type = torch.cat([edge_type,self.loop_type],dim=0)

        output = self.propagate(edge_index,x=x,edge_type=edge_type,rel_emb=r)
        output = self.drop(output)
        output = self.bn(output)

        return F.relu(output),r[:-1] # ignore self loop embed

    def message(self,edge_index_i,x_i,x_j,edge_type,rel_emb):
        # [N_edge,k,out_channel//k]
        x_i = x_i.view(-1,self.k,self.out_channel//self.k)
        x_j = x_j.view(-1,self.k,self.out_channel//self.k)
        r = th.index_select(rel_emb,0,edge_type)
        r = r.view(-1,self.k,self.out_channel//self.k)
        #  [N_edge,k]
        alpha = (th.concat([x_j,r,x_i],dim=-1)*self.attn_w).sum(dim=-1)
        alpha = F.leaky_relu(alpha,self.negative_slope)
        # [N_edge,k]
        alpha = softmax(alpha,index=edge_index_i)
        # [N_edge,k,out_channel//k]
        alpha = F.dropout(alpha,p=self.dropout,training=True)
        out = (x_j*r)*alpha.view(-1,self.k,1)
        return out.view(-1,self.k*self.out_channel//self.k)

    def update(self, aggr_out: Tensor) -> Tensor:
        # aggr_out= aggr_out.view(-1,self.k*self.out_channel//self.k)
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.k)












