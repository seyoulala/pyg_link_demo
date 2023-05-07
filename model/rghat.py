#!/usr/bin/env python
# encoding: utf-8
'''
@author: Eason
@software: Pycharm
@file: rghat.py
@time: 2023/5/7 20:08
@desc:
'''
import torch
import torch as th
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import  softmax,remove_self_loops,add_self_loops,degree
from torch_geometric.typing import Tensor,Adj
from torch_geometric.nn import GATConv

class RGHATConv(MessagePassing):
    def __init__(self, in_channel, out_channel,heads, num_rels,concat=False,dropout= 0.1,bias=True,
                 negative_slope=0.2,add_self_loops=True, params=None,**kwargs):
        """
        """
        kwargs.setdefault('aggr', 'add')
        super(RGHATConv, self).__init__(**kwargs)

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_rels = num_rels
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.heads = heads
        self.p = params

        self.ent_wk = nn.Linear(self.in_channel,self.heads*self.out_channel,bias=False)
        # k rel weight aspect weight
        self.rel_wk = nn.Linear(self.in_channel,self.heads*self.out_channel,bias=False)
        self.attn_w = nn.Parameter(th.Tensor(1,self.heads,self.out_channel))

        self.w1 = nn.Parameter(th.Tensor(self.out_channel*2,self.out_channel))
        self.w2 = nn.Parameter(th.Tensor(self.out_channel*2,self.out_channel))

        self.p = nn.Parameter(th.Tensor(self.heads,self.out_channel))
        self.q = nn.Parameter(th.Tensor(self.heads,self.out_channel))


        self.activation = nn.LeakyReLU(negative_slope=self.negative_slope)

        if not bias:
            self.register_parameter('bias',None)
        else:
            self.bias = nn.Parameter(th.Tensor(self.out_channel))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.ent_wk)
        glorot(self.rel_wk)
        glorot(self.attn_w)
        glorot(self.w1)
        glorot(self.p)
        glorot(self.q)
        zeros(self.bias)

    def forward(self,x,edge_index,edge_type,rel_emb=None,size=None):
        x = self.ent_wk(x)
        rel_emb = self.rel_wk(rel_emb)

        out = self.propagate(edge_index=edge_index,x=x,edge_type=edge_type,rel_emb=rel_emb,size=size)



    def message(self,x_i,x_j,edge_index_i,edge_type,rel_emb) -> Tensor:
        x_i = x_i.view(-1,self.heads,self.out_channel)
        x_j = x_j.view(-1,self.heads,self.out_channel)
        r = th.index_select(rel_emb,0,edge_type)
        r = r.view(-1,self.heads,self.out_channel)
        # [n_edge,heads,output_channel]
        a_hr = th.concat([x_i,r],dim=-1).matmul(self.w1)
        # [n_edge,heads]
        alpha_hr = (a_hr*self.p).sum(dim=-1)
        alpha_hr = self.activation(alpha_hr)
        # [n_ent]
        deg = degree(edge_index_i)
        # [n_edge,1]
        ent_deg =th.index_select(deg,0,edge_index_i).view(-1,1)
        alpha_hr = alpha_hr.div(ent_deg)
        r_alpha = softmax(alpha_hr,edge_index_i,dim=0)
        r_alpha = r_alpha*ent_deg
        # cal witn_in ent attention
        b_hrt = th.concat([a_hr,x_j],dim=-1).matmul(self.w2)
        alpha_bht = (b_hrt*self.q).sum(dim=-1)
        alpha_bht = self.activation(alpha_bht)
        # [n_edge,k]
        across_out = torch.zeros_like(r_alpha)
        for i in range(self.num_rels):
            mask= edge_type==i
            across_out[mask] =softmax(alpha_bht[mask],edge_index_i[mask],dim=0)
        u_hrt = r_alpha*across_out













