#!/usr/bin/env python
# encoding: utf-8
'''
@author: Eason
@software: Pycharm
@file: rghat_conv.py
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


class RGHATConv(MessagePassing):
    def __init__(self, in_channel, out_channel,heads, num_rels,dropout= 0.1,bias=True,
                 negative_slope=0.2,add_self_loops=True, params=None,**kwargs):
        """
        """
        kwargs.setdefault('aggr', 'add')
        super(RGHATConv, self).__init__(node_dim=0,**kwargs)

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_rels = num_rels
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.heads = heads
        self.combine = params.combine
        self.bn = nn.BatchNorm1d(self.heads)
        self.ent_wk = nn.Linear(self.in_channel,self.heads*self.out_channel,bias=False)
        # k rel weight aspect weight
        self.rel_wk = nn.Linear(self.in_channel,self.heads*self.out_channel,bias=False)
        self.attn_w = nn.Parameter(th.Tensor(1,self.heads,self.out_channel))

        self.w1 = nn.Parameter(th.Tensor(self.out_channel*2,self.out_channel))
        self.w2 = nn.Parameter(th.Tensor(self.out_channel*2,self.out_channel))

        self.w3 = nn.Parameter(th.Tensor(self.out_channel,self.out_channel))
        self.w4 = nn.Parameter(th.Tensor(self.out_channel,self.out_channel))

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
        glorot(self.w2)
        glorot(self.w3)
        glorot(self.w4)

        glorot(self.p)
        glorot(self.q)
        zeros(self.bias)

    def forward(self,x,edge_index,edge_type,rel_emb=None,size=None):
        x = self.ent_wk(x).view(-1,self.heads,self.out_channel)
        rel_emb = self.rel_wk(rel_emb).view(-1,self.heads,self.out_channel)
        out = self.propagate(edge_index=edge_index,x=x,edge_type=edge_type,rel_emb=rel_emb,size=size)
        # out = out.view(-1,self.heads,self.out_channel)
        # x = x.view(-1,self.heads,self.out_channel)
        out = F.dropout(out,self.dropout,training=self.training)
        out = self.bn(out)
        if self.combine =='add':
            out = th.matmul(out+x,self.w3)
            out = self.activation(out)
            out = out.mean(dim=1)

        elif self.combine =='mult':
            out = th.matmul(out*x,self.w4)
            out = self.activation(out)
            out = out.mean(dim=1)
        else:
            out = 1/2*(self.activation(th.matmul(out*x,self.w4)) + self.activation(th.matmul(out+x,self.w3)))
            out = out.mean(dim=1)
        # out = self.bn(out)
        return out,rel_emb.mean(dim=1)

    def message(self,x_j,x_i,edge_index_i,edge_type,rel_emb) -> Tensor:
        r = th.index_select(rel_emb,0,edge_type)
        # r = r.view(-1,self.heads,self.out_channel)
        # [n_edge,heads,output_channel]
        x_i = th.concat([x_i,r],dim=-1).matmul(self.w1)
        # a_hr = th.concat([x_i,r],dim=-1).matmul(self.w1)
        # [n_edge,heads]
        alpha_hr = (x_i*self.p).sum(dim=-1)
        alpha_hr = self.activation(alpha_hr)
        # [n_edge,1]
        ent_deg = torch.zeros(edge_index_i.size(0)).to(x_i.device)
        for i in range(self.num_rels*2):
            mask = edge_type==i
            rdeg = degree(edge_index_i[mask])
            ent_deg[mask] = th.index_select(rdeg,0,edge_index_i[mask])
        # ent_deg = th.index_select(deg,0,edge_index_i).view(-1,1)
        alpha_hr = alpha_hr.div(ent_deg.view(-1,1))
        r_alpha = softmax(alpha_hr,edge_index_i,dim=0)
        r_alpha = r_alpha*ent_deg.view(-1,1)
        # cal witn_in ent attention
        x_i = th.concat([x_i,x_j],dim=-1).matmul(self.w2)
        # a_hr = th.concat([a_hr,x_j],dim=-1).matmul(self.w2)
        alpha_bht = (x_i*self.q).sum(dim=-1)
        # alpha_bht = (a_hr*self.q).sum(dim=-1)
        alpha_bht = self.activation(alpha_bht)
        # [n_edge,k]
        across_out = torch.zeros_like(r_alpha).to(x_i.device)
        for i in range(self.num_rels*2):
            mask= edge_type==i
            across_out[mask] =softmax(alpha_bht[mask],edge_index_i[mask],dim=0)
        # [n_edge,k]
        u_hrt = r_alpha*across_out
        if self.training and self.dropout>0:
            u_hrt = F.dropout(u_hrt,self.dropout,training=self.training)
        out = x_i * u_hrt.unsqueeze(-1)
        # return  out.view(-1,self.heads*self.out_channel)
        return  out

    def update(self, agg_out: Tensor) -> Tensor:
        if self.bias is not None:
            agg_out = agg_out+self.bias
            return agg_out
        return agg_out

