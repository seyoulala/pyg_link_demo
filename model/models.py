#!/usr/bin/env python
# encoding: utf-8
'''
@author: Eason
@software: Pycharm
@file: predict.py
@time: 2023/4/23 11:17
@desc:
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import JumpingKnowledge

from helper import *
from model.compgcn_conv import CompGCNConv
from model.compgcn_conv_basis import CompGCNConvBasis
from model.rgat_conv import RGATConv
from model.rghat_conv import RGHATConv


class BaseModel(torch.nn.Module):
    def __init__(self, params):
        super(BaseModel, self).__init__()

        self.p = params
        self.act = torch.tanh
        self.bceloss = torch.nn.BCELoss()

    def loss(self, pred, true_label):
        return self.bceloss(pred, true_label)


class RGATBase(BaseModel):
    def __init__(self, edge_index, edge_type, ent_feature, num_rel, params=None):
        super(RGATBase, self).__init__(params)

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.ent_feature = ent_feature
        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        # 需要加上节点的额外的embedding特
        # id embedding
        self.device = self.edge_index.device
        self.init_dim = self.p.init_dim

        self.id_embed = get_param((self.p.num_ent, self.init_dim))
        self.gender_embed = nn.Embedding(3, self.init_dim)
        self.age_embed = nn.Embedding(9, self.init_dim)
        self.level_embed = nn.Embedding(11, self.init_dim)

        if self.p.feature_method == 'concat':
            self.init_dim = self.init_dim * 4

        # [num_ent,init_dim*4]
        # self.init_embed = torch.concat([self.id_embed,self.gender_embed,self.age_embed,self.level_embed],dim=0)
        self.init_rel = get_param((num_rel * 2, self.init_dim))

        self.conv1 = RGATConv(self.init_dim, self.p.gcn_dim, num_rel, self.p.k_kernel)
        self.conv2 = RGATConv(self.p.gcn_dim, self.p.embed_dim, num_rel,
                              self.p.k_kernel) if self.p.gcn_layer == 2 else None

    def forward_base(self, sub, rel, drop1, drop2):
        r = self.init_rel
        # x [N_ent,k,output_channel//k]
        self.x1 = self.gender_embed(self.ent_feature[:, 0].view(-1))
        self.x2 = self.age_embed(self.ent_feature[:, 1].view(-1))
        self.x3 = self.level_embed(self.ent_feature[:, 2].view(-1))
        if self.p.feature_method == 'concat':
            self.init_embed = torch.cat([self.id_embed, self.x1, self.x2, self.x3], dim=1).to(self.device)
        elif self.p.feature_method == 'sum':
            self.init_embed = torch.cat([self.id_embed, self.x1, self.x2, self.x3], dim=0).to(self.device)
            self.init_embed = torch.sum(self.init_embed, dim=0)
        elif self.p.feature_method == 'mean':
            self.init_embed = torch.cat([self.id_embed, self.x1, self.x2, self.x3], dim=0).to(self.device)
            self.init_embed = torch.mean(self.init_embed, dim=0)

        x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_emb=r)
        x = drop1(x)
        x, r = self.conv2(x, self.edge_index, self.edge_type, rel_emb=r) if self.p.gcn_layer == 2 else (x, r)
        x = drop2(x) if self.p.gcn_layer == 2 else x
        # [batch_ent,k,output_channel//k]
        sub_emb = torch.index_select(x, 0, sub)
        # [batch_ent,output_channel]
        rel_emb = torch.index_select(r, 0, rel)
        return sub_emb, rel_emb, x


class RGAT_LINK(RGATBase):
    def __init__(self, edge_index, edge_type, ent_feature, params=None):
        super(RGAT_LINK, self).__init__(edge_index, edge_type, ent_feature, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)
        self.p = params
        self.act = F.relu
        self.w1 = nn.Parameter(torch.Tensor(self.p.embed_dim // self.p.k_kernel, self.p.d_q))
        self.w2 = nn.Parameter(torch.Tensor(self.p.embed_dim, self.p.d_q))
        self.w3 = nn.Parameter(torch.Tensor(self.p.embed_dim + self.p.embed_dim // self.p.k_kernel, self.p.d_q))
        self.lin = nn.Linear(self.p.k_kernel * self.p.d_q, self.p.embed_dim)
        self.bn = nn.BatchNorm1d(self.p.embed_dim)

        self.reset_paramters()

    def reset_paramters(self):
        glorot(self.w1)
        glorot(self.w2)
        glorot(self.w3)

    def forward(self, sub, rel, obj=None):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)
        sub_emb = sub_emb.view(-1, self.p.k_kernel, self.p.embed_dim // self.p.k_kernel)
        # [batch_ent,output_channel]
        rel_emb = rel_emb.unsqueeze(dim=1).repeat(1, self.p.k_kernel, 1)
        # [batch_ent,k,d_q]
        key = torch.matmul(sub_emb, self.w1)
        # [batch_ent,k,d_q]
        query = torch.matmul(rel_emb, self.w2)
        # [batch_ent,k]
        attn = torch.softmax((key * query).sum(dim=-1) / torch.sqrt(torch.tensor(self.p.d_q)), dim=1)
        # [batch_ent,k,d_]
        embe_concat = torch.concat([sub_emb, rel_emb], dim=-1)
        # [batch_ent,k,d_q]
        embe_concat = torch.matmul(embe_concat, self.w3) * (attn.view(-1, self.p.k_kernel, 1))
        # [batch_ent,k*d_q]
        embe_concat = embe_concat.view(-1, self.p.k_kernel * self.p.d_q)
        output = self.act(self.bn(self.lin(embe_concat)))

        if obj is None:
            # [batch_size,num_ent]
            x = torch.mm(output, all_ent.transpose(1, 0))
        else:
            dst_emb = torch.index_select(all_ent, 0, obj)
            # [batch_size]
            x = torch.sum(output * dst_emb, dim=1, keepdim=False)
        # score = torch.sigmoid(x)
        # return score
        return x


class RHGATBase(BaseModel):
    def __init__(self, edge_index, edge_type, edge_type_p, ent_feature, num_rel, params=None):
        super(RHGATBase, self).__init__(params)

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.edge_type_p = edge_type_p
        self.ent_feature = ent_feature
        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        # 需要加上节点的额外的embedding特
        # id embedding
        self.device = self.edge_index.device
        self.add_parent_rel = self.p.add_parent_rel
        #
        self.id_embed = get_param((self.p.num_ent, self.p.init_dim))

        self.gender_embed = nn.Embedding(3, self.p.user_fe_dim)
        self.age_embed = nn.Embedding(9, self.p.user_fe_dim)
        self.level_embed = nn.Embedding(11, self.p.user_fe_dim)

        if self.p.feature_method == 'concat':
            self.ent_dim = self.p.init_dim + 3 * self.p.user_fe_dim
        else:
            self.ent_dim = self.p.init_dim

        if self.p.feature_method == 'concat':
            if self.add_parent_rel:
                # self.rel_dim =self.ent_dim//2
                self.rel_dim = self.ent_dim
            else:
                self.rel_dim = self.ent_dim
        else:
            self.rel_dim = self.ent_dim

        if self.add_parent_rel:
            self.init_rel_p = get_param((2 * self.p.num_rel_p, self.rel_dim))

        self.init_rel = get_param((num_rel * 2, self.rel_dim))
        self.conv1 = RGHATConv(self.ent_dim, self.p.gcn_dim, heads=self.p.heads, num_rels=num_rel, params=params)
        self.conv2 = RGHATConv(self.p.gcn_dim, self.p.embed_dim, self.p.heads, num_rel,
                               params=params) if self.p.gcn_layer == 2 else None

    def forward_base(self, sub, rel, relp, drop1, drop2):
        if self.add_parent_rel:
            r = (self.init_rel, self.init_rel_p)
        else:
            r = self.init_rel
        # x [N_ent,k,output_channel//k]
        self.x1 = self.gender_embed(self.ent_feature[:, 0].view(-1))
        self.x2 = self.age_embed(self.ent_feature[:, 1].view(-1))
        self.x3 = self.level_embed(self.ent_feature[:, 2].view(-1))
        if self.p.feature_method == 'concat':
            self.init_embed = torch.cat([self.id_embed, self.x1, self.x2, self.x3], dim=1).to(self.device)
        elif self.p.feature_method == 'sum':
            self.init_embed = self.id_embed + self.x1 + self.x2 + self.x3
            self.init_embed = self.init_embed.to(self.device)
        elif self.p.feature_method == 'mean':
            self.init_embed = (self.id_embed + self.x1 + self.x2 + self.x3) / 4
            self.init_embed = self.init_embed.to(self.device)
        else:
            self.init_embed = self.id_embed.to(self.device)

        x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, self.edge_type_p, rel_emb=r)
        x = drop1(x)
        x, r = self.conv2(x, self.edge_index, self.edge_type, self.edge_type_p,
                          rel_emb=r) if self.p.gcn_layer == 2 else (x, r)
        x = drop2(x) if self.p.gcn_layer == 2 else x
        # [batch_ent,k,output_channel//k]
        sub_emb = torch.index_select(x, 0, sub)
        # [batch_ent,output_channel]
        if isinstance(r, tuple):
            r1 = torch.index_select(r[0], 0, rel)
            r2 = torch.index_select(r[1], 0, relp)
            # r = torch.concat([r1,r2],dim=1)
            r = r1 + r2
        # rel_emb = torch.index_select(r, 0, rel)
        return sub_emb, r, x


class RHGAT_ConvE(RHGATBase):
    def __init__(self, edge_index, edge_type, ent_feature, params=None):
        super(RHGAT_ConvE, self).__init__(edge_index, edge_type, ent_feature, params.num_rel, params)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                       stride=1, padding=0, bias=self.p.bias)

        # flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_w - self.p.ker_sz + 1
        # flat_sz_h = int(2*self.p.k_h) - self.p.ker_sz + 1
        flat_sz_h = int(2 * self.p.embed_dim // self.p.k_w) - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.embed_dim // self.p.k_w, self.p.k_w))
        return stack_inp

    def forward(self, sub, rel, obj=None):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.hidden_drop, self.hidden_drop2)
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)
        #
        # x = F.adaptive_avg_pool2d(x,(1,1))
        if obj is None:
            x = torch.mm(x, all_ent.transpose(1, 0))
        else:
            dst_emb = torch.index_select(all_ent, 0, obj)
            x = torch.sum(x * dst_emb, dim=1, keepdim=False)
        # x += self.bias.expand_as(x)
        # score = torch.sigmoid(x)
        # return score
        return x


class RHGAT_DistMult(RHGATBase):
    def __init__(self, edge_index, edge_type, edge_type_p, ent_feature, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, edge_type_p, ent_feature, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel, relp=None, obj=None):

        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, relp, self.drop, self.drop)
        # [batch_size,embed_dim]
        obj_emb = sub_emb * rel_emb

        if obj is None:
            # [batch_size,num_ent]
            x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        else:
            dst_emb = torch.index_select(all_ent, 0, obj)
            # [batch_size]
            x = torch.sum(obj_emb * dst_emb, dim=1, keepdim=False)
        # x += self.bias.expand_as(x)
        # score = torch.sigmoid(x)
        # return score
        return x


class CompGCNBase(BaseModel):
    def __init__(self, edge_index, edge_type, ent_feature, num_rel, params=None):
        super(CompGCNBase, self).__init__(params)

        self.edge_index = edge_index
        self.edge_type = edge_type
        self.ent_feature = ent_feature
        self.p.gcn_dim = self.p.embed_dim if self.p.gcn_layer == 1 else self.p.gcn_dim
        # self.init_embed = get_param((self.p.num_ent, self.p.init_dim))
        self.device = self.edge_index.device

        self.id_embed = get_param((self.p.num_ent, self.p.init_dim))
        self.gender_embed = nn.Embedding(3, self.p.init_dim)
        self.age_embed = nn.Embedding(9, self.p.init_dim)
        self.level_embed = nn.Embedding(11, self.p.init_dim)

        if self.p.feature_method == 'concat':
            self.p.init_dim = self.p.init_dim * 4
        if self.p.num_bases > 0:
            self.init_rel = get_param((self.p.num_bases, self.p.init_dim))

        else:
            if self.p.score_func == 'transe':
                self.init_rel = get_param((num_rel, self.p.init_dim))
            else:
                self.init_rel = get_param((num_rel * 2, self.p.init_dim))

        if self.p.num_bases > 0:
            self.conv1 = CompGCNConvBasis(self.p.init_dim, self.p.gcn_dim, num_rel, self.p.num_bases, act=self.act,
                                          cache=self.p.cache, params=self.p)
            self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act,
                                     params=self.p) if self.p.gcn_layer == 2 else None
        else:
            self.conv1 = CompGCNConv(self.p.init_dim, self.p.gcn_dim, num_rel, act=self.act, params=self.p)
            self.conv2 = CompGCNConv(self.p.gcn_dim, self.p.embed_dim, num_rel, act=self.act,
                                     params=self.p) if self.p.gcn_layer == 2 else None

        self.register_parameter('bias', nn.Parameter(torch.zeros(self.p.num_ent)))

    def forward_base(self, sub, rel, drop1, drop2):
        r = self.init_rel if self.p.score_func != 'transe' else torch.cat([self.init_rel, -self.init_rel], dim=0)
        self.x1 = self.gender_embed(self.ent_feature[:, 0].view(-1))
        self.x2 = self.age_embed(self.ent_feature[:, 1].view(-1))
        self.x3 = self.level_embed(self.ent_feature[:, 2].view(-1))
        if self.p.feature_method == 'concat':
            self.init_embed = torch.cat([self.id_embed, self.x1, self.x2, self.x3], dim=1).to(self.device)
        elif self.p.feature_method == 'sum':
            self.init_embed = torch.cat([self.id_embed, self.x1, self.x2, self.x3], dim=0).to(self.device)
            self.init_embed = torch.sum(self.init_embed, dim=0)
        elif self.p.feature_method == 'mean':
            self.init_embed = torch.cat([self.id_embed, self.x1, self.x2, self.x3], dim=0).to(self.device)
            self.init_embed = torch.mean(self.init_embed, dim=0)
        x, r = self.conv1(self.init_embed, self.edge_index, self.edge_type, rel_embed=r)
        x = drop1(x)
        x, r = self.conv2(x, self.edge_index, self.edge_type, rel_embed=r) if self.p.gcn_layer == 2 else (x, r)
        x = drop2(x) if self.p.gcn_layer == 2 else x
        # 需要对整个图的节点表征进行一次计算，然后选出当前batch中的节点的embed
        sub_emb = torch.index_select(x, 0, sub)
        rel_emb = torch.index_select(r, 0, rel)

        return sub_emb, rel_emb, x


class CompGCN_TransE(CompGCNBase):
    def __init__(self, edge_index, edge_type, ent_feature, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, ent_feature, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel, obj=None):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)
        obj_emb = sub_emb + rel_emb

        x = self.p.gamma - torch.norm(obj_emb.unsqueeze(1) - all_ent, p=1, dim=2)
        # score = torch.sigmoid(x)

        # return score
        return x


class CompGCN_DistMult(CompGCNBase):
    def __init__(self, edge_index, edge_type, ent_feature, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, ent_feature, params.num_rel, params)
        self.drop = torch.nn.Dropout(self.p.hid_drop)

    def forward(self, sub, rel, obj=None):
        """

        Parameters
        ----------
        sub
        rel
        obj

        Returns [batch_size,num_ent]
        -------

        """
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.drop, self.drop)
        # [batch_size,embed_dim]
        obj_emb = sub_emb * rel_emb

        if obj is None:
            # [batch_size,num_ent]
            x = torch.mm(obj_emb, all_ent.transpose(1, 0))
        else:
            dst_emb = torch.index_select(all_ent, 0, obj)
            # [batch_size]
            x = torch.sum(obj_emb * dst_emb, dim=1, keepdim=False)
        # x += self.bias.expand_as(x)
        # score = torch.sigmoid(x)
        # return score
        return x


class CompGCN_ConvE(CompGCNBase):
    def __init__(self, edge_index, edge_type, ent_feature, params=None):
        super(self.__class__, self).__init__(edge_index, edge_type, ent_feature, params.num_rel, params)

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.p.num_filt)
        self.bn2 = torch.nn.BatchNorm1d(self.p.embed_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.hid_drop)
        self.hidden_drop2 = torch.nn.Dropout(self.p.hid_drop2)
        self.feature_drop = torch.nn.Dropout(self.p.feat_drop)
        self.m_conv1 = torch.nn.Conv2d(1, out_channels=self.p.num_filt, kernel_size=(self.p.ker_sz, self.p.ker_sz),
                                       stride=1, padding=0, bias=self.p.bias)

        flat_sz_h = int(2 * self.p.k_w) - self.p.ker_sz + 1
        flat_sz_w = self.p.k_h - self.p.ker_sz + 1
        self.flat_sz = flat_sz_h * flat_sz_w * self.p.num_filt
        self.fc = torch.nn.Linear(self.flat_sz, self.p.embed_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.p.embed_dim)
        rel_embed = rel_embed.view(-1, 1, self.p.embed_dim)
        stack_inp = torch.cat([e1_embed, rel_embed], 1)
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape((-1, 1, 2 * self.p.k_w, self.p.k_h))
        return stack_inp

    def forward(self, sub, rel, obj=None):
        sub_emb, rel_emb, all_ent = self.forward_base(sub, rel, self.hidden_drop, self.feature_drop)
        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = F.relu(x)

        if obj is None:
            x = torch.mm(x, all_ent.transpose(1, 0))
        else:
            dst_emb = torch.index_select(all_ent, 0, obj)
            x = torch.sum(x * dst_emb, dim=1, keepdim=False)
        # x += self.bias.expand_as(x)
        # score = torch.sigmoid(x)
        # return score
        return x
