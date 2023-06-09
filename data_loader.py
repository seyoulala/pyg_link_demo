#!/usr/bin/env python
# encoding: utf-8
'''
@author: Eason
@software: Pycharm
@file: data_loader.py
@time: 2023/4/18 17:17
@desc:
'''
import gc
import random
from collections import defaultdict as ddict
from typing import *

import pandas as pd
import torch
from ordered_set import OrderedSet
from torch.utils.data import DataLoader, Dataset


class TrainDataset(Dataset):
    def __init__(self, triples: List, sr2o: dict, num_ent: int, num_neg=1):
        self.triples = triples
        self.sr2o = sr2o
        self.num_ent = num_ent
        self.num_neg = num_neg

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        ele = self.triples[idx]
        src, rel, relp, obj = ele
        triples = [torch.LongTensor([src, rel, relp, obj])]
        labels = [torch.FloatTensor([1.0])]
        while True:
            neg_obj = random.randint(0, self.num_ent - 1)
            if neg_obj not in self.sr2o[(src, rel,relp)]:
                triples.append(torch.LongTensor([src, rel, relp, neg_obj]))
                labels.append(torch.FloatTensor([0.0]))
            if len(triples) > self.num_neg:
                break

        return torch.stack(triples, 0), torch.stack(labels, 0)

    @staticmethod
    def collate_fn(data):
        triples = []
        labels = []
        for triple, label in data:
            triples.append(triple)
            labels.append(label)
        triple = torch.cat(triples, dim=0)
        trp_label = torch.cat(labels, dim=0)
        return triple, trp_label


class TestDataset(Dataset):
    def __init__(self, sr2o, triple2idx=None):
        self.sr2o = sr2o
        self.triples, self.ids = [], []
        for (s, r, r1), o_list in self.sr2o.items():
            self.triples.append([s, r, r1, -1])
            if triple2idx is None:
                self.ids.append(0)
            else:
                self.ids.append([triple2idx[(s, r)]])

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        return torch.LongTensor(self.triples[idx]), torch.LongTensor(self.ids[idx])

    @staticmethod
    def collate_fn(data):
        triples = []
        ids = []
        for triple, idx in data:
            triples.append(triple)
            ids.append(idx)
        triples = torch.stack(triples, dim=0)
        trp_ids = torch.stack(ids, dim=0)
        return triples, trp_ids


class Data(object):
    def __init__(self, data_dir, num_workers, batch_size, num_neg):
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_neg = num_neg

        ent_set, rel_set = OrderedSet(), OrderedSet()
        # 读入事件类型表，event_id表示所有的事件类型
        event_info = pd.read_json("{}/event_info.json".format(self.data_dir))
        rel_set.update(event_info['event_id'].tolist())
        print('Number of events: {}'.format(len(rel_set)))
        # 读入用户信息表
        user_info = pd.read_json("{}/user_info.json".format(self.data_dir))
        ent_set.update(user_info['user_id'].tolist())
        print('Number of users: {}'.format(len(ent_set)))

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel + "_reverse": idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.num_ent = len(self.ent2id)
        self.num_rel = len(self.rel2id) // 2

        # 添加父场景的信息
        self.partner_map = {}
        for rows in event_info.itertuples():
            self.partner_map[rows.event_id] = rows.parent_event_id
            self.partner_map[rows.event_id + '_reverse'] = rows.parent_event_id + '_reverse'

        parent_set = OrderedSet()
        parent_set.update(event_info['parent_event_id'].tolist())
        print('Number of parent events: {}'.format(len(parent_set)))

        self.parrel2id = {rel: idx for idx, rel in enumerate(parent_set)}
        self.parrel2id.update({rel + "_reverse": idx + len(self.parrel2id) for idx, rel in enumerate(parent_set)})
        self.id2parrel = {idx: rel for rel, idx in self.parrel2id.items()}

        self.num_rel_p = len(self.parrel2id) // 2

        # 添加用户相关的特征,user_age,user_level,user_sex
        age_set = set(user_info['age_level'].tolist())
        gender_set = set(user_info['gender_id'].tolist())
        level_set = set(user_info['user_level'].tolist())

        self.age2id = {age: idx for idx, age in enumerate(age_set)}
        self.gender2id = {gender: idx for idx, gender in enumerate(gender_set)}
        self.level2id = {level: idx for idx, level in enumerate(level_set)}

        self.ent_fe = {}
        for rows in user_info.itertuples():
            self.ent_fe[self.ent2id[rows.user_id]] = torch.LongTensor(
                [self.gender2id[rows.gender_id], self.age2id[rows.age_level],
                 self.level2id[rows.user_level]])
        self.ent_feid = []
        for entid in range(self.num_ent):
            self.ent_feid.append(self.ent_fe[entid])

        del self.ent_fe, self.age2id, self.gender2id, self.level2id
        gc.collect()

        self.data = ddict(list)
        self.sr2o = dict()
        for split in ['train', 'valid', 'test']:
            self.sr2o[split] = ddict(set)

        src = []
        dst = []
        rels = []
        inver_src = []
        inver_dst = []
        inver_rels = []
        df = pd.read_json("{}/source_event_preliminary_train_info.json".format(self.data_dir))
        records = df.to_dict('records')
        for line in records:
            sub, rel, obj = line['inviter_id'], line['event_id'], line['voter_id']
            relp = self.partner_map[rel]
            # 三元组转换
            sub_id, rel_id, relp_id, obj_id = (
                self.ent2id[sub],
                self.rel2id[rel],
                self.parrel2id[relp],
                self.ent2id[obj],
            )
            self.data['train'].append((sub_id, rel_id, relp_id, obj_id))
            self.sr2o['train'][(sub_id, rel_id, relp_id)].add(obj_id)
            self.sr2o['train'][(obj_id, rel_id + self.num_rel, relp_id + self.num_rel_p)].add(sub_id)  # 添加反向边
            src.append(sub_id)
            dst.append(obj_id)
            rels.append(rel_id)
            inver_src.append(obj_id)
            inver_dst.append(sub_id)
            inver_rels.append(rel_id + self.num_rel)

        ratio = 0.3
        few_shot_valid_cnt = ddict(int)
        df = pd.read_json("{}/target_event_preliminary_train_info.json".format(self.data_dir))
        rel_cnt_dict = df['event_id'].value_counts().to_dict()
        records = df.to_dict('records')
        random.shuffle(records)
        for line in records:
            sub, rel, obj = line['inviter_id'], line['event_id'], line['voter_id']
            relp = self.partner_map[rel]
            sub_id, rel_id, relp_id, obj_id = (
                self.ent2id[sub],
                self.rel2id[rel],
                self.parrel2id[relp],
                self.ent2id[obj],
            )
            # 目标场景的相关的三元组，每种关系都保留200条样本作为验证集，其余加入到训练集中进行训练
            if few_shot_valid_cnt[rel]/rel_cnt_dict[rel] < ratio:
                self.sr2o['valid'][(sub_id, rel_id, relp_id)].add(obj_id)
                self.data['valid'].append([sub_id, rel_id, relp_id, obj_id])
                few_shot_valid_cnt[rel] += 1
            else:
                self.data['train'].append((sub_id, rel_id, relp_id, obj_id))
                self.sr2o['train'][(sub_id, rel_id, relp_id)].add(obj_id)
                self.sr2o['train'][(obj_id, rel_id + self.num_rel, relp_id + self.num_rel_p)].add(sub_id)
                src.append(sub_id)
                dst.append(obj_id)
                rels.append(rel_id)
                inver_src.append(obj_id)
                inver_dst.append(sub_id)
                inver_rels.append(rel_id + self.num_rel)

        self.triple2idx = dict()
        df = pd.read_json("{}/target_event_preliminary_test_info.json".format(self.data_dir))
        records = df.to_dict('records')
        # 测试样本的relation和target场景的relation是一样的
        for line in records:
            triple_id = int(line['triple_id'])
            sub, rel = line['inviter_id'], line['event_id']
            relp = self.partner_map[rel]
            sub_id, rel_id, relp_id = self.ent2id[sub], self.rel2id[rel], self.parrel2id[relp]
            self.sr2o['test'][(sub_id, rel_id, relp_id)] = set()
            # index
            self.triple2idx[(sub_id, rel_id)] = triple_id

        print('****************************')
        for split in ['train', 'valid']:
            print('Number of {} triples: {}'.format(split, len(self.data[split])))

        print('****************************')
        self.data = dict(self.data)
        for split in ['valid', 'test']:
            print('Number of {} queries: {}'.format(split, len(self.sr2o[split])))

        # construct pyg graph
        src = src + inver_src
        dst = dst + inver_dst
        rels = rels + inver_rels

        self.edge_index = torch.stack([torch.LongTensor(src), torch.LongTensor(dst)], dim=0)
        self.edge_type = torch.LongTensor(rels)
        self.ent_feid = torch.stack(self.ent_feid, dim=0)
        # 父关系
        self.edge_type_parent = torch.LongTensor([self.parrel2id[self.partner_map[self.id2rel[idx]]] for idx in rels])
        print("Number of user fe: {}".format(self.ent_feid.shape))

        # identify in and out edges
        def get_train_data_loader(split, batch_size, shuffle=True):
            return DataLoader(
                TrainDataset(
                    self.data[split], self.sr2o[split], self.num_ent, self.num_neg
                ),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.num_workers),
                collate_fn=TrainDataset.collate_fn,
                pin_memory=True
            )

        def get_test_data_loader(split, batch_size, shuffle=False):
            triple2idx = None if split == 'valid' else self.triple2idx
            return DataLoader(
                TestDataset(
                    self.sr2o[split], triple2idx
                ),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.num_workers),
                collate_fn=TestDataset.collate_fn,
                pin_memory=True
            )

        # train/valid/test dataloaders
        self.data_iter = {
            "train": get_train_data_loader("train", self.batch_size),
            "valid": get_test_data_loader("valid", 800),
            "test": get_test_data_loader("test", 1024),
        }
