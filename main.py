#!/usr/bin/env python
# encoding: utf-8
'''
@author: Eason
@software: Pycharm
@file: predict.py
@time: 2023/4/23 11:17
@desc:
'''
import os
import argparse
from time import time

import torch.backends.cudnn
from tqdm import tqdm


import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data_loader import Data
from torch.cuda.amp import  autocast,GradScaler
from model.models import CompGCN_DistMult,CompGCN_ConvE,RGAT_LINK,RHGAT_ConvE,RHGAT_DistMult

import heapq
from collections import defaultdict as ddict
from utils import save_to_json

# link prediction task
def get_metrics(pred, gt_list, train_list, top_k):
    metrics = ['HITS', 'MRR']
    result = {metric: 0.0 for metric in metrics}
    pred = np.array(pred)
    # 将训练集中的obj的概率重置为-np.inf
    pred[train_list] = -np.inf
    pred_dict = {idx: score for idx, score in enumerate(pred)}
    pred_dict = heapq.nlargest(top_k, pred_dict.items(), key=lambda kv :kv[1])
    pred_list = [k for k,v in pred_dict]

    for gt in gt_list:
        if gt in pred_list:
            index = pred_list.index(gt)
            result['MRR'] = max(result['MRR'], 1/(index+1))
            result['HITS'] = 1
    return result


def get_candidate_voter_list(model, device, data, submit_path, top_k):
    print('Start inference...')
    model.eval()
    all_triples, all_preds, all_ids = [], [], []
    submission = []
    with th.no_grad():
        test_iter = iter(data.data_iter['test'])
        for step, (triples, trp_ids) in tqdm(enumerate(test_iter)):
            triples = triples.to(device)
            sub, rel, obj = (
                triples[:, 0],
                triples[:, 1],
                triples[:, 2],
            )
            preds = model(sub, rel) # (batch_size, num_ent)

            triples = triples.cpu().tolist()
            preds = preds.cpu().tolist()
            ids = trp_ids.cpu().tolist()

            for (triple, pred, triple_id) in zip(triples, preds, ids):
                s, r, _ = triple
                train_set = data.sr2o['train'][(s, r)]
                train_set.update(data.sr2o['valid'][(s, r)])
                train_list = np.array(list(train_set), dtype=np.int64)

                pred = np.array(pred)
                pred[train_list] = -np.inf
                pred_dict = {idx: score for idx, score in enumerate(pred)}
                candidate_voter_dict = heapq.nlargest(top_k, pred_dict.items(), key=lambda kv :kv[1])
                candidate_voter_list = [data.id2ent[k] for k,v in candidate_voter_dict]
                
                submission.append({
                    'triple_id': '{:04d}'.format(triple_id[0]),
                    'candidate_voter_list': candidate_voter_list
                })
    save_to_json(submit_path, submission)
        

def evaluate(model, device, data, top_k=5):
    model.eval()
    results = ddict(list)
    all_triples, all_preds = [], []
    with th.no_grad():
        test_iter = iter(data.data_iter['valid'])
        for step, (triples, trp_ids) in enumerate(test_iter):
            triples = triples.to(device)
            sub, rel, obj = (
                triples[:, 0],
                triples[:, 1],
                triples[:, 2],
            )
            preds = model(sub, rel) # (batch_size, num_ent)
            all_triples += triples.cpu().tolist()
            all_preds += preds.cpu().tolist()

    for triple, pred in zip(all_triples, all_preds):
        s, r, _ = triple
        # valid集的 object节点集合
        gt_set = data.sr2o['valid'][(s, r)]
        # train集中的object节点集合
        train_set = data.sr2o['train'][(s, r)]
        train_list = np.array(list(train_set), dtype=np.int64)
        gt_list = np.array(list(gt_set), dtype=np.int64)
        result = get_metrics(pred, gt_list, train_list, top_k)
        for k,v in result.items():
            results[k].append(v)

    results = {k: np.mean(v)  for k,v in results.items()}
    return results


def main(args):
    if args.gpu >= 0 and th.cuda.is_available():
        device = "cuda:{}".format(args.gpu)
        torch.backends.cudnn.benchmark=True
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.enabled=True
    else:
        device = "cpu"
    args.device=device
    data = Data(args.data_dir, args.num_workers, args.batch_size,args.num_neg)
    data_iter = data.data_iter
    args.num_rel = data.num_rel
    args.num_ent = data.num_ent
    print(args)
    data.edge_index = data.edge_index.to(device)
    data.edge_type  = data.edge_type.to(device)
    data.ent_feid  = data.ent_feid.to(device)
    model = None

    if args.model_name =='compgcn':
        if args.score_func=='dist':
            model = CompGCN_DistMult(data.edge_index,data.edge_type,data.ent_feid,args)
        elif args.score_func =='conve':
            model = CompGCN_ConvE(data.edge_index,data.edge_type,data.ent_feid,args)
    elif args.model_name=='rgat':
        if args.score_func == 'qaat':
            model = RGAT_LINK(data.edge_index,data.edge_type,data.ent_feid,args)
    elif args.model_name=='rhgat':
        if args.score_func =='conve':
            model = RHGAT_ConvE(data.edge_index,data.edge_type,data.ent_feid,args)
        elif args.score_func =='dist':
            model = RHGAT_DistMult(data.edge_index,data.edge_type,data.ent_feid,args)

    model = model.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scaler  = GradScaler()
    compiled_model = torch.compile(model,mode='reduce-overhead')

    best_epoch = -1
    best_mrr = 0.0
    kill_cnt = 0
    submit_path = "{}/preliminary_submission.json".format(args.output_dir)
    
    print('****************************')
    print('Start training...')
    for epoch in range(args.max_epochs):
        compiled_model.train()
        train_loss = []
        t0 = time()
        for step, batch in enumerate(data_iter['train']):
            triple, label = batch[0].to(device), batch[1].to(device)
            sub, rel, obj, label = (
                triple[:, 0],
                triple[:, 1],
                triple[:, 2],
                label.squeeze(),
            )
            with autocast():
                logits = compiled_model(sub, rel, obj)
                tr_loss = loss_fn(logits, label)
            train_loss.append(tr_loss.item())
            scaler.scale(tr_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # optimizer.zero_grad()
            # tr_loss.backward()
            # optimizer.step()
        train_loss = np.sum(train_loss)

        if (epoch + 1) % 20 == 0:
            t1 = time()
            val_results = evaluate(compiled_model, device, data, top_k=5)
            t2 = time()

            if val_results["MRR"] > best_mrr:
                best_mrr = val_results["MRR"]
                best_epoch = epoch
                th.save(compiled_model.state_dict(), "{}/baseline_ckpt.pth".format(args.ckpt_dir))
                kill_cnt = 0
                print("Saving model...")
            else:
                kill_cnt += 1
                if kill_cnt > 7:
                    print("Early stop. Best MRR {} at Epoch".format(best_mrr, best_epoch))
                    break
            print("In Epoch {}, Train Loss: {:.4f}, Valid MRR: {:.5}, Valid HITS: {:.5}, Train Time: {:.2f}, Valid Time: {:.2f}".format(
                    epoch, train_loss, val_results["MRR"],  val_results["HITS"], t1 - t0, t2 - t1))
                    
        else:
            t1 = time()
            print("In Epoch {}, Train Loss: {:.4f}, Train Time: {:.2f}".format(epoch, train_loss, t1 - t0))

    compiled_model.eval()
    compiled_model.load_state_dict(th.load("{}/baseline_ckpt.pth".format(args.ckpt_dir)))
    get_candidate_voter_list(compiled_model, device, data, submit_path, top_k=5)
    print("Submission file has been saved to: {}.".format(submit_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser For Arguments", formatter_class=argparse.ArgumentDefaultsHelpFormatter,)
    parser.add_argument("--data_dir", dest="data_dir", default="./data", help="Dataset dir", )
    parser.add_argument("--output_dir", dest="output_dir", default="./output", help="Output dir", )
    parser.add_argument("--ckpt_dir", dest="ckpt_dir", default="./checkpoint", help="Checkpoint dir", )
    parser.add_argument("--opn", dest="opn", default="corr", help="Composition Operation to be used in CompGCN", )
    parser.add_argument("--batch", dest="batch_size", default=10, type=int, help="Batch size" )
    parser.add_argument("--gpu", type=int, default=0, help="Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0", )
    parser.add_argument("--epoch", dest="max_epochs", type=int, default=1000, help="Number of epochs",)
    parser.add_argument("--l2", type=float, default=0.0, help="L2 Regularization for Optimizer")
    parser.add_argument("--lr", type=float, default=0.01, help="Starting Learning Rate")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of processes to construct batches",)
    parser.add_argument("--seed", dest="seed", default=41504, type=int, help="Seed for randomization", )
    parser.add_argument("--num_bases", dest="num_bases", default=-1, type=int, help="Number of basis relation vectors to use", )
    parser.add_argument("--init_dim", dest="init_dim", default=50, type=int, help="Initial dimension size for entities and relations", )
    parser.add_argument('--embed_dim',dest="embed_dim",default=200,type=int ,help="Embedding dimension to give as input to score function")
    parser.add_argument('--gcn_layer', dest='gcn_layer', default=1, type=int, help='Number of GCN Layers to use')
    parser.add_argument("--gcn_dim",dest="gcn_dim",default=200,type=int,help="Number of hidden uints of GCN layer")
    parser.add_argument("--gcn_drop", dest="dropout", default=0.1, type=float, help="Dropout to use in GCN Layer", )
    parser.add_argument('--hid_drop', dest='hid_drop', default=0.3, type=float, help='Dropout after GCN')
    parser.add_argument("--layer_dropout", nargs="?", default="[0.3]", help="List of dropout value after each compGCN layer", )
    parser.add_argument('--model_name', dest='model_name', default='rhgat', help='Gnn model as  encoder ')
    parser.add_argument('--score_func', dest='score_func', default='conve', help='Score Function for Link prediction')
    parser.add_argument('--bias', dest='bias', action='store_true', help='Whether to use bias in the model')
    parser.add_argument('--cache', dest='cache', action='store_true', help='Whether to use cache  in the gcn model')
    parser.add_argument('--num_neg', dest='num_neg', default=1,type=int, help='Number of Negative sample')
    parser.add_argument('--feature_method',dest='feature_method',default='sum',type=str,help='Feature combine method')
    # ConvE specific hyperparameters
    parser.add_argument('--hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('--feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('--k_w', dest='k_w', default=20, type=int, help='ConvE: k_w')
    parser.add_argument('--k_h', dest='k_h', default=10, type=int, help='ConvE: k_h')
    parser.add_argument('--num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('--ker_sz', dest='ker_sz', default=7, type=int, help='ConvE: Kernel size to use')

    # rGAT specific hyperparameters
    parser.add_argument('--k_kernel', dest='k_kernel', default=2, type=int, help='k Kernel to use for ent')
    parser.add_argument('--d_q', dest='d_q', default=200, type=int, help='Embedding dimension to give as input to score function')
    ### RHGAT specific hyperparameters
    parser.add_argument('--heads', dest='heads', default=4, type=int, help='multi heads for attention')
    parser.add_argument('--combine', dest='combine', default='add', type=str, help='combination method ')

    args = parser.parse_args()

    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    main(args)
