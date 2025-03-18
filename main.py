import os
import time
import glob
import argparse
import csv

import torch
import torch.nn.functional as F
from torch_geometric.transforms import Constant, OneHotDegree
from torch_geometric.utils import degree
import numpy as np
import pandas as pd 

from model import Model
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=200, help='random seed')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--lamda_im', type=float, default=0.2, help='parameter of loss_im')
parser.add_argument('--lamda_pp', type=float, default=0.1, help='parameter of loss_pp')
parser.add_argument('--lamda_sep', type=float, default=1.0, help='parameter of loss_sep')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--dropout_ratio', type=float, default=0.5, help='dropout ratio')
parser.add_argument('--tau', type=float, default=1.5, help='temperature in info-nce loss')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--data_path', type=str, default='.', help='data path')
parser.add_argument('--source', type=str, default='DBLPv7', help='source domain data')
parser.add_argument('--target', type=str, default='ACMv9', help='target domain data')
parser.add_argument('--epochs', type=int, default=200, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
    
args = parser.parse_args()

if args.source in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), args.data_path, 'data/Citation', args.source)
    source_dataset = CitationDataset(path, args.source)
if args.target in {'DBLPv7', 'ACMv9', 'Citationv1'}:
    path = osp.join(osp.dirname(osp.realpath(__file__)), args.data_path, 'data/Citation', args.target)
    target_dataset = CitationDataset(path, args.target)

source_data = source_dataset[0] 
args.num_classes = len(np.unique(source_data.y.numpy()))
args.num_features = source_data.x.size(1)
source_data = source_data.to(args.device)
target_data = target_dataset[0].to(args.device)
    
print(args)


def train():
    min_loss = 1e10
    patience_cnt = 0
    loss_values = []
    best_epoch = 0
    
    model = Model(args).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    t = time.time()
    model.train()

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        correct = 0

        source_feat = model.feat_bottleneck(source_data, source_data.edge_index)
        target_feat = model.feat_bottleneck(target_data, target_data.edge_index)
        
        target_cls = model.feat_classifier(target_feat, target_data.edge_index)
        target_prob = F.softmax(target_cls, dim=1)
        loss_im, _, _, ent_loss_t = cls_im(target_prob)
        
        source_proto = obtain_source_prototype(source_feat, source_data.y)
        target_proto = obtain_target_prototype(target_prob, target_feat, 
                                               target_data.edge_index, ent_loss_t)
        loss_pp = proto_alignment(target_proto, source_proto)
        
        loss_sep = seperate_center(source_proto, source_feat, source_data.y)
        
        output = model(source_data, source_data.edge_index)
        train_loss = F.nll_loss(output, source_data.y)
        
        loss = train_loss + args.lamda_im * loss_im + \
                args.lamda_pp * loss_pp + args.lamda_sep * loss_sep
        
        loss.backward()
        optimizer.step()
        
        with torch.no_grad():
            pred = output.max(dim=1)[1]
            correct = pred.eq(source_data.y).sum().item()
            train_acc = correct * 1.0 / len(source_data.y)
    
            macro_f1, micro_f1, test_loss = evaluate(target_data, model)

            print('Epoch: {:04d}'.format(epoch + 1), 'test_loss: {:.6f}'.format(test_loss),
                  'macro_f1: {:.6f}'.format(macro_f1), 'micro_f1: {:.6f}'.format(micro_f1),\
                  'time: {:.6f}s'.format(time.time() - t))

        loss_values.append(loss)
        torch.save(model.state_dict(), '{}.pth'.format(epoch))
        
        if loss_values[-1] < min_loss:
            min_loss = loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(f)
    time_use = time.time() - t
    print('Optimization Finished! Total time elapsed: {:.6f}'.format(time_use))

    return model, best_epoch, time_use
    

def cls_im(prob):
    mean_prob = prob.mean(dim=0)
    div_loss = torch.sum(mean_prob * torch.log(mean_prob + 1e-12))
    ent_loss_temp = - torch.sum(prob * torch.log(prob + 1e-12), dim=1)
    ent_loss = torch.mean(ent_loss_temp)
    loss_im = div_loss + ent_loss
    
    return loss_im, div_loss, ent_loss, ent_loss_temp
        
    
def obtain_source_prototype(feat, label):
    onehot = torch.eye(args.num_classes).to(args.device)[label]
    center = torch.mm(feat.t(), onehot) / (onehot.sum(dim=0))
    
    return center.t()


def obtain_target_prototype(prob, feat, edge_index, ent):
    pagerank = PageRank().to(args.device)
    num_nodes = feat.size(0)
    ent = ent.unsqueeze(-1)
    _, pred = torch.max(prob, dim=1)
    onehot = torch.eye(args.num_classes).to(args.device)[pred]
    biased_center = (torch.mm(feat.t(), onehot) / (onehot.sum(dim=0) + 1e-12)).t()
    
    matrix = []
    for i in range(args.num_classes):
        if torch.sum(pred == i) == 0:
            matrix.append(torch.zeros(1, args.num_classes).to(args.device))
        else:
            matrix.append(torch.mean(prob[pred == i], dim=0, keepdim=True))
    matrix = torch.cat(matrix)
    eye = torch.eye(args.num_classes, args.num_classes).to(args.device)
    matrix = F.normalize(matrix * (1 - eye), dim=1)
    matrix_every = prob * (matrix[pred])
    matrix_every = F.normalize(matrix_every, dim=1)
    bias_center_every = biased_center[pred]
    unbias_center_every = torch.mm(matrix_every, biased_center) \
                            / torch.sum(matrix_every, dim=1, keepdim=True)
    unbias_center_every = F.normalize(pagerank(unbias_center_every - bias_center_every, edge_index)\
                                      , dim=1)
    center_every_tt = bias_center_every + ent * unbias_center_every
    center_every = (torch.mm(center_every_tt.t(), onehot) / (onehot.sum(dim=0) + 1e-12)).t()
    
    return center_every


def seperate_center(center, feat, label):
    num_nodes = feat.size(0)
    proto_norm = F.normalize(center, dim=1)
    sim = torch.matmul(proto_norm, proto_norm.t())
    sim = torch.exp(sim / args.tau)
    pos_sim = sim[range(args.num_classes), range(args.num_classes)]
    loss = (sim.sum(dim=1) - pos_sim) / (args.num_classes - 1)
    loss = torch.log(loss + 1e-8) 
    loss = torch.mean(loss)
    
    return loss


def proto_alignment(target_proto, source_proto):
    num_proto = target_proto.size(0)

    target_norm = F.normalize(target_proto, dim=1)
    source_norm = F.normalize(source_proto, dim=1)

    sim_matrix_ts = torch.matmul(target_norm, source_norm.t())
    sim_matrix_ts = torch.exp(sim_matrix_ts / args.tau)
    pos_sim_ts = sim_matrix_ts[range(num_proto), range(num_proto)]

    sim_matrix_tt = torch.matmul(target_norm, target_norm.t())
    sim_matrix_tt = torch.exp(sim_matrix_tt / args.tau)
    pos_sim_tt = sim_matrix_tt[range(num_proto), range(num_proto)]

    sim_matrix_ss = torch.matmul(source_norm, source_norm.t())
    sim_matrix_ss = torch.exp(sim_matrix_ss / args.tau)
    pos_sim_ss = sim_matrix_ss[range(num_proto), range(num_proto)]

    denominator = sim_matrix_ts.sum(dim=1) - pos_sim_ts \
                    + sim_matrix_tt.sum(dim=1) - pos_sim_tt \
                    + sim_matrix_ss.sum(dim=1) - pos_sim_ss
    logit = pos_sim_ts / (denominator + 1e-12)
    loss = - torch.log(logit + 1e-12).mean()

    return loss
        


if __name__ == '__main__':
    macro_f1_dict = []
    micro_f1_dict = []
    time_dict = []
    
    for i in range(1):
        # Model training
        model, best_model, time_use = train()
        # Restore best model for test set
        model.load_state_dict(torch.load('{}.pth'.format(best_model)))

        macro_f1, micro_f1, test_loss = evaluate(target_data, model)
        print('Target {} all set results, loss = {:.6f}, macro_f1 = {:.6f}, micro_f1 = {:.6f}'\
              .format(args.target, test_loss, macro_f1, micro_f1))
        
        macro_f1_dict.append(macro_f1)
        micro_f1_dict.append(micro_f1)
        time_dict.append(time_use)
    
    macro_f1_dict_print = [float('{:.6f}'.format(i)) for i in macro_f1_dict]
    micro_f1_dict_print = [float('{:.6f}'.format(i)) for i in micro_f1_dict]
    print(macro_f1_dict_print,  'mean {:.4f}'.format(np.mean(macro_f1_dict)),\
                                ' std {:.4f}'.format(np.std(macro_f1_dict)))
    print(micro_f1_dict_print,  'mean {:.4f}'.format(np.mean(micro_f1_dict)), \
                                ' std {:.4f}'.format(np.std(micro_f1_dict)))
    print('time use mean {:.4f}'.format(np.mean(time_dict)),\
          ' std {:.4f}'.format(np.std(time_dict)))
