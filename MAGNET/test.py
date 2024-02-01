from random import shuffle
import torch
from torch import tensor
import torch_geometric as pyg
import matplotlib.pyplot as plt
from torch_geometric.data import Data, Dataset
from pathlib import Path
import h5py
import numpy as np
import networkx as nx
from gnn_pyg.data.dataloaders import WTA
from gnn_pyg.models.gnns import *
import argparse 
from torch import optim, nn, utils, Tensor, tensor
from tqdm import tqdm
import os 
from torch_geometric.loader import DataLoader
import random
import sys
import pickle
import time

class Tester():
    def __init__(self, model, val_loader, test_loader, args, node_to_assignment_func='na', prune=False, checkpoint_path=None, nout=1, nhid=1, depth=1):
        self.model = model
        self.val_loader = val_loader
        self.test_loader = test_loader    
        self.args = args
        self.prune = prune
        self.node_to_assignment_func = node_to_assignment_func
        if node_to_assignment_func == 'concat_mlp':
            self.mlp = MLP(nout * 2, nhid, 1, depth=depth)
            self.mlp.load_state_dict(torch.load(checkpoint_path)['assignment_mlp'])
            self.mlp.eval()

    def adjust_indexing(self, data, num_graphs):

        total_sizes = data.v_len + data.w_len
        cum_sizes = torch.cat((torch.Tensor([0]).int(), torch.cumsum(total_sizes, dim=0)), dim=0)

        v_indices = []
        w_indices = []
        for i in range(num_graphs):
            all_graph_idx = torch.arange(cum_sizes[i], cum_sizes[i+1]).long()
            v_idx = all_graph_idx[:data.v_len[i]]
            w_idx = all_graph_idx[data.v_len[i]:]
            v_indices.append(v_idx)
            w_indices.append(w_idx)

        data.v_idx = v_indices
        data.w_idx = w_indices
        data.tot_idx = cum_sizes

    def gen_label_index(self, data, num_graphs):

        label_cutoffs = []
        for i in range(num_graphs):
            label_cutoffs.append((data.v_len[i] * data.w_len[i]).item())
        label_idx = torch.cat((torch.Tensor([0]).int(), torch.cumsum(torch.Tensor(label_cutoffs), dim=0).int()), dim=0)
 
        data.label_idx = label_idx

    def unflatten_label(self, data, graph_idx):

        start_idx, end_idx = data.label_idx[graph_idx], data.label_idx[graph_idx + 1]
        graph_y = data.y[start_idx:end_idx].reshape(data.v_len[graph_idx], data.w_len[graph_idx])

        return graph_y

    def node_to_assignment(self, hid_v, hid_w):
        if self.node_to_assignment_func == 'outer_product':
            output = torch.outer(hid_v.flatten(), hid_w.flatten())
        elif self.node_to_assignment_func == 'matmul':
            output = torch.matmul(hid_v, hid_w.T)
        elif self.node_to_assignment_func == 'concat_mlp':
            # need to concat all combinations of v and w together
            # for v, makes w copies of the entire tensor
            hid_v_dup = torch.tile(hid_v, (hid_w.shape[0], 1))
            # for w, make v copes of each row 
            hid_w_dup = torch.repeat_interleave(hid_w, hid_v.shape[0], dim=0)
            # concatenate together
            hidden_vw = torch.cat((hid_v_dup, hid_w_dup), dim=1)
            output = self.mlp(hidden_vw).reshape(hid_v.shape[0], hid_w.shape[0])
        return output 

    def get_feasible(self, output, data):

        cost = data.cost_matrix
        gaps = 0
        # if use P, we can have duplicates 
        if self.args.use_P:
            feasible_out = np.zeros(data.orig_label.shape)
            # go through all nodes
            for i in range(data.orig_w):
                node_idx_set = torch.where(data.w_index == i)[0]
                node_scores = output[:, node_idx_set]
                if len(node_scores.shape) == 1:
                    # there was no expansion in this case
                    node_scores = node_scores.reshape(-1, 1)

                maxes = np.max(node_scores, axis=0)
                target_choices = np.argmax(node_scores, axis=0)
                sorted_maxes = np.argsort(-1*maxes)

                tot_cost = 0
                candidate_sol = []

                for s in sorted_maxes:
                    target_idx = target_choices[s].item()
                    
                    cost_val = cost[i, target_idx]
                    
                    # adding this assignment does not cause us to be infeasible 
                    if tot_cost + cost_val <= data.w[i]:
                        candidate_sol.append(target_idx)
                        tot_cost += cost_val
                     
                # build up non-expanded solution 
                for c in candidate_sol:
                    feasible_out[i, c] += 1
                
                # determine how much capacity was leftover 
                gaps += (data.w[i] - tot_cost)

        # if not use P, cant have duplicates    
        else:
            # didnt expand 
            feasible_out = np.zeros(data.orig_label.shape)
            # go through all nodes
            for i in range(data.orig_w):
                # get scores for that node to all targets
                node_scores = output[:, i]
                # sort the scores based on targets
                sorted_scores = np.argsort(node_scores)

                tot_cost = 0
                candidate_sol = []
                for s in sorted_scores:
                    cost_val = cost[i, s]
                    
                    # adding this assignment does not cause us to be infeasible 
                    if tot_cost + cost_val <= data.w[i]:
                        candidate_sol.append(s)
                        tot_cost += cost_val 

                # build up non-expanded solution 
                for c in candidate_sol:
                    feasible_out[i, c] += 1

                # determine how much capacity was leftover 
                gaps += (data.w[i] - tot_cost)

        return feasible_out, gaps

    def get_opt_value(self, V, P, A, minimize=False):

        P_fail = 1 - P
        P_fail_log = np.log(P_fail)

        objective = P_fail_log*A
        objective = np.sum(objective, axis=0)
        objective = np.exp(objective)
        objective = V*objective
        objective = np.sum(objective)

        if not minimize:
            objective = np.sum(V) - objective
    
     
        return objective
        


    def test_wb(self, objective=True):

        val_perf = []
        test_perf = []
        test_gt_perf = []
        test_gap = []
        test_runtime = []

        for data in self.val_loader:
            num_graphs = len(data.v_len)

            if self.args.model == 'GLAN':
                hid_rep, edge_attr = model(data.x, data.edge_index, edge_attr=data.edge_attr)
            else: 
                self.adjust_indexing(data, num_graphs)
                self.gen_label_index(data, num_graphs)
                hid_rep = self.model(data.x, data.edge_index, edge_attr=data.edge_attr)

            # with node-level representations, will now translate to assignment problem 
            for b in range(num_graphs):
                # with node-level representations, will now translate to assignment problem 
                if self.args.model == 'GLAN':
                    output = (torch.sum(edge_attr.reshape(int(edge_attr.shape[0]/2), 2), dim=1)/2.0).reshape(data.v_len, data.w_len).detach().numpy()
                else: 
                    v_idx_batch = data.v_idx[b]
                    w_idx_batch = data.w_idx[b]
                    hid_v = hid_rep[v_idx_batch,:]
                    hid_w = hid_rep[w_idx_batch,:]
                    output = self.node_to_assignment(hid_v, hid_w).detach().numpy()
                output, _ = self.get_feasible(output, data)
                label_y = data.orig_label.numpy()

                if objective:
                    opt_val_predicted = self.get_opt_value(data.v.detach().numpy(), data.prob_matrix.detach().numpy(), output)
                    opt_val_gt = self.get_opt_value(data.v.detach().numpy(), data.prob_matrix.detach().numpy(), label_y)
                    val_perf.append(objective)
                else: 
                    hid_rep = output.flatten()
                    data_y = label_y.flatten()
                    N = len(hid_rep)
                    TP = 0
                    FP = 0 
                    FN = 0
                    for h in range(N):
                        if hid_rep[h] == 0 and data_y[h] == 0:
                            continue
                        else:
                            max_amnt = int(np.max((hid_rep[h], data_y[h])))
                            pred = np.zeros(max_amnt)
                            pred[:int(hid_rep[h])] = 1
                            gt = np.zeros(max_amnt)
                            gt[:int(data_y[h])] = 1
                            
                            TP += ((pred == 1) & (gt == 1)).sum()
                            FP += ((pred == 1) & (gt == 0)).sum()
                            FN += ((pred == 0) & (gt == 1)).sum()
                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN)
                    if TP == 0:
                        f1_score = 0
                    else: 
                        f1_score = (2*precision*recall)/(precision+recall)
                    val_perf.append(f1_score)
          
        for data in self.test_loader:
            num_graphs = len(data.v_len)
            start_time = time.time()

            if self.args.model == 'GLAN':
                hid_rep, edge_attr = model(data.x, data.edge_index, edge_attr=data.edge_attr)
            else: 
                self.adjust_indexing(data, num_graphs)
                self.gen_label_index(data, num_graphs)
                hid_rep = self.model(data.x, data.edge_index, edge_attr=data.edge_attr)

            # with node-level representations, will now translate to assignment problem 
            for b in range(num_graphs):
                # with node-level representations, will now translate to assignment problem 
                if self.args.model == 'GLAN':
                    output = (torch.sum(edge_attr.reshape(int(edge_attr.shape[0]/2), 2), dim=1)/2.0).reshape(data.v_len, data.w_len).detach().numpy()
                else: 
                    v_idx_batch = data.v_idx[b]
                    w_idx_batch = data.w_idx[b]
                    hid_v = hid_rep[v_idx_batch,:]
                    hid_w = hid_rep[w_idx_batch,:]
                    output = self.node_to_assignment(hid_v, hid_w).detach().numpy()
                output, gaps = self.get_feasible(output, data)

                time_diff = (time.time() - start_time) + data.time.item()
                label_y = data.orig_label.numpy()

                if objective:
                    opt_val_predicted = self.get_opt_value(data.v.detach().numpy(), data.prob_matrix.detach().numpy(), output)
                    opt_val_gt = self.get_opt_value(data.v.detach().numpy(), data.prob_matrix.detach().numpy(), label_y)
                    test_perf.append(opt_val_predicted)
                    test_gt_perf.append(opt_val_gt)
                else: 
                    hid_rep = output.flatten()
                    data_y = label_y.flatten()
                    N = len(hid_rep)
                    TP = 0
                    FP = 0 
                    FN = 0
                    for h in range(N):
                        if hid_rep[h] == 0 and data_y[h] == 0:
                            continue
                        else:
                            max_amnt = int(np.max((hid_rep[h], data_y[h])))
                            pred = np.zeros(max_amnt)
                            pred[:int(hid_rep[h])] = 1
                            gt = np.zeros(max_amnt)
                            gt[:int(data_y[h])] = 1
                            
                            TP += ((pred == 1) & (gt == 1)).sum()
                            FP += ((pred == 1) & (gt == 0)).sum()
                            FN += ((pred == 0) & (gt == 1)).sum()

                    precision = TP / (TP + FP)
                    recall = TP / (TP + FN)
                    if TP == 0:
                        f1_score = 0
                    else: 
                        f1_score = (2*precision*recall)/(precision+recall)
                    
                    test_perf.append(f1_score)
                test_gap.append(gaps.item())
                test_runtime.append(time_diff)
               
        if objective: 
            return val_perf, test_perf, test_gap, test_runtime, test_gt_perf
        else:
            return val_perf, test_perf, test_gap, test_runtime

    def test_line_graph(self,  objective=True):

        val_perf = []
        test_perf = []
        test_gt_perf = []
        test_gap = []
        test_runtime = []

        for data in self.val_loader:
            hid_rep = self.model(data.x, data.edge_index, edge_attr=data.edge_attr)
            hid_rep = hid_rep.flatten().reshape(data.v_len, data.w_len)

            hid_rep = hid_rep.detach().numpy()
            output, _ = self.get_feasible(hid_rep, data)
            label_y = data.orig_label.numpy()
            
            if objective:
                #print(data.optimal_val)
                opt_val_predicted = self.get_opt_value(data.v.detach().numpy(), data.prob_matrix.detach().numpy(), output)
                opt_val_gt = self.get_opt_value(data.v.detach().numpy(), data.prob_matrix.detach().numpy(), label_y)
                val_perf.append(opt_val_predicted)


                #print(opt_val_predicted)
                #print(opt_val_gt)
            else: 
                hid_rep = hid_rep.flatten()
                data_y = label_y.flatten()
                N = len(hid_rep)
                TP = 0
                FP = 0 
                FN = 0
                for h in range(N):
                    if hid_rep[h] == 0 and data_y[h] == 0:
                        continue
                    else:
                        max_amnt = int(np.max((hid_rep[h], data_y[h])))
                        pred = np.zeros(max_amnt)
                        pred[:int(hid_rep[h])] = 1
                        gt = np.zeros(max_amnt)
                        gt[:int(data_y[h])] = 1
                        
                        TP += ((pred == 1) & (gt == 1)).sum()
                        FP += ((pred == 1) & (gt == 0)).sum()
                        FN += ((pred == 0) & (gt == 1)).sum()
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                if TP == 0:
                    f1_score = 0
                else: 
                    f1_score = (2*precision*recall)/(precision+recall)
                    
                val_perf.append(f1_score)

        for d, data in enumerate(self.test_loader):
            start_time = time.time()

            hid_rep = self.model(data.x, data.edge_index, edge_attr=data.edge_attr, prune=self.prune, v_len=data.v_len, w_len=data.w_len)
            hid_rep = hid_rep.flatten().reshape(data.v_len, data.w_len)

            # target x wep 
            hid_rep = hid_rep.detach().numpy()
            output, gaps = self.get_feasible(hid_rep, data)
            label_y = data.orig_label.numpy()
            
            time_diff = (time.time() - start_time) + data.time.item()

            if objective:
                opt_val_predicted = self.get_opt_value(data.v.detach().numpy(), data.prob_matrix.detach().numpy(), output)
                opt_val_gt = self.get_opt_value(data.v.detach().numpy(), data.prob_matrix.detach().numpy(), label_y)
                #print(opt_val_predicted)
                #print(opt_val_gt)
                test_perf.append(opt_val_predicted)
                test_gt_perf.append(opt_val_gt)
            else: 
                hid_rep = hid_rep.flatten()
                data_y = label_y.flatten()
                N = len(hid_rep)
                TP = 0
                FP = 0 
                FN = 0
                for h in range(N):
                    if hid_rep[h] == 0 and data_y[h] == 0:
                        continue
                    else:
                        max_amnt = int(np.max((hid_rep[h], data_y[h])))
                        pred = np.zeros(max_amnt)
                        pred[:int(hid_rep[h])] = 1
                        gt = np.zeros(max_amnt)
                        gt[:int(data_y[h])] = 1
                        
                        TP += ((pred == 1) & (gt == 1)).sum()
                        FP += ((pred == 1) & (gt == 0)).sum()
                        FN += ((pred == 0) & (gt == 1)).sum()

                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                if TP == 0:
                    f1_score = 0
                else: 
                    f1_score = (2*precision*recall)/(precision+recall)
            
                test_perf.append(f1_score)
            test_gap.append(gaps.item())
            test_runtime.append(time_diff)
        
        if objective: 
            return val_perf, test_perf, test_gap, test_runtime, test_gt_perf
        else:
            return val_perf, test_perf, test_gap, test_runtime





if __name__ == '__main__':

    parser = argparse.ArgumentParser("milp")
    #parser.add_argument("--path", type=str, default='gnn_pyg/data/generate/assignment_data/data_2023-10-03_23-34-13.hdf5', help="Full path for the data to be saved to.")
    #parser.add_argument("--path", type=str, default="gnn_pyg/data/generate/assignment_data/data_2023-10-04_17-09-04.hdf5", help="Full path for the data to be saved to.")
    parser.add_argument("--data_path", type=str, default="gnn_pyg/data/generate/assignment_data/data_2023-10-07_18-48-05.hdf5", help="Full path for the data to be saved to.")
    
    parser.add_argument("--seed", type=int, default=123, help='seed to use for various random components of code, such as data shuffling')

    parser.add_argument("--model", type=str, default='GLAN',  help='GNN model to use')
    parser.add_argument('--nhids', type=int, default=64, help='hidden dim for GNN model')
    parser.add_argument('--depths', type=int, default=8, help='depth of GNN model')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout probability')
    parser.add_argument('--node_to_assignment_func', type=str, default='na', choices=['na', 'outer_product', 'matmul', 'concat_mlp'], help='method to converting learned node representations into final assignment matrix')

    parser.add_argument('--lrs', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--regularize_l1s', type=float, default=0.5)

    args = parser.parse_args()
    
    objective = True

    reprs = ['line_graph', 'weighted_bipartite']
    paths = ['gnn_pyg/data/generate/assignment_data/data_2023-10-03_23-34-13.hdf5', "gnn_pyg/data/generate/assignment_data/data_2023-10-04_17-09-04.hdf5"]
    path_name = [p.split('/')[-1].rstrip('.hdf5') for p in paths] 
    # different experiments 
    test_dict = {}

    for p, path in enumerate(paths): 
        args.path = path
        if p >= 0:
            args.data_path = path
            test_data = False
        else:
            test_data = True
        for representation in reprs: 
            args.repr = representation

            if args.repr == 'line_graph':
                prune = False
            else:
                prune = False

            dataset = WTA(args.data_path, repr=args.repr)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Get with no idx gives all graphs 
            data = dataset.get(device=device)
            data_size = len(data)
            dataset_name = args.path.split('/')[-1].rstrip('.hdf5')
            
            # if probabilistic, use expansions
            args.use_P = data[0].use_P
            args.use_C = data[0].use_C   

            nhids = [16, 32]
            depths = [2, 3]
            dropout = 0.3
            lrs = [0.001, 0.0001]
            models = ['GCNII', 'GCN', 'FAGCN', 'GPRGNN', 'GLAN']

            for model_type in models:
                print(model_type)
                if model_type == 'GLAN':
                    if args.repr == 'line_graph':
                        continue 
                    node_to_assignment_funcs = ['na']
                    regs = [0.0]
                    nout = 1
                else:
                    regs = [0.0, 0.5]
                    if args.repr == 'line_graph':
                        node_to_assignment_funcs = ['na']
                        nout = 1
                    else:
                        node_to_assignment_funcs = ['matmul', 'concat_mlp']
                        nout = 16

                for node_to_assignment_func in node_to_assignment_funcs:
                    args.model = model_type
                    args.node_to_assignment_func = node_to_assignment_func
        
                    
                    if args.repr == 'line_graph':
                        save_folder = 'gnn_pyg/models_results/{4}/{0}_{1}_{2}_{3}/'.format('wta', args.model, args.repr, args.seed, dataset_name)
                    else:
                        save_folder = 'gnn_pyg/models_results/{4}/{0}_{1}_{2}_{5}_{3}/'.format('wta', args.model, args.repr, args.seed, dataset_name, args.node_to_assignment_func)

                    random.seed(args.seed)
                    # Get data loader which batches elements

                    random.shuffle(data)

                    if test_data:
                        val_cutoff = round(0.8 * data_size)
                        val_loader = DataLoader(data[:val_cutoff], batch_size=1, shuffle=True)
                        test_loader = DataLoader(data[val_cutoff:], batch_size=1, shuffle=True)
                        print('this')
                    else: 
                        train_cutoff = round(0.8 * data_size)
                        val_cutoff = round(0.9 * data_size)

                        # get same val and test data
                        train_loader = DataLoader(data[:train_cutoff], batch_size=1, shuffle=True)
                        val_loader = DataLoader(data[train_cutoff:val_cutoff], batch_size=1, shuffle=True)
                        test_loader = DataLoader(data[val_cutoff:], batch_size=1, shuffle=True)
                    
                    # get model with best val and save test loss 
                    best_val_perf = 0

                    for nhid in nhids:
                        for depth in depths: 
                            for lr in lrs: 
                                for reg in regs:
                                # get model
                                    if args.model == 'GCN':
                                        model = GCN(data[0].x.shape[1], nhid, nout, depth=depth, dropout=dropout, act='relu', args=args)
                                    elif args.model == 'GCNII':
                                        model = GCNII(data[0].x.shape[1], nhid, nout, depth=depth, dropout=dropout, alpha=0.1, act='relu', args=args)
                                    elif args.model == 'GPRGNN':
                                        model = GPRGNN(data[0].x.shape[1], nhid, nout, depth=depth, dropout=dropout, K=10, alpha=0.1, act='relu', args=args)
                                    elif args.model == 'FAGCN':
                                        model = FAGCN(data[0].x.shape[1], nhid, nout, depth=depth, dropout=dropout, eps=0.1, act='relu', args=args)
                                    elif args.model == 'GLAN':
                                        model = GLAN(data[0].x.shape[1], nhid, nout, depth=depth, dropout=dropout, act='relu', args=args)
                                        

                                    model_checkpoint_path = save_folder + 'nhid_{0}_depth_{1}_nout_{2}_lr_{3}_reg_{4}.pth'.format(nhid, depth, nout, lr, reg)
                                    model.load_state_dict(torch.load(model_checkpoint_path)['model'])
                                    model.eval()

                                    if args.repr == 'weighted_bipartite':
                                        tester = Tester(model, val_loader, test_loader, args, node_to_assignment_func=args.node_to_assignment_func, checkpoint_path=model_checkpoint_path, nout=nout, nhid=nhid, depth=depth)
                                        if objective: 
                                            val_perf, test_perf, test_gap, test_runtime, test_perf_gt = tester.test_wb(objective=objective)
                                        else: 
                                            val_perf, test_perf, test_gap, test_runtime = tester.test_wb(objective=objective)
                                    else:
                                        tester = Tester(model, val_loader, test_loader, args, checkpoint_path=model_checkpoint_path, nout=nout, nhid=nhid, depth=depth, prune=prune)
                                        if objective:
                                            val_perf, test_perf, test_gap, test_runtime, test_perf_gt = tester.test_line_graph(objective=objective)
                                        else: 
                                            val_perf, test_perf, test_gap, test_runtime = tester.test_line_graph(objective=objective)

                                    #print(val_perf)
                                    avg_val_perf = sum(val_perf)/len(val_perf)
                                    #print(avg_val_perf)
                                    if avg_val_perf >= best_val_perf:
                                        #print(test_perf_gt)
                                        #print(test_perf)
                                        best_val_perf = avg_val_perf
                                        best_test_perf = sum(test_perf)/len(test_perf)
                                        best_test_gap = sum(test_gap)/len(test_gap)
                                        best_test_runtime = sum(test_runtime)/len(test_runtime)
                                        best_runtime_std = np.std(test_runtime)
                                        if objective: 
                                            optimality_gaps = (np.array(test_perf_gt) - np.array(test_perf))/np.array(test_perf_gt) 
                                            #print(optimality_gaps)
                                            
                                            best_optimality_gap = sum(optimality_gaps)/len(optimality_gaps)
                                            best_opt_std = np.std(optimality_gaps)
                                            #print(best_optimality_gap)
                    
                    if objective:
                        best_test_perf = best_optimality_gap
                        best_opt_std = best_opt_std
                        #print(best_test_perf)
                    if args.repr == 'line_graph':
                        test_dict['linegraph' + '_' + model_type + '_' + node_to_assignment_func + '_' + path.split('/')[-1].rstrip('.hdf5')] = (round(best_test_perf, 2), \
                                                                                                                                            round(best_opt_std, 2), \
                                                                                                                                            round(best_test_gap, 2), \
                                                                                                                                            round(best_test_runtime,4), \
                                                                                                                                            round(best_runtime_std,4))
                    else: 
                        if node_to_assignment_func == 'concat_mlp':
                            node_to_assignment_func_name = 'concatmlp'
                        else:
                            node_to_assignment_func_name = node_to_assignment_func
                        test_dict['weightedbipartite' + '_' + model_type + '_' + node_to_assignment_func_name + '_' + path.split('/')[-1].rstrip('.hdf5')] = (round(best_test_perf, 2), \
                                                                                                                                            round(best_opt_std, 2), \
                                                                                                                                            round(best_test_gap, 2), \
                                                                                                                                            round(best_test_runtime,4), \
                                                                                                                                            round(best_runtime_std,4))

            


    #print(test_dict)

                
    for method in ['matmul', 'concatmlp', 'GLAN', 'line']:
        sub_test_dict = {key: value for (key, value) in test_dict.items() if method in key}
        if method != 'GLAN':
            for model in ['_GCN_', 'GCNII', 'FAGCN', 'GPRGNN']:
                sub_model_test_dict = {key: value for (key, value) in sub_test_dict.items() if model in key}
                
                sub_model_data_test_dict_p0 = {key: value for (key, value) in sub_model_test_dict.items() if path_name[0] in key}
                sub_model_data_test_dict_p1 = {key: value for (key, value) in sub_model_test_dict.items() if path_name[1] in key}
                # only one key here
                values_p0 = list(sub_model_data_test_dict_p0.values())[0]
                values_p1 = list(sub_model_data_test_dict_p1.values())[0]

                key = list(sub_model_data_test_dict_p0.keys())[0]

                repres, model_name, assign, _, _, _ = key.split('_')
                if assign == 'matmul':
                    model_aggr = 'Node2Vec'
                elif assign == 'concatmlp':
                    model_aggr = 'Edge2Vec'
                # na case 
                else:
                    model_aggr = 'Line'
                
                if model == '_GCN_':
                    model_name = 'GCN'
                else:
                    model_name = model 


                if method == 'line':
                    prnt_str = r'{0}-{1} & - & \checkmark & ${2} \pm {3}$ & ${4} \pm {5}$ & ${6} \pm {7}$ & ${8} \pm {9}$ \\ '.format(model_name, model_aggr, values_p0[0], values_p0[1], values_p0[3], values_p0[4], \
                                                                                                                              values_p1[0], values_p1[1], values_p1[3], values_p1[4])
                else:
                    prnt_str = r'{0}-{1} & \checkmark & - & ${2} \pm {3}$ & ${4} \pm {5}$ & ${6} \pm {7}$ & ${8} \pm {9}$ \\ '.format(model_name, model_aggr, values_p0[0], values_p0[1], values_p0[3], values_p0[4], \
                                                                                                                              values_p1[0], values_p1[1], values_p1[3], values_p1[4])
                if model == 'GPRGNN':
                    prnt_str += ' \hline'
                print(prnt_str)

        else: 
            #print(sub_test_dict)
            sub_model_data_test_dict_p0 = {key: value for (key, value) in sub_test_dict.items() if path_name[0] in key}
            sub_model_data_test_dict_p1 = {key: value for (key, value) in sub_test_dict.items() if path_name[1] in key}
            # only one key here
            values_p0 = list(sub_model_data_test_dict_p0.values())[0]
            values_p1 = list(sub_model_data_test_dict_p1.values())[0]

            key = list(sub_model_data_test_dict_p0.keys())[0]

            repres, model_name, assign, _, _, _ = key.split('_')
            print(r'{0} & \checkmark & - & ${2} \pm {3}$ & ${4} \pm {5}$ & ${6} \pm {7}$ & ${8} \pm {9}$ \\ \hline '.format(model_name, '_', values_p0[0], values_p0[1], values_p0[3], values_p0[4], \
                                                                                                                    values_p1[0], values_p1[1], values_p1[3], values_p1[4]))
