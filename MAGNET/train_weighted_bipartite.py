from random import shuffle
import torch
from torch import tensor
import torch.functional as F
import torch_geometric as pyg
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
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from torch_geometric.loader import DataLoader
import random
import sys
import pickle

class Hook(list):
    """ Hook to capture module outputs.
    """
    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            # GPRGNN returns both x and hid representation, just want hid 
            self.append(output[1])
        else:
            self.append(output)

class Trainer():
    def __init__(self, model, train_loader, val_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args 

        # attach hook to grab intermediates 
        if self.args.regularize_l1 > 0.0:
            self.intermed_hook = Hook()
            # SGC and GPRGNN does not have activation functions during propogation, thus will take directly from convolution
            if args.model == 'GPRGNN':
                # in wb graph use linear layer
                if args.use_P and args.use_C:
                    self.model.lins_prob_cost_hid.register_forward_hook(self.intermed_hook)
                else: 
                    self.model.conv_prob.register_forward_hook(self.intermed_hook)
            else:
                self.model.act.register_forward_hook(self.intermed_hook)

        if args.node_to_assignment_func == 'concat_mlp':
            self.mlp = MLP(args.nout * 2, args.nhid, 1, depth=args.depth)
        
        self.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(args.pos_weight))
     
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
        if self.args.model == 'GLAN':
            # dont need to do anything 
            pass
        else:
            if self.args.node_to_assignment_func == 'outer_product':
                output = torch.outer(hid_v.flatten(), hid_w.flatten())
            elif self.args.node_to_assignment_func == 'matmul':
                output = torch.matmul(hid_v, hid_w.T)
            elif self.args.node_to_assignment_func == 'concat_mlp':
                # need to concat all combinations of v and w together
                # for v, makes w copies of the entire tensor
                hid_v_dup = torch.tile(hid_v, (hid_w.shape[0], 1))
                # for w, make v copes of each row 
                hid_w_dup = torch.repeat_interleave(hid_w, hid_v.shape[0], dim=0)
                # concatenate together
                hidden_vw = torch.cat((hid_v_dup, hid_w_dup), dim=1)
                output = self.mlp(hidden_vw).reshape(hid_v.shape[0], hid_w.shape[0])
        return output 

    def get_dead_indices(self, hid_rep, data_y):
        # determine which duplicated nodes are not needed and remove them from backpropogation 
        return torch.where(data_y.sum(dim=0) != 0.0)[0]
        
    def train_wta(self):

        train_losses = []
        val_losses = []

        for epoch in tqdm(range(self.args.epochs)):
            losses = []
            # iterating through training graphs 
            for data in self.train_loader: 
                self.optimizer.zero_grad()
                
                # if GLAN, we do not need an assignment function 
                if args.model == 'GLAN':
                    hid_rep, edge_attr = self.model(data.x, data.edge_index, edge_attr=data.edge_attr)

                    avg_edge = (torch.sum(edge_attr.reshape(int(edge_attr.shape[0]/2), 2), dim=1)/2.0).reshape(data.v_len, data.w_len)
                    y = data.y.reshape(data.v_len, data.w_len)
                    
                    loss = self.criterion(avg_edge, y.float())
                    loss.backward()
                        
                    losses.append(loss.item())
                else: 
                    hid_rep = self.model(data.x, data.edge_index, edge_attr=data.edge_attr)
                    
                    # after performing all message passing, need to use each graph seperately
                    # TODO: try to speed this up later 
                    num_graphs = len(data.v_len)
                    self.adjust_indexing(data, num_graphs)
                    self.gen_label_index(data, num_graphs)
                    for b in range(num_graphs):
    
                        # with node-level representations, will now translate to assignment problem 
                        v_idx_batch = data.v_idx[b]
                        w_idx_batch = data.w_idx[b]
                        hid_v = hid_rep[v_idx_batch,:]
                        hid_w = hid_rep[w_idx_batch,:]

                        hid_rep = self.node_to_assignment(hid_v, hid_w)

                        # data.y is flat vector, need to turn back into 2D
                        data_y = self.unflatten_label(data, b)

                        idx = self.get_dead_indices(hid_rep, data_y)
                        loss = self.criterion(hid_rep[:, idx], data_y.float()[:, idx])

                        if self.args.regularize_l1 > 0.0:
                            for output in self.intermed_hook:
                                loss += (torch.norm(output, 1) * args.regularize_l1)

                        # must retain since we backward over each element in batch
                        loss.backward() 
                        if self.args.regularize_l1 > 0.0:
                            self.intermed_hook.clear()

                        losses.append(loss.item())

                # accumulate gradients and do final update 
                self.optimizer.step() 
            avg_loss = sum(losses)/len(losses)
            train_losses.append(avg_loss) 

        return train_losses

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser("milp")

    parser.add_argument('--data_type', type=str, default='wta', choices=['wta'], help='data type to generate, changes shape of output')
    parser.add_argument("--path", type=str, default="gnn_pyg/data/generate/lp_data/data_06-07-2023_15-30-20.hdf5", help="Full path for the data to be saved to.")
    parser.add_argument("--seed", type=int, default=123, help='seed to use for various random components of code, such as data shuffling')

    parser.add_argument("--model", type=str, default='GCN',  help='GNN model to use')
    parser.add_argument('--nhid', type=int, default=64, help='hidden dim for GNN model')
    parser.add_argument('--depth', type=int, default=8, help='depth of GNN model')
    parser.add_argument('--nout', type=int, default=1, help='output dim that will be used as intermediary representation for final assignment')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout probability')
    parser.add_argument('--node_to_assignment_func', type=str, default='na', choices=['na', 'outer_product', 'matmul', 'concat_mlp'], help='method to converting learned node representations into final assignment matrix')

    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--regularize_l1', type=float, default=0.0)

    args = parser.parse_args()

    random.seed(args.seed)


    print(args.path)
    # set up naming
    dataset_name = args.path.split('/')[-1].rstrip('.hdf5')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device 
    args.repr = 'weighted_bipartite'

    # Load in dataset 
    dataset = WTA(args.path, repr=args.repr)
    # Get with no idx gives all graphs 
    data = dataset.get(device=device)

    print(data[0].use_P)
    print(data[0].use_C)

    # if probabilistic, use expansions
    args.use_P = data[0].use_P
    args.use_C = data[0].use_C    

    if args.regularize_l1 > 0.0:
        print(args.regularize_l1)
        args.intermed_return = True 

    if (args.node_to_assignment_func == 'outer_product') and (args.nout != 1):
       print('If using outer product, can only have 1D vector output')
       sys.exit()

    save_folder = 'gnn_pyg/models_results/{0}/{1}_{2}_{3}_{4}_{5}/'.format(dataset_name, 'wta', args.model, 'weighted_bipartite', args.node_to_assignment_func, args.seed)

    os.makedirs(save_folder, exist_ok=True)

    if os.path.isfile(save_folder + 'nhid_{0}_depth_{1}_nout_{2}_lr_{3}_reg_{4}.pth'.format(args.nhid, args.depth, args.nout, args.lr, args.regularize_l1)):
        print("Already processed")
        sys.exit()

    # Get data loader which batches elements
    data_size = len(data)
   
    random.shuffle(data)
    train_cutoff = round(0.8 * data_size)
    val_cutoff = round(0.9 * data_size)
    
    train_loader = DataLoader(data[:train_cutoff], batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(data[train_cutoff:val_cutoff], batch_size=args.batch_size, shuffle=True)

    # get model
    if args.model == 'GCN':
        model = GCN(data[0].x.shape[1], args.nhid, args.nout, depth=args.depth, dropout=args.dropout, act='relu', args=args)
    elif args.model == 'GCNII':
        model = GCNII(data[0].x.shape[1], args.nhid, args.nout, depth=args.depth, dropout=args.dropout, alpha=0.1, act='relu', args=args)
    elif args.model == 'GPRGNN':
        model = GPRGNN(data[0].x.shape[1], args.nhid, args.nout, depth=args.depth, dropout=args.dropout, K=10, alpha=0.1, act='relu', args=args)
    elif args.model == 'FAGCN':
        model = FAGCN(data[0].x.shape[1], args.nhid, args.nout, depth=args.depth, dropout=args.dropout, eps=0.1, act='relu', args=args)
    elif args.model == 'GLAN':
        model = GLAN(data[0].x.shape[1], args.nhid, args.nout, depth=args.depth, dropout=args.dropout, act='relu', args=args)
    print(model)

    # get the loss weight based on the number of targets
    neg = (data[0].y == 0.0).sum()
    pos = (data[0].y).sum()

    pos_weight = neg/pos
    args.pos_weight = pos_weight

    trainer = Trainer(model, train_loader, val_loader, args)
    
    train_losses = trainer.train_wta() 

    # Save the models 
    if args.node_to_assignment_func != 'concat_mlp':
        save_models = {'model': trainer.model.state_dict()}
    else: 
        save_models = {'model': trainer.model.state_dict(), 'assignment_mlp': trainer.mlp.state_dict()}
    torch.save(save_models, save_folder + 'nhid_{0}_depth_{1}_nout_{2}_lr_{3}_reg_{4}.pth'.format(args.nhid, args.depth, args.nout, args.lr, args.regularize_l1))

    results = {
            'train_losses': train_losses,
        }

    if args.regularize_l1 > 0.0:
        with open(save_folder + 'nhid_{0}_depth_{1}_nout_{2}_lr_{3}_reg_{4}.pkl'.format(args.nhid, args.depth, args.nout, args.lr, args.regularize_l1), 'wb') as f:
            pickle.dump(results, f)


