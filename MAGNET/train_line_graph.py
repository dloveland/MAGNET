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
                self.model.conv.register_forward_hook(self.intermed_hook)
            else:
                self.model.act.register_forward_hook(self.intermed_hook)

        self.optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(args.pos_weight))

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
                
                hid_rep = self.model(data.x, data.edge_index, edge_attr=data.edge_attr)
               
                # this case, we are learning directly on nodes (edges in original graph) 
                hid_rep = hid_rep.flatten().reshape(data.v_len, data.w_len)
                data_y = data.y.reshape(data.v_len, data.w_len)
                
                idx = self.get_dead_indices(hid_rep, data_y)
                loss = self.criterion(hid_rep[:, idx], data_y.float()[:, idx])

                # Compute the L1 penalty over the act outputs captured by the hook.
                if self.args.regularize_l1 > 0.0:
                    for output in self.intermed_hook:
                        loss += (torch.norm(output, 1) * args.regularize_l1)

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

    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--regularize_l1', type=float, default=0.0)

    args = parser.parse_args()

    args.repr = 'line_graph'
    random.seed(args.seed)

    # set up naming
    dataset_name = args.path.split('/')[-1].rstrip('.hdf5')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device 


    if args.regularize_l1 > 0.0:
        print(args.regularize_l1)
        args.intermed_return = True 

    save_folder = 'gnn_pyg/models_results/{0}/{1}_{2}_{3}_{4}/'.format(dataset_name, 'wta', args.model, 'line_graph', args.seed)
    
    os.makedirs(save_folder, exist_ok=True)
   
    # Load in dataset 
    dataset = WTA(args.path, repr=args.repr)

    # Get with no idx gives all graphs 
    data = dataset.get(device=device)

    # if probabilistic, use expansions
    args.use_P = data[0].use_P
    args.use_C = data[0].use_C    
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
    print(model)

    # get the loss weight based on the number of targets
    neg = (data[0].y == 0.0).sum()
    pos = (data[0].y).sum()

    pos_weight = neg/pos
    args.pos_weight = pos_weight

    trainer = Trainer(model, train_loader, val_loader, args)
    
    train_losses = trainer.train_wta() 

    # Save the models 
    save_models = {'model': trainer.model.state_dict()}
   
    torch.save(save_models, save_folder + 'nhid_{0}_depth_{1}_nout_{2}_lr_{3}_reg_{4}.pth'.format(args.nhid, args.depth, args.nout, args.lr, args.regularize_l1))

    results = {
            'train_losses': train_losses,
        }

    if args.regularize_l1 > 0.0:
        with open(save_folder + 'nhid_{0}_depth_{1}_nout_{2}_lr_{3}_reg_{4}.pkl'.format(args.nhid, args.depth, args.nout, args.lr, args.regularize_l1), 'wb') as f:
            pickle.dump(results, f)


