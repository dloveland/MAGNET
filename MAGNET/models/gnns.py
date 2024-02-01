import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.nn import Parameter, Linear
from torch_geometric.nn import GATConv, GCNConv, MessagePassing, APPNP, GATConv, GCN2Conv, FAConv
from copy import deepcopy
from torch_geometric.nn.conv.sage_conv import SAGEConv
import types
from torch_sparse import SparseTensor, matmul
import scipy
from torch_geometric.utils import subgraph 
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import spmm


############ GLAN ##############

class GLAN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, depth=2, dropout=0.3, act='relu', args=None):
        super(GLAN, self).__init__()
        
        # edge convolution
        in_dim_edge = (int(args.use_C) + int(args.use_P))

        self.convs = nn.ModuleList()
        for d in range(depth):
            if d == 0:
                self.convs.append(
                    GLAN_Conv(nfeat, in_dim_edge, nhid, args))
            elif d == (depth - 1):
                self.convs.append(
                    GLAN_Conv(nhid, nhid, nclass, args))
            else: 
                self.convs.append(
                    GLAN_Conv(nhid, nhid, nhid, args))

        self.dropout = dropout
        self.act = torch.nn.ReLU()


    def forward(self, x, edge_index, edge_attr=None):

        for c, conv in enumerate(self.convs):
            x, edge_attr = conv(x, edge_index, edge_attr) 
        # the edge_attr is our predictions 
        return x, edge_attr

class GLAN_Conv(MessagePassing):
    def __init__(self, in_dim_node, in_dim_edge, out_dim, args=None):
        super().__init__(aggr='mean')
        # all MLPS will have depth of 1


        ## ASSUME both Kappas should hve same dim as original in_dim 
        # edge convolution
        # in is 3*in_dim since we concat max, min, mean
        self.kappa_v = GLAN_MLP(3*in_dim_node, in_dim_node)

        # node convolution
        # in is 3*in_dim since we concat max, min, mean
        self.kappa_e = GLAN_MLP(3*in_dim_edge, in_dim_edge)

        # edge update
        self.pe = GLAN_MLP(2*in_dim_node + in_dim_edge, out_dim)

        # node update
        ## ASSUME tau is outputting a scalar weight 
        self.tau = GLAN_MLP(2*in_dim_node, 1)
        # in is 2*out_dim since we concat edges and nodes 
        self.pv_1 = GLAN_MLP(in_dim_node + in_dim_edge, out_dim)
        # in is in_dim + out_dim as we concat orig feat with aggr feat after embed 
        self.pv_2 = GLAN_MLP(in_dim_node +out_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):

        if isinstance(edge_attr, list):
            edge_attr = torch.cat(edge_attr, axis=1)

        ##### edge convolution #####
        ## get the attention weights cv and ce ##
        # cv = kv(max(v), min(v), mean(v)) 
        concat_x = torch.cat((x.max(dim=0).values.reshape(1, -1), x.min(dim=0).values.reshape(1, -1), x.mean(dim=0).reshape(1, -1)), dim=1)
        cv = self.kappa_v(concat_x)
        # ce = ke(max(e), min(e), mean(e))
        concat_edge = torch.cat((edge_attr.max(dim=0).values.reshape(1, -1), \
                                     edge_attr.min(dim=0).values.reshape(1, -1), \
                                     edge_attr.mean(dim=0).reshape(1, -1)), dim=1)
  
        ce = self.kappa_e(concat_edge)
        ## Update edge attributes ##
        # e = pe(e_bar); e_bar = (vi dot cv, vj dot cv, e dot c)
        v_i = x[edge_index[0, :]]
        v_j = x[edge_index[1, :]]

        ##### node convolution #####
        ## Get weights for neighboring nodes
        # w = tau(v_i, v_j)
        w = self.tau(torch.cat((v_i, v_j), dim=1))

        ## node aggregation (going to assume p doesnt depend on v)
        # v = sum(pv(e dot ce, w(vj dot cv)))
        # first transform edges (e dot ce)
        e_trans = edge_attr * ce
        # second transform neighbor nodes (vj dot cv)
        x_trans = x * cv
        # propagate messages and aggregate
        v_aggr = self.propagate(edge_index, x=x_trans, w=w, e=e_trans)
        x = self.pv_2(torch.cat((x, v_aggr), dim=1))

        ## final edge update
        ## ASSUME, we cant use new vi and vj here since they have changed dims and no longer work with cv and ce 
        edge_attr = self.pe(torch.cat((v_i * cv, v_j * cv, edge_attr * ce), dim=1))
        return x, edge_attr

    def message(self, x_i, x_j, w, e):
        # message is pv_1(e dot ce, w(vj dot cv))
        return self.pv_1(torch.cat((e, (w * x_j)), dim=1))

class GLAN_MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GLAN_MLP, self).__init__()
        
        self.lins = Linear(in_dim, out_dim)

    def forward(self, x):
        
        x = self.lins(x)
        x = torch.relu(x)

        return x

############ MLP ##############
class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, depth=2, dropout=0.3, act='relu'):
        super(MLP, self).__init__()
        
        self.lins = nn.ModuleList()
        
        self.lins.append(Linear(nfeat, nhid))

        for _ in range(depth - 2):
            self.lins.append(
                Linear(nhid, nhid))

        self.lins.append(Linear(nhid, nclass))
        self.dropout = dropout
        self.act = act

    def reset_parameters(self):
        for lins in self.lins:
            lins.reset_parameters()

    def forward(self, x):
        for lins in self.lins[:-1]:
            x = lins(x)
            if self.act == 'relu':
                x = torch.relu(x)
            elif self.act == 'sigmoid':
                x = torch.sigmoid(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x



############ GCN ##############
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, depth=2, dropout=0.3, act='relu', edge_update=False, number_conv_heads=1, args=None):
        super(GCN, self).__init__()
        
        self.args = args
        self.depth = depth
        self.repr = args.repr

        if self.repr == 'weighted_bipartite':
            in_mult = int(self.args.use_C) + int(self.args.use_P)

            if self.args.use_C:
                self.convs_cost = nn.ModuleList()
                self.convs_cost.append(GCNConv(nfeat, nhid))
            
                for _ in range(depth - 2):
                    self.convs_cost.append(
                        GCNConv(in_mult*nhid, nhid))

                self.convs_cost.append(GCNConv(in_mult*nhid, nout))

            if self.args.use_P:
                self.convs_prob = nn.ModuleList()
                self.convs_prob.append(GCNConv(nfeat, nhid))
            
                for _ in range(depth - 2):
                    self.convs_prob.append(
                        GCNConv(in_mult*nhid, nhid))

                self.convs_prob.append(GCNConv(in_mult*nhid, nout))

            # If use both, at end we will have final linear layer
            if self.args.use_C and self.args.use_P:
                self.linear = Linear(in_mult*nout, nout)
        # if line graph, dont need seperate heads 
        else: 
            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(nfeat, nhid))
        
            for _ in range(depth - 2):
                self.convs.append(
                    GCNConv(nhid, nhid))

            self.convs.append(GCNConv(nhid, nout))

        self.dropout = dropout
        self.act = torch.nn.ReLU()

    def reset_parameters(self):
        if self.repr == 'weighted_bipartite':
            for conv in self.convs_cost:
                conv.reset_parameters()
            for conv in self.convs_prob:
                conv.reset_parameters()
        else:
            for conv in self.convs:
                conv.reset_parameters() 

    def forward(self, x, edge_index, edge_attr=None, prune=None, v_len=None, w_len=None, prune_factor=0.25):

        # assume edge attr is ordered prob and then cost, if both 
        if self.repr == 'weighted_bipartite':
            if isinstance(edge_attr, list):
                probs = edge_attr[0]
                cost = edge_attr[1]
            else: 
                # if edge_attr is not a list, its a tensor and can only be prob OR cost
                if self.args.use_C:
                    cost = edge_attr
                elif self.args.use_P:
                    probs = edge_attr
                    
            for l in range(self.depth):
                if self.args.use_C:
                    x_cost = self.convs_cost[l](x, edge_index, edge_weight=cost)
                    if l != self.depth:
                        x_cost = self.act(x_cost)
                        x_cost = F.dropout(x_cost, p=self.dropout, training=self.training)
                if self.args.use_P:
                    x_prob = self.convs_prob[l](x, edge_index, edge_weight=probs)
                    if l != self.depth: 
                        x_prob = self.act(x_prob)
                        x_prob = F.dropout(x_prob, p=self.dropout, training=self.training)
                if self.args.use_P and self.args.use_C:
                    x = torch.cat([x_cost, x_prob], dim=1)
                else:
                    if self.args.use_C:
                        x = x_cost
                    elif self.args.use_P:
                        x = x_prob

            if self.args.use_P and self.args.use_C:  
                x = self.linear(x)
        else: 
            # node mask (edges in original graph)
            if prune:
                top_idx = torch.ones((x.shape[0])).long()

            for l in range(self.depth):
                x = self.convs[l](x, edge_index, edge_weight=edge_attr)
                if l != self.depth:
                    x = self.act(x)
                    x = F.dropout(x, p=self.dropout, training=self.training)

                if prune: 
                    # make sure zerod out before norm in case activation does something weird 
                    x[(top_idx == 0), :] = 0

                    # norm the values 
                    norm_x = torch.norm(x, dim=1) 

                    # reshape norm_x so we can drop from each wep allocation 
                    norm_x = norm_x.reshape(v_len, w_len)

                    # argsort norm to get biggest and smallest 
                    idx = torch.argsort(norm_x, axis=0, descending=True)

                    # drop 1/3 every step 
                    # get the top indices scores per wep, the nodes we are keeping
                    drop_from = math.ceil(v_len * (prune_factor/(l + 1)))
                    print(drop_from)
                    drop_idx = idx[drop_from:, :]

                    # transforms top_idx back into flat vector idx
                    flat_idx = ((drop_idx * w_len) + torch.arange(0, w_len.item())).flatten()
                    top_idx[flat_idx] = 0

                    # subgraph function requires node indices
                    subgraph_idx = torch.where(top_idx == 1)[0]

                    # remove edges that arent attached to top_idx nodes 
                    edge_index, edge_attr = subgraph(subgraph_idx, edge_index, edge_attr=edge_attr, num_nodes=x.shape[0])

                    # once we have figured out what to remove, zero out the node features 
                    #x[(top_idx == 0), :] = 0
        
        if prune:
            x[(top_idx == 0), :] = 0
        return x


############### GCN-II ###############

class GCNII(torch.nn.Module):
    def __init__(self, nfeat, nhid, nout, depth=2, alpha=0.1, theta=0.5,
                 shared_weights=True, dropout=0.3, act='relu', args=None):
        super(GCNII, self).__init__()

        self.args = args
        self.depth = depth
        self.repr = args.repr

        # check repr
        if self.repr == 'weighted_bipartite':
            
            in_mult = int(self.args.use_C) + int(self.args.use_P)

            # init and final lins to embed nodes 
            self.lins = torch.nn.ModuleList()
            self.lins.append(Linear(nfeat, nhid))
            # multiplier since we will concat prob and cost if available 
            self.lins.append(Linear(in_mult*nhid, nout))

            if self.args.use_C:
                self.convs_cost = nn.ModuleList()            
                for layer in range(depth):
                    self.convs_cost.append(
                        GCN2Conv(nhid, alpha, theta, layer + 1,
                                shared_weights, normalize=True))

            if self.args.use_P:
                self.convs_prob = nn.ModuleList()
                for layer in range(depth):
                    self.convs_prob.append(
                        GCN2Conv(nhid, alpha, theta, layer + 1,
                                shared_weights, normalize=True))

            # If use both, between each layer we have to concat and re-embed to get back to nhid
            if self.args.use_C and self.args.use_P:
                self.lins_prob_cost = torch.nn.ModuleList() 
                for layer in range(depth - 1):
                    self.lins_prob_cost.append(Linear(in_mult*nhid, nhid))
            
        else:
            self.lins = torch.nn.ModuleList()
            self.lins.append(Linear(nfeat, nhid))
            self.lins.append(Linear(nhid, nout))

            self.convs = torch.nn.ModuleList()
            for layer in range(depth):
                self.convs.append(
                    GCN2Conv(nhid, alpha, theta, layer + 1,
                            shared_weights, normalize=True))

        self.dropout = dropout
        self.act = torch.nn.ReLU()
        
    def reset_parameters(self):
        if self.repr == 'weighted_bipartite':
            for conv in self.convs_cost:
                conv.reset_parameters()
            for conv in self.convs_prob:
                conv.reset_parameters()
        else:
            for conv in self.convs:
                conv.reset_parameters()


    def forward(self, x, edge_index, edge_attr=None, prune=False, v_len=None, w_len=None, prune_factor=0.25):

        # always do initial embedding to get node features to nhid 
        x = x_0 = self.lins[0](x)
        x = self.act(x)
        x = F.dropout(x, self.dropout, training=self.training)

        if self.repr == 'weighted_bipartite':
            if isinstance(edge_attr, list):
                probs = edge_attr[0]
                cost = edge_attr[1]
            else: 
                # if edge_attr is not a list, its a tensor and can only be prob OR cost
                if self.args.use_C:
                    cost = edge_attr
                elif self.args.use_P:
                    probs = edge_attr

            for l in range(self.depth):
                if self.args.use_C:
                    x_cost = self.convs_cost[l](x, x_0, edge_index, edge_weight=cost)
                    x_cost = self.act(x_cost)
                    x_cost = F.dropout(x_cost, p=self.dropout, training=self.training)
                if self.args.use_P:
                    x_prob = self.convs_prob[l](x, x_0, edge_index, edge_weight=probs)
                    x_prob = self.act(x_prob)
                    x_prob = F.dropout(x_prob, p=self.dropout, training=self.training)
                if self.args.use_P and self.args.use_C:
                    x = torch.cat([x_cost, x_prob], dim=1)
                    # if we arent at last layer, need to re-embed to get back to nhid dims 
                    if l < self.depth-1: 
                        x = self.lins_prob_cost[l](x)
                else:
                    if self.args.use_C:
                        x = x_cost
                    elif self.args.use_P:
                        x = x_prob

        # line graph, just do directly 
        else: 
            if prune:
                top_idx = torch.ones((x.shape[0])).long()

            for l in range(self.depth):
                x = self.convs[l](x, x_0, edge_index, edge_weight=edge_attr)
                x = self.act(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

                if prune: 
                    x[(top_idx == 0), :] = 0

                    # norm the values 
                    norm_x = torch.norm(x, dim=1) 

                    # reshape norm_x so we can drop from each wep allocation 
                    norm_x = norm_x.reshape(v_len, w_len)

                    # argsort norm to get biggest and smallest 
                    idx = torch.argsort(norm_x, axis=0, descending=True)

                    # drop 1/3 every step 
                    # get the top indices scores per wep, the nodes we are keeping
                    drop_from = math.ceil(v_len * (prune_factor/(l + 1)))
                    drop_idx = idx[drop_from:, :]

                    # transforms top_idx back into flat vector idx
                    flat_idx = ((drop_idx * w_len) + torch.arange(0, w_len.item())).flatten()
                    top_idx[flat_idx] = 0

                    # subgraph function requires node indices
                    subgraph_idx = torch.where(top_idx == 1)[0]

                    # remove edges that arent attached to top_idx nodes 
                    edge_index, edge_attr = subgraph(subgraph_idx, edge_index, edge_attr=edge_attr, num_nodes=x.shape[0])
        
                    # zero out before next pass so they are dead 
                    #x[(top_idx == 0), :] = 0

        x = self.lins[1](x)

        # final zero out of whatever we have left
        if prune:
            x[(top_idx == 0)] = 0

        return x 


###### GPRGNN #######
class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha
        self.Gamma = Gamma

        assert Init in ['PPR', 'NPPR', 'Random', 'WS']
        if Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        if self.Init == 'PPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha*(1-self.alpha)**k
            self.temp.data[-1] = (1-self.alpha)**self.K
        elif self.Init == 'NPPR':
            for k in range(self.K+1):
                self.temp.data[k] = self.alpha**k
            self.temp.data = self.temp.data/torch.sum(torch.abs(self.temp.data))
        elif self.Init == 'WS':
            self.temp.data = self.Gamma

    def forward(self, x, hidden, edge_index, norm, k):
    
        x = self.propagate(edge_index, x=x, norm=norm)
        gamma = self.temp[k+1]
        hidden = hidden + gamma*x

        return x, hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class GPRGNN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, depth=2, dropout=0.3, K=2, alpha=0.1, init='PPR', act='relu', args=None):
        super(GPRGNN, self).__init__()

        self.repr = args.repr
        self.args = args

        self.K = K 
        self.lin_start = Linear(nfeat, nhid)
        self.lin_mid = nn.ModuleList()
        for i in range(depth - 2):
            self.lin_mid.append(Linear(nhid, nhid))
        self.lin_end = Linear(nhid, nclass)

        if self.repr == 'weighted_bipartite':
            in_mult = int(self.args.use_C) + int(self.args.use_P)

            # create both prop heads 
            if self.args.use_C:
                self.conv_cost = GPR_prop(K, alpha, init)

            if self.args.use_P:
                self.conv_prob = GPR_prop(K, alpha, init)

            # If use both, will need to combine together at end 
            if self.args.use_C and self.args.use_P:
                self.lins_prob_cost_x = Linear(in_mult*nclass, nclass)
                self.lins_prob_cost_hid = Linear(in_mult*nclass, nclass)

        # if line graph, dont need seperate heads 
        else: 
            self.conv = GPR_prop(K, alpha, init)
        
        self.dropout = dropout
        self.act = torch.nn.ReLU()

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, prune=False, v_len=None, w_len=None, prune_factor=0.25):

        # initial processing of features 
        x = F.relu(self.lin_start(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        for lin in self.lin_mid:
            x = F.relu(lin(x))
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.lin_end(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_cost = x_prob = x 
        # start propogation 
        if self.repr == 'weighted_bipartite':
            if isinstance(edge_attr, list):
                probs = edge_attr[0]
                cost = edge_attr[1]
            else: 
                # if edge_attr is not a list, its a tensor and can only be prob OR cost
                if self.args.use_C:
                    cost = edge_attr
                elif self.args.use_P:
                    probs = edge_attr

            if self.args.use_C:
                edge_index_cost, norm_cost = gcn_norm(
                    edge_index, cost, num_nodes=x.size(0), dtype=x.dtype)
                hidden_cost = x_cost*(self.conv_cost.temp[0])
            if self.args.use_P:
                edge_index_prob, norm_prob = gcn_norm(
                    edge_index, probs, num_nodes=x.size(0), dtype=x.dtype)
                hidden_prob = x_prob*(self.conv_prob.temp[0])

            for k in range(self.K): 
                if self.args.use_C:
                    x_cost, hidden_cost = self.conv_cost(x_cost, hidden_cost, edge_index_cost, norm_cost, k)
                if self.args.use_P: 
                    x_prob, hidden_prob = self.conv_prob(x_prob, hidden_prob, edge_index_prob, norm_prob, k)
            
            if self.args.use_P and self.args.use_C:
                x = torch.cat([x_cost, x_prob], dim=1)
                hidden = torch.cat([hidden_cost, hidden_prob], dim=1)

                x = self.lins_prob_cost_x(x)
                hidden = self.lins_prob_cost_hid(hidden)
            else:
                if self.args.use_C:
                    x = x_cost
                    hidden = hidden_cost
                elif self.args.use_P:
                    x = x_prob
                    hidden = hidden_prob 
        else: 
            if prune:
                top_idx = torch.ones((x.shape[0])).long()

            edge_index, norm = gcn_norm(
                edge_index, edge_attr, num_nodes=x.size(0), dtype=x.dtype)

            hidden = x*(self.conv.temp[0])

            for k in range(self.K): 
                x, hidden = self.conv(x, hidden, edge_index, norm, k)
                if prune: 
                    x[(top_idx == 0), :] = 0
                    hidden[(top_idx == 0), :] = 0

                    # norm the values 
                    norm_x = torch.norm(x, dim=1) 

                    # reshape norm_x so we can drop from each wep allocation 
                    norm_x = norm_x.reshape(v_len, w_len)

                    # argsort norm to get biggest and smallest 
                    idx = torch.argsort(norm_x, axis=0, descending=True)

                    # drop 1/3 every step 
                    # get the top indices scores per wep, the nodes we are keeping
                    drop_from = math.ceil(v_len * (prune_factor/(k + 1)))
                    drop_idx = idx[drop_from:, :]

                    # transforms top_idx back into flat vector idx
                    flat_idx = ((drop_idx * w_len) + torch.arange(0, w_len.item())).flatten()
                    top_idx[flat_idx] = 0

                    # subgraph function requires node indices
                    subgraph_idx = torch.where(top_idx == 1)[0]

                    # remove edges that arent attached to top_idx nodes 
                    edge_index, norm = subgraph(subgraph_idx, edge_index, edge_attr=norm, num_nodes=x.shape[0])
    
                    #edge_index, norm = gcn_norm(
                    #    edge_index, norm, num_nodes=x.size(0), dtype=x.dtype)

                    #x[(top_idx == 0), :] = 0
                    #hidden[(top_idx == 0), :] = 0
        if prune:     
            hidden[(top_idx == 0), :] = 0

        return hidden

###### FAGCN #########
class FAGCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, depth=2, dropout=0.3, eps=0.1, act='relu', args=None):
        super(FAGCN, self).__init__()

        self.repr = args.repr
        self.args = args

        self.depth = depth

        self.lin_start = Linear(nfeat, nhid)
        self.dropout = dropout

        if self.repr == 'weighted_bipartite':
            in_mult = int(self.args.use_C) + int(self.args.use_P)

            # create both prop heads 
            if self.args.use_C:
                self.conv_cost = nn.ModuleList() 
                for i in range(depth): 
                    self.conv_cost.append(FAConv(nhid, eps, dropout, normalize=False))
            if self.args.use_P:
                self.conv_prob = nn.ModuleList() 
                for i in range(depth): 
                    self.conv_prob.append(FAConv(nhid, eps, dropout, normalize=False))
            # final linear to take all conv and put into single output 
            self.lin_end = Linear(in_mult*nhid, nclass)
        else: 
            self.layers = nn.ModuleList()
            for i in range(depth):
                # normalize with weights rather than degree
                self.layers.append(FAConv(nhid, eps, dropout, normalize=False))
            self.lin_end = Linear(nhid, nclass)
        self.reset_parameters()
        self.act = torch.nn.ReLU()
        
    def reset_parameters(self):
        nn.init.xavier_normal_(self.lin_start.weight, gain=1.414)
        nn.init.xavier_normal_(self.lin_end.weight, gain=1.414)

    def forward(self, x, edge_index, edge_attr=None, prune=False, v_len=None, w_len=None, prune_factor=0.25):

        x = x_0 = F.relu(self.lin_start(x))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_cost = x_prob = x

        if self.repr == 'weighted_bipartite':
            if isinstance(edge_attr, list):
                probs = edge_attr[0]
                cost = edge_attr[1]
            else: 
                # if edge_attr is not a list, its a tensor and can only be prob OR cost
                if self.args.use_C:
                    cost = edge_attr
                elif self.args.use_P:
                    probs = edge_attr

            for k in range(self.depth): 
                if self.args.use_C:
                    x_cost = self.conv_cost[k](x_cost, x_0, edge_index, edge_weight=cost.flatten())
                if self.args.use_P: 
                    x_prob = self.conv_prob[k](x_prob, x_0, edge_index, edge_weight=probs.flatten())
            
            if self.args.use_P and self.args.use_C:
                x = torch.cat([x_cost, x_prob], dim=1)

            else:
                if self.args.use_C:
                    x = x_cost
                elif self.args.use_P:
                    x = x_prob

        else: 
            if prune:
                top_idx = torch.ones((x.shape[0])).long()

            for l, layer in enumerate(self.layers):
                x = layer(x, x_0, edge_index, edge_weight=edge_attr.flatten())
    
                if prune: 
                    x[(top_idx == 0), :] = 0

                    # norm the values 
                    norm_x = torch.norm(x, dim=1) 

                    # reshape norm_x so we can drop from each wep allocation 
                    norm_x = norm_x.reshape(v_len, w_len)

                    # argsort norm to get biggest and smallest 
                    idx = torch.argsort(norm_x, axis=0, descending=True)

                    # drop 1/3 every step 
                    # get the top indices scores per wep, the nodes we are keeping
                    drop_from = math.ceil(v_len * (prune_factor/(l + 1)))
                    drop_idx = idx[drop_from:, :]

                    # transforms top_idx back into flat vector idx
                    flat_idx = ((drop_idx * w_len) + torch.arange(0, w_len.item())).flatten()
                    top_idx[flat_idx] = 0

                    # subgraph function requires node indices
                    subgraph_idx = torch.where(top_idx == 1)[0]

                    # remove edges that arent attached to top_idx nodes 
                    edge_index, edge_attr = subgraph(subgraph_idx, edge_index, edge_attr=edge_attr, num_nodes=x.shape[0])
        
                    x[(top_idx == 0), :] = 0

        x = self.lin_end(x)

        if prune:
            x[(top_idx == 0), :] = 0

        return x