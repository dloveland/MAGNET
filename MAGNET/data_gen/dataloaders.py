from random import shuffle
import torch
from torch import tensor
from torch.utils.data import random_split
import torch_geometric as pyg
from torch_geometric.data import Data, Dataset
from pathlib import Path
import h5py
import numpy as np
import networkx as nx
from ast import literal_eval
# from ..util import binarize
import sys
if sys.flags.debug:
    import pdb
from torch_geometric.utils import to_networkx
import matplotlib.pyplot as plt
import torch_geometric.transforms as T
from torch_geometric.transforms import line_graph
from torch_geometric.utils import coalesce
np.set_printoptions(threshold=sys.maxsize)
import tqdm
import time
torch.set_printoptions(threshold=1000, edgeitems=4)

class WTA(Dataset):

    def __init__(self, 
        filename, 
        transform=None, 
        pre_transform=None, 
        pre_filter=None,
        data_subset=None,
        repr='weighted_bipartite',
        data_amnt=1000,
        incumbent=False,
        **kwargs
    ):
        self.filename = Path(filename)
        root = str(self.filename.parent)

        super().__init__(root, transform, pre_transform, pre_filter)

        self._file = h5py.File(str(self.filename), 'r')
        print(self._file)
        self.keylist = list(self._file['keys'])
        self.incumbent = incumbent 
        self.size = len(self.keylist) 
        if self.size > data_amnt:
            self.size = data_amnt

        self._next_idx = 0

        self.repr = repr
        # Allows you to limit the dataset to a contiguous subset
        self.data_subset = data_subset

    def len(self):
        return self.size


    def get(self, idx=0, device='cpu', flatten_y=True):

        data = []

        for idx in range(self.size): 
            start_time = time.time()

            k = self.keylist[idx]

            if self.incumbent: 
                mosek_lowerbound = np.array(self._file["mosek"]['lower_bound'][k])
                print(mosek_lowerbound)
            if self.incumbent: 
                label = np.round(np.array(self._file["labels"]["incumbent_point"][k]))
            else: 
                label = np.round(np.array(self._file["labels"]["optimal_point"][k]))
            
            if self.incumbent: 
                optimal_val = np.array(self._file["labels"]["incumbent_value"][k])
            else: 
                optimal_val = np.array(self._file["labels"]["optimal_value"][k])

            v = np.array(self._file['data']['V'][k])
            w = np.array(self._file['data']['W'][k])

            v_max = int(np.array(self._file['data']['max_V'][k]))
            w_max = int(np.array(self._file['data']['max_W'][k]))
            
            # will need to add check if this available, for now assume it is
            P = np.array(self._file['data']['P_success'][k])
            prob_matrix = P
            use_P = True
            # Load in C, but may be all 1s 
            C  = np.array(self._file['data']['C'][k])
            cost_matrix = C
            # if everything is 1s, we dont use C 
            if np.all(C == 1):
                use_C = False
            else:
                use_C = True

            v_len = int(np.array(self._file['data']['V_len'][k]))
            w_len = int(np.array(self._file['data']['W_len'][k]))
            orig_w = w_len 
            
            # turn into single-target from multi-target
            # only expand when probabilistic 
            if use_P:
                # Compute the duplication amnt here, with floor(capacity/min(cost))
                min_cost = np.min(C, axis=1)
                soft_alloc = w/min_cost
                w_alloc = np.floor(soft_alloc).astype(int)
            
                w_len = int(np.sum(w_alloc))
         
                # ducplicate capacities on W
                w_index = []
                for i in range(len(w)):
                    num_dup = w_alloc[i]
                    for n in range(num_dup):
                        w_index.append(i)
                w = np.concatenate(([np.repeat([w[i]], w_alloc[i], axis=0) for i in range(len(w))]))

                # duplicate rows of P to accomodate single target W 
                P = np.concatenate(([np.repeat([P[i, :]], w_alloc[i], axis=0) for i in range(P.shape[0])]))

                # duplicate rows of C to accomdate single target W
                C = np.concatenate(([np.repeat([C[i, :]], w_alloc[i], axis=0) for i in range(C.shape[0])]))

                # Now have expanded W, expanded P, and same V, now must make Y match 
                Y_expanded = np.zeros(P.shape)
                
                # go through weapons; there is a chance w_alloc for a node is greater than the number of ground truth assignments
                counter = 0 
                for weapon in range(label.shape[0]): 
                    alloc, targets = np.unique(label[weapon, :], return_index=True)
                    alloc = alloc[1:]
                    targets = targets[1:]
                    duplicated_w = w_alloc[weapon]

                    for t, target in enumerate(targets):
                        Y_expanded[counter:counter+int(alloc[t]), target] = 1
                        counter += int(alloc[t])
                        duplicated_w -= int(alloc[t])
                    
                    # at the end, duplicated_w will hold whatever didnt get allocated, skip over them through counter
                    counter += duplicated_w 
 
                orig_label = label 
                label = Y_expanded

            # after pre-processing, start building tensors 
            node_features = np.concatenate((v, w)).reshape(-1, 1)
            
            # will take transpose here to accomodate V x W matrix
            label = label.T
            if flatten_y:
                label = label.flatten()

            edge_index = []
            edge_prob = []
            edge_cost = []
            for vi in range(v.shape[0]):
                for wi in range(w.shape[0]):
                    # Offset by number of V in node idx list to reach W
                    edge_index.append([vi, v.shape[0] + wi])
                    edge_index.append([v.shape[0] + wi, vi])

                    # duplicate for both directions, same weight
                    edge_prob.append(P[wi, vi])
                    edge_prob.append(P[wi, vi])

                    # set up costs
                    edge_cost.append(C[wi, vi])
                    edge_cost.append(C[wi, vi])

            edge_index = np.array(edge_index).T
            edge_prob = np.array(edge_prob).reshape(-1, 1)
            edge_cost = np.array(edge_cost).reshape(-1, 1)

            if use_P and use_C:
                edge_attr = [torch.Tensor(edge_prob).to(device), torch.Tensor(edge_cost).to(device)]
            else:
                if use_P:
                    edge_attr = torch.Tensor(edge_prob).to(device)
                elif use_C:
                    edge_attr = torch.Tensor(edge_cost).to(device)
      
            data_p = Data(
                x=tensor(node_features).to(device).float(),
                edge_index=tensor(edge_index).to(device),
                edge_attr=edge_attr,
                y=tensor(label).to(device),
                v_len=v_len,
                w_len=w_len,
                orig_w=orig_w,
                orig_label=torch.Tensor(orig_label),
                v=tensor(v).to(device),
                w=tensor(w).to(device),
                w_index=torch.Tensor(w_index), 
                prob_matrix=torch.Tensor(prob_matrix),
                cost_matrix=torch.Tensor(cost_matrix), 
                use_P=use_P,
                use_C=use_C,
                optimal_val=optimal_val
                )
  
            if self.incumbent:
                data_p.mosek_lower = mosek_lowerbound

            # if line graph, update the data object directly 
            if self.repr == 'line_graph': 

                # In original data point 
                # x is node features of size (v + w) * 1
                # edge_index will be v * w, edge_attr same (*2 for undirected)
                # y will be of size v*w 

                # when we have prob and cost, we need to combine together so that transform can move to node feats
                if use_P and use_C:
                    data_p.edge_attr = torch.cat(data_p.edge_attr, axis=1)
                # Need to retain the edge associated with each node in new graph -- can get through coalescing 
                original_edge_map = coalesce(data_p.edge_index)[:, :int(data_p.edge_index.shape[1]/2)]
            
                # Perform linegraph transformation 
                transform = T.Compose([T.LineGraph()])

                # Line graph transform throws away old features, need to use those to create edge weights 
                node_features = data_p.x 
        
                # do the transformation to line graph, edge attr autamatically become node feat
                data_p = transform(data_p)

                # for each element in new edge_index, we need to figure out which node it originally was 
                # In new data point
                # x is of size (v * w) + 1, for each edge we get the edge weight 
                # edge_index will be sum of degrees choose 2 for each node 

                # need to populate edge weights by going through each edge
                start_nodes = data_p.edge_index[0, :]
                end_nodes = data_p.edge_index[1, :]


                orig_start = original_edge_map[:, start_nodes]
                orig_end = original_edge_map[:, end_nodes]
                similar_idx_mask = original_edge_map[:, start_nodes] == original_edge_map[:, end_nodes]

                shared_idx = torch.sum(orig_start * similar_idx_mask, axis=0)
                data_p.edge_attr = node_features[shared_idx]
                
            time_diff = time.time() - start_time
            data_p.time = time_diff
            data.append(data_p)
            
        return data


    def plot(self, data, nogui=True):
        """
        Uses NetworkX to plot the graph

        Somewhat useful for small graphs; less so for large ones.
        """

        if self.repr == 'weighted_bipartite':
            
            networkx_g = to_networkx(data)
            nx.draw_networkx(networkx_g, pos = nx.drawing.layout.bipartite_layout(networkx_g, range(data.v_len)))
            plt.savefig('bipartite.png')


    def __iter__(self):
        if self.data_subset is None:
            return (self.get(ii) for ii in range(self.size))
        else:
            return (self.get(ii) for ii in self.data_subset)

    def __del__(self):
        # Close the HDF5 file
        # Seems as though the HDF5 is already closed by time del is called 
        try: 
            self._file.close()
        except:
            pass





if __name__ == "__main__":

    wta_dataset = WTA('generate/assignment_data2023-10-03_19-51-20.hdf5', repr='line_graph')
    data = wta_dataset.get()

    