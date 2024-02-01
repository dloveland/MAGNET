# MAGNET
Base implementation for MAGNET

## Data Generation
Data generation found in 'data_gen/gen_task.py' which will allow a user to generate and solve problems (using MOSEK) according Equation 1 in paper.

Parser for HDF5 file is available in 'data_gen/dataloaders.py', as well as a PyTorch Geometric dataloader. Module 1, the pre-processing step, is performed within the dataloader. 

## Training
Training can be done either through 'train_line_graph.py', or 'train_weighted_bipartite.py', depending on the representation the user wants to use. The line graph training faciliates Module 2 of MAGNET. 

GNN architectures, which define their own pruning methodology depending on the architecture, can be found in 'models/gnns.py'. 

## Testing
Testing can be done through 'test.py', which will load in the trained model and test it on the test set. Module 3, the post-processing step, is performed within the test file, generating feasible solutions given the parameters and predictions. 


