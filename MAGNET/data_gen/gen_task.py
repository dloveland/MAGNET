import numpy as np
import torch
from uuid import uuid4
import cvxpy as cp
from pathlib import Path
import h5py
import argparse
from ast import literal_eval
from datetime import datetime
import tqdm 
import sys
if sys.flags.debug:
    import pdb


class GeneralizedAssignmentGenerator:

    def __init__(self,
        num_tasks_range=[5, 6],
        num_agents_range=[5, 6],
        probability_range=[0.1, 0.9],
        task_value_range=[100, 500],
        cost_range=[0.1, 9.9],
        capacity_range=[0.1, 9.9],
        filename="assignment_data.hdf5",
        solver="mosek",
        verbose=False,
        rng=np.random.default_rng(),
        normalize_input=True,
        gen_WTA=False,
    ) -> None:

        # Assign all parameters to self
        self.num_tasks_range = num_tasks_range
        self.num_agents_range = num_agents_range
        self.probability_range = probability_range
        self.task_value_range = task_value_range
        self.cost_range = cost_range
        self.capacity_range = capacity_range
        self.filename = filename
        self.solver = solver
        self.verbose = verbose
        self.rng = rng
        self.normalize_input = normalize_input
        self.gen_WTA = gen_WTA

    def generate_dict(self,
        num_probs,
        rng=None,             
        optimizer_params={},
    ):
        """
        Generates generalized assignment problems.

        Explanation of optimization problem variables:

            * V: Value of each task
            * W: Agent capacity
            * C: Cost of each task
        """
        
        if rng is None:
            rng = self.rng

        output_dict = {}

        for ii in tqdm.tqdm(range(num_probs)):
            sys.stdout.flush()
            model_id = str(uuid4())

            # Number of tasks
            t_min, t_max = self.num_tasks_range
            num_tasks = rng.integers(t_min, t_max+1)

            # Number of agents
            a_min, a_max = self.num_agents_range
            num_agents = rng.integers(a_min, a_max+1)

            # Values of each task
            v_min, v_max = self.task_value_range
            V = (v_max - v_min) * rng.random(num_tasks) + v_min

            # Cost matrix
            if self.gen_WTA:
                # WTA problems assume all tasks have a cost of 1
                C = np.ones((num_agents, num_tasks))
            else:
                c_min, c_max = self.cost_range
                C = (c_max - c_min) * rng.random((num_agents, num_tasks)) + c_min

            # Capacity vector
            w_min, w_max = self.capacity_range
            if self.gen_WTA:
                W = rng.integers(w_min, w_max+1, num_agents)
            else:
                W = (w_max - w_min) * rng.random(num_agents) + w_min

            # Probability matrix
            p_min, p_max = self.probability_range
            P_success = (p_max - p_min) * rng.random((num_agents, num_tasks)) + p_min
            P_fail = 1 - P_success
            P_fail_log = np.log(P_fail)

            # # # Solve the optimization problem

            # Problem variable
            A = cp.Variable((num_agents, num_tasks), integer=True)

            # Objective function

            objective = cp.multiply(P_fail_log, A) 
            objective = cp.sum(objective, axis=0) # TODO: Check axis?
            objective = cp.exp(objective)
            objective = cp.multiply(V, objective)
            objective = cp.sum(objective)
            objective = cp.sum(V) - objective

            constraints = [cp.sum(cp.multiply(C,A), axis=1) <= W]
            pdb.set_trace() if sys.flags.debug else None

            for jj in range(A.shape[0]):
                for kk in range(A.shape[1]):
                    # A matrix is nonnegative
                    constraints.append(0 <= A[jj,kk])

            print("Solving...")
            sys.stdout.flush()
            problem = cp.Problem(cp.Maximize(objective), constraints)

            try:
                problem.solve(solver=cp.MOSEK, verbose=self.verbose, mosek_params=optimizer_params)
                status = problem.status
            except Exception as e:
                # Print exception
                print(repr(e))
                status = 'error'
            print('Solved')

            sys.stdout.flush()
            # Save the information to a dictionary
            if status not in ['infeasible', 'unbounded', 'infeasible_or_unbounded', 'error']:

                # Dividing by the maximum yields the same problem
                # Scaling using (X - min) / (max - min) produces different problems and constraints. Don't do this.

                max_V = np.max(V)
                max_C = np.max(C)
                max_W = np.max(W)

                # The scaling variable for C and U must be the same.
                max_CW = np.maximum(max_C, max_W)

                if self.normalize_input:
                    V = V / max_V
                    C = C / max_CW
                    W = W / max_CW

                output_dict[model_id] = {
                    'num_tasks': num_tasks,
                    'num_agents': num_agents,
                    'V': V,
                    'max_V': max_V,
                    'C': C,
                    'max_C': max_C,
                    'W': W,
                    'max_W': max_W,
                    'max_CW': max_CW,
                    'P_success': P_success,
                    'P_fail': P_fail,
                    'P_fail_log': P_fail_log,
                    'A': A.value,
                    'status': status,
                    'optimal_point': A.value,
                    'optimal_value': problem.value,
                }   

                print(f"Problem {ii+1} of {num_probs} solved successfully")

            else:
                print(f"Error: Problem {ii+1} of {num_probs} was {status}")

            
        pdb.set_trace() if sys.flags.debug else None
        return output_dict

    def generate_HDF5_graph(self,
        num_probs,
        rng=None,
        optimizer_params={},
    ):
        """
        Converts the results from generate_dict into a COO graph format and saves to an HDF5 file.

        NOTE: HDF5 is row-major.
        """

        if rng is None:
            rng = self.rng

        filename = Path(self.filename)
        prob_dict = self.generate_dict(num_probs, rng=rng, optimizer_params=optimizer_params)

        with h5py.File(str(filename.absolute()), "w") as file:
            pass

            file.create_group('data/num_tasks')
            file.create_group('data/num_agents')
            file.create_group('data/V')
            file.create_group('data/C')
            file.create_group('data/W')
            file.create_group('data/P_success')
            file.create_group('data/max_V')
            file.create_group('data/max_C')
            file.create_group('data/max_W')
            file.create_group('data/max_CW')
            file.create_group('data/V_len')
            file.create_group('data/C_shape')
            file.create_group('data/W_len')
            file.create_group('data/P_success_shape')
            file.create_group('data/P_success_size')

            file.create_group('labels/optimal_point')
            file.create_group('labels/optimal_value')

            file.create_group('graph/coo')
            file.create_group('graph/node_features')
            file.create_group('graph/edge_features')

            for key in prob_dict.keys():

                num_agents = prob_dict[key]['num_agents']
                num_tasks = prob_dict[key]['num_tasks']
                V = prob_dict[key]['V']
                C = prob_dict[key]['C']
                W = prob_dict[key]['W']
                P_success = prob_dict[key]['P_success']
                max_V = prob_dict[key]['max_V']
                max_C = prob_dict[key]['max_C']
                max_W = prob_dict[key]['max_W']
                max_CW = prob_dict[key]['max_CW']
                A = prob_dict[key]['A']
                optimal_point = prob_dict[key]['optimal_point']
                optimal_value = prob_dict[key]['optimal_value']

                # # Store values in HDF5 file
                file.create_dataset(f'data/num_tasks/{key}', data=num_tasks, dtype=np.int64)
                file.create_dataset(f'data/num_agents/{key}', data=num_agents, dtype=np.int64)
                file.create_dataset(f'data/V/{key}', data=V, dtype=np.float32)
                file.create_dataset(f'data/C/{key}', data=C, dtype=np.float32)
                file.create_dataset(f'data/W/{key}', data=W, dtype=np.float32)
                file.create_dataset(f'data/P_success/{key}', data=P_success, dtype=np.float32)
                file.create_dataset(f'data/max_V/{key}', data=max_V, dtype=np.float32)
                file.create_dataset(f'data/max_C/{key}', data=max_C, dtype=np.float32)
                file.create_dataset(f'data/max_W/{key}', data=max_W, dtype=np.float32)
                file.create_dataset(f'data/max_CW/{key}', data=max_CW, dtype=np.float32)
                file.create_dataset(f'data/V_len/{key}', data=len(V), dtype=np.int64)
                file.create_dataset(f'data/C_shape/{key}', data=np.array(C.shape), dtype=np.int64)
                file.create_dataset(f'data/W_len/{key}', data=len(W), dtype=np.int64)
                file.create_dataset(f'data/P_success_shape/{key}', data=np.array(P_success.shape), dtype=np.int64)
                file.create_dataset(f'data/P_success_size/{key}', data=P_success.size, dtype=np.int64)
                file.create_dataset(f'labels/optimal_point/{key}', data=optimal_point, dtype=np.float32)
                file.create_dataset(f'labels/optimal_value/{key}', data=optimal_value, dtype=np.float32)
                
            # Create the keys
            # NOTE: HDF5 only works with NumPy's 'S' dtype. This results in a
            #       'b' string: b'my string'. 
            #       This can be decoded back to UTF-8 as follows:
            #           b"my_string".decode("UTF-8")
            #       See the following for more details: https://docs.h5py.org/en/stable/strings.html
            key_list = np.array(list(prob_dict.keys()), dtype="S")

            file.create_dataset("keys", data=key_list, dtype=key_list.dtype)
            pdb.set_trace() if sys.flags.debug else None

            print(f"\n\nData saved in {str(filename.absolute())}")



if __name__ == "__main__":
    """
    Generate problem instances
    """

    parser = argparse.ArgumentParser("assignment")
    parser.add_argument("--path", type=str, default="./assignment_data", help="Full path for the data to be saved to.")
    parser.add_argument("--num_probs", type=int, default=100, help="Number of problems to be generated")
    parser.add_argument("--verbose", action="store_true", help="Sets the solver verbosity to True.")
    parser.add_argument("--solver", type=str, default="mosek", help="Set the MICP solver. Options are 'mosek'.")
    parser.add_argument("--maxtime", type=float, default=-1, help="Maximum solver runtime. Corresponds to `optimizerMaxTime` Mosek parameter.")
    parser.add_argument("--wta", action="store_true", help="Generate WTA problems instead of generalized assignment problems.")

    parser.add_argument('-t', "--num_tasks_range", type=str, default="[15, 15]", help="Range of number of tasks.")
    parser.add_argument('-a', "--num_agents_range", type=str, default="[15, 15]", help="Range of number of agents.")
    parser.add_argument('-p', "--probability_range", type=str, default="[0.1, 0.9]", help="Range of probabilities.")
    parser.add_argument('-v', "--task_value_range", type=str, default="[1.0, 10.0]", help="Range of task values.")
    parser.add_argument('-c', "--cost_range", type=str, default="[0.5, 1.0]", help="Range of task costs.")
    parser.add_argument('-u', "--capacity_range", type=str, default="[1.0, 3.0]", help="Range of agent capacities.")

    parser.add_argument("-n", "--no_normalize", action="store_false", help="Do not normalize the input data.")
    parser.add_argument("--datatype", type=str, default="default", help="Type of data to be generated. Options are 'default'")

    args = parser.parse_args()

    print("Generating data...")
    gen = GeneralizedAssignmentGenerator()
    gen.num_tasks_range = literal_eval(f"{args.num_tasks_range}")
    gen.num_agents_range = literal_eval(f"{args.num_agents_range}")
    gen.probability_range = literal_eval(f"{args.probability_range}")
    gen.task_value_range = literal_eval(f"{args.task_value_range}")
    gen.cost_range = literal_eval(f"{args.cost_range}")
    gen.capacity_range = literal_eval(f"{args.capacity_range}")
    gen.gen_WTA = args.wta

    data_name = datetime.now().strftime("data_%Y-%m-%d_%H-%M-%S.hdf5")

    save_path = Path(args.path)
    if not save_path.exists():
        save_path.mkdir(parents=True, exist_ok=True)

    mosek_params = {
        "MSK_DPAR_OPTIMIZER_MAX_TIME": args.maxtime,
    }

    gen.filename = Path(str(save_path.absolute()) + '/' + data_name)
    gen.solver = args.solver
    gen.verbose = args.verbose

    if args.no_normalize:
        gen.normalize_input = False
    else:
        gen.normalize_input = True

    if args.datatype.lower() == "default":
        gen.generate_HDF5_graph(args.num_probs, optimizer_params=mosek_params)

    args_str = ''
    for k, v in args.__dict__.items():
        args_str += (f"{k}: {v}\n")
    args_str = args_str.rstrip('\n')
    
    txt_file_path = data_name.rstrip('.hdf5')

    with open(str(save_path.absolute()) + '/' + txt_file_path + '.txt', 'w') as f:
        f.write(args_str)
