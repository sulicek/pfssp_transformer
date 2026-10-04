import docplex
from docplex.cp.model import *
from docplex.cp.parameters import CpoParameters

import torch

from tqdm import tqdm

from tools import build_warm_start_pfs

class CplexFS:
    def __init__(self, time_limit = 60, seed=0, log=False):
        self.time_limit = time_limit
        self.seed = seed
        self.parameters = CpoParameters(RandomSeed=self.seed)
        # Turn off automatic logs
        if not log:
            docplex.cp.config.context.log_output = None
        self.progress_listener = None
        self.factor = 100000
        self.solution_callback = None

    def run_cplex_pfs_batched(self, x, bo, warm_start_sequence = [], leave=True):
        bs = x.shape[0]
        tours = -torch.ones((bs, x.shape[1]), device=x.device, dtype=torch.long)
        times = -torch.ones((bs), device=x.device, dtype=torch.float64)
        for b in tqdm(range(bs), leave=leave): # no option other than lazy batching
            tour, time = self.run_cplex_pfs(x[b], bo[b] if bo is not None else None, warm_start_sequence[b] if warm_start_sequence!=[] else [])
            tours[b] = tour
            times[b] = time
        return tours, times

    def run_cplex_npfs_batched(self, x, bo, warm_start_sequence = [], leave=True):
        bs = x.shape[0]
        tours = -torch.ones((bs, x.shape[1]*x.shape[2]), device=x.device, dtype=torch.long)
        times = -torch.ones((bs), device=x.device, dtype=torch.float64)
        for b in tqdm(range(bs), leave=leave): # no option other than lazy batching
            tour, time = self.run_cplex_npfs(x[b], bo[b] if bo is not None else None, warm_start_sequence[b] if warm_start_sequence!=[] else [])
            tours[b] = tour
            times[b] = time
        return tours, times

    def run_cplex_pfs(self, x, bo, warm_start_sequence = []):

        assert len(x.shape) == 2, "Shape of x must not be batched"

        NB_JOBS = x.shape[0]
        NB_MACHINES = x.shape[1]

        OP_DURATIONS = x
        OP_DURATIONS = torch.tensor(OP_DURATIONS, dtype=torch.int32).cpu().numpy()

        BO_TIMES = None

        # Create model
        mdl = CpoModel()

        operations = [[interval_var(size=OP_DURATIONS[j][m], name='J{}-M{}'.format(j, m)) for m in range(NB_MACHINES)] for j in range(NB_JOBS)] # [job machine]

        # set warm start
        # For now, we are not setting any ws
        if warm_start_sequence != []:
            warm_start_sequence = build_warm_start_pfs(warm_start_sequence, operations, OP_DURATIONS, BO_TIMES, mdl)
            mdl.set_starting_point(warm_start_sequence)

        op_sequences = [sequence_var([operations[i][j] for i in range(NB_JOBS)], name='M{}'.format(j)) for j in range(NB_MACHINES)] # [job machine]

        # Force each operation to start after the end of the previous
        for j in range(NB_JOBS):
            for m in range(1, NB_MACHINES):
                mdl.add(end_before_start(operations[j][m-1], operations[j][m]))

        # Force no overlap for operations executed on a same machine
        for m in range(NB_MACHINES):
            if bo is not None:
                mdl.add(no_overlap(op_sequences[m], distance_matrix=BO_TIMES[m]))
            else:
                mdl.add(no_overlap(op_sequences[m]))

        # Force sequences to be all identical on all machines for pfs
        for m in range(1, NB_MACHINES):
            mdl.add(same_sequence(op_sequences[0], op_sequences[m]))

        # Minimize termination date
        mdl.add(minimize(max([end_of(operations[i][NB_MACHINES-1]) for i in range(NB_JOBS)])))

        cplex_start = time.perf_counter()
        if self.solution_callback != None:
            mdl.add_solver_callback(self.solution_callback)


        # setting the time limit to an hour
        res = mdl.solve(FailLimit=9999999999, TimeLimit=self.time_limit, params=self.parameters)
        cplex_time = time.perf_counter() - cplex_start

        # print permutation solution
        solution = [(res.get_var_solution(operations[j][0]).get_start(), j) for j in range(NB_JOBS)] #get solution for the first machine
        solution = sorted(solution, key=lambda item:item[0]) #sort solution by the start time
        tour_cplex = [item[1] for item in solution]
        tour_cplex = torch.tensor(tour_cplex, device=x.device)
        return tour_cplex, cplex_time

    def run_cplex_npfs(self, x, bo, warm_start_sequence = []):

        assert len(x.shape) == 2, "Shape of x must not be batched"

        NB_JOBS = x.shape[0]
        NB_MACHINES = x.shape[1]

        OP_DURATIONS = x
        OP_DURATIONS = torch.tensor(OP_DURATIONS, dtype=torch.int32).cpu().numpy()

        BO_TIMES = None

        # Create model
        mdl = CpoModel()


        operations = [[interval_var(size=OP_DURATIONS[j][m], name='J{}-M{}'.format(j, m)) for m in range(NB_MACHINES)] for j in range(NB_JOBS)] # [job machine]

        # set warm start
        # For now, we are setting ws only from pfs tours
        if warm_start_sequence != []:
            warm_start_sequence = build_warm_start_pfs(warm_start_sequence, operations, OP_DURATIONS, BO_TIMES, mdl)
            mdl.set_starting_point(warm_start_sequence)

        op_sequences = [sequence_var([operations[i][j] for i in range(NB_JOBS)], name='M{}'.format(j)) for j in range(NB_MACHINES)] # [job machine]

        # Force each operation to start after the end of the previous
        for j in range(NB_JOBS):
            for m in range(1, NB_MACHINES):
                mdl.add(end_before_start(operations[j][m-1], operations[j][m]))

        # Force no overlap for operations executed on a same machine
        for m in range(NB_MACHINES):
            if bo is not None:
                mdl.add(no_overlap(op_sequences[m], distance_matrix=BO_TIMES[m]))
            else:
                mdl.add(no_overlap(op_sequences[m]))

        # Minimize termination date
        mdl.add(minimize(max([end_of(operations[i][NB_MACHINES-1]) for i in range(NB_JOBS)])))

        # setting the time limit to an hour
        cplex_start = time.perf_counter()
        res = mdl.solve(FailLimit=9999999999, TimeLimit=self.time_limit, params=self.parameters)
        cplex_time = time.perf_counter() - cplex_start

        # print permutation solution
        result = -torch.ones((NB_MACHINES, NB_JOBS), dtype=torch.long)
        for m in range(NB_MACHINES):
            solution = [(res.get_var_solution(operations[j][m]).get_start(), j) for j in range(NB_JOBS)] #get solution for the first machine
            solution = sorted(solution, key=lambda item:item[0]) #sort solution by the start time
            tour_cplex = [item[1] for item in solution]
            result[m] = torch.tensor(tour_cplex, device=x.device, dtype=torch.long) + torch.tensor(NB_JOBS * m).repeat((NB_JOBS))

        result = result.flatten()
        return result, cplex_time
