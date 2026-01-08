import os, sys

import numpy as np

sys.path.append('../code')

from utilities import make_opt_settings, sec2friendly, sample_concentrations
from makefuncs import set_target_library
from opt_utils import AnalyzePymoo, minimize_wrapper, analyze_convergence
from pymoo.algorithms.soo.nonconvex.de import DE

# from scipy.optimize import minimize, brute, differential_evolution
# from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.problems.functional import FunctionalProblem
from pymoo.optimize import minimize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.operators.sampling.lhs import LHS
from tuning import TuneK
from pdb import set_trace as bp
import argparse
from time import time

parser1 = argparse.ArgumentParser()
## Settings for TuneK
parser1.add_argument('--base_dir', default='../results/default', type=str) # base directory for output
parser1.add_argument('--target_lib_name', default='SinCos', type=str) # Name for target library
parser1.add_argument('--target_lib_file', default='../data/metaclusters/hc_3M_metaClusterBasis_thresh3.npy', type=str) # file for reading target functions
parser1.add_argument('--m', default=3, type=int) #number of total monomers
parser1.add_argument('--n_input_samples', default=40, type=int) #Number of values to titrate the input monomer species. Values spaced evenly on a log10 scale
parser1.add_argument('--acc_opt', default="outer", type=str)
parser1.add_argument('--w_opt', default="inner", type=str)
parser1.add_argument('--single_beta', default=0, type=int)
parser1.add_argument('--plot_inner_opt', default=1, type=int)
FLAGS_tune, __ = parser1.parse_known_args()

# Settings for Differential Evolution optimizer within TuneK (i.e. for a single call of loss(K))
parser3 = argparse.ArgumentParser()
parser3.add_argument('--maxiter_O', default=15, type=int)
parser3.add_argument('--popsize_O', default=10, type=int)
parser3.add_argument('--polish_O', default=1, type=int)
parser3.add_argument('--workers_O', default=1, type=int) # default is to use 1 worker (not paralleized). -1 uses all available workers!
FLAGS_optsetts, __ = parser3.parse_known_args()
tune_dict = FLAGS_tune.__dict__
tune_dict['opt_settings_outer'] = make_opt_settings(FLAGS_optsetts.__dict__)
FOO = TuneK(**tune_dict)

# Settings for Differential Evolution optimizer over binding affinity K
parser2 = argparse.ArgumentParser()
parser2.add_argument('--maxiter_K', default=15, type=int)
parser2.add_argument('--popsize_K', default=10, type=int)
parser2.add_argument('--polish_K', default=1, type=int)
parser2.add_argument('--workers_K', default=1, type=int) # default is to use 1 worker (not paralleized). -1 uses all available workers!
FLAGS_diffev, __ = parser2.parse_known_args()
opt_setts_K = make_opt_settings(FLAGS_diffev.__dict__)

def my_loss(x):
    return FOO.loss_k(x)

def opt_wrapper(opt_setts, seed=12, Nk=10):
    n_var = FOO.m - 1

    np.random.seed(seed)
    K_list = np.random.uniform(low=FOO.param_lb, high=FOO.param_ub, size=(Nk,len(FOO.param_lb)))
    for i in range(10):
        np.random.seed(seed)
        # K = np.random.uniform(low=FOO.param_lb, high=FOO.param_ub)
        FOO.algorithm = DE(CR=0.9)
            # CR=CR,
            # pop_size=10*n_var,
            # variant="DE/rand/1/bin",
            # dither="vector",
            # jitter=False)
        K = K_list[i]
        FOO.loss_k(K, final_run=True)



######
if __name__ == '__main__':

    analyze_convergence('../results/default/07252022105441_acc-outer_weights-inner')

    t0 = time()
    opt_wrapper(opt_setts_K)
    # run overall analysis
    analyze_convergence(FOO.output_dir)
    print('# Total run time = {}'.format(sec2friendly(time()-t0)))
