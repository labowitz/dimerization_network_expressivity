import os, sys

import numpy as np

sys.path.append('../code')

from utilities import *
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

parser0 = argparse.ArgumentParser()
## Run settings
parser0.add_argument('--n_rounds', default=10, type=int)
parser0.add_argument('--n_targets_per_round', default=10, type=int)
FLAGS_run, __ = parser0.parse_known_args()

parser1 = argparse.ArgumentParser()
## Settings for TuneK
parser1.add_argument('--base_dir', default='../results/infer_metaclusters/a-outer_w-inner', type=str) # base directory for output
parser1.add_argument('--target_lib_name', default='SinCos', type=str) # Name for target library
parser1.add_argument('--target_lib_file', default='../data/metaclusters/hc_3M_metaClusterBasis_thresh3.npy', type=str) # file for reading target functions
parser1.add_argument('--m', default=3, type=int) #number of total monomers
parser1.add_argument('--n_input_samples', default=40, type=int) #Number of values to titrate the input monomer species. Values spaced evenly on a log10 scale
parser1.add_argument('--acc_opt', default="outer", type=str)
parser1.add_argument('--w_opt', default="inner", type=str)
parser1.add_argument('--single_beta', default=0, type=int)
parser1.add_argument('--polish', default=0, type=int)
parser1.add_argument('--lsq_linear_method', default='bvls', type=str)
parser1.add_argument('--plot_inner_opt', default=1, type=int)
FLAGS_tune, __ = parser1.parse_known_args()

# Settings for Differential Evolution optimizer within TuneK (i.e. for a single call of loss(K))
parser3 = argparse.ArgumentParser()
parser3.add_argument('--maxiter_O', default=15, type=int)
parser3.add_argument('--popsize_O', default=50, type=int)
parser3.add_argument('--polish_O', default=1, type=int)
parser3.add_argument('--workers_O', default=1, type=int) # default is to use 1 worker (not paralleized). -1 uses all available workers!
FLAGS_optsetts, __ = parser3.parse_known_args()
tune_dict = FLAGS_tune.__dict__
tune_dict['opt_settings_outer'] = make_opt_settings(FLAGS_optsetts.__dict__)

# Settings for Differential Evolution optimizer over binding affinity K
parser2 = argparse.ArgumentParser()
parser2.add_argument('--maxiter_K', default=3, type=int)
parser2.add_argument('--popsize_K', default=20, type=int)
parser2.add_argument('--polish_K', default=0, type=int)
parser2.add_argument('--workers_K', default=1, type=int) # default is to use 1 worker (not paralleized). -1 uses all available workers!
parser2.add_argument('--nstarts_K', default=2, type=int) # number of outer optimizations to be restarted for learning K
parser2.add_argument('--nxsurface_K', default=100, type=int) # number gridpoints for loss surface
FLAGS_diffev, __ = parser2.parse_known_args()
opt_setts_K = make_opt_settings(FLAGS_diffev.__dict__)

def opt_wrapper(opt_setts, n_rounds=10, n_targets_per_round=50):

    datadir = '../data/sims_3m'
    clusterdir = '../data/metaclusters'

    # read in relevant simulation files
    M = FLAGS_tune.m

    # load meta-cluster instead of direct outputs
    output = np.atleast_2d(np.load(os.path.join(clusterdir, 'hc_{}M_metaClusterBasis_thresh3.npy'.format(M))))

    for c in range(n_rounds):
        ind_F = np.random.randint(output.shape[0], size=n_targets_per_round)
        F = output[ind_F]

        FOO = TuneK(**tune_dict)
        FOO.lsq_bounds = (-np.Inf, np.Inf)

        # set the target
        FOO.set_target(F) #wipes other targets and focuses on this one (interpolates, then re-discretizes)

        # next, perform full optimization over K and a, theta
        FOO.optimize_binding(seed=0, **opt_setts)
    return

######
if __name__ == '__main__':

    # analyze_convergence('../results/default/07252022105441_acc-outer_weights-inner')
    np.random.seed(0)
    t0 = time()
    opt_wrapper(opt_setts_K, **FLAGS_run.__dict__)
    # run overall analysis
    # analyze_convergence(FOO.output_dir)
    print('# Total run time = {}'.format(sec2friendly(time()-t0)))
