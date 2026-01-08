import os, sys

import numpy as np

sys.path.append('../code')

from utilities import make_opt_settings, sec2friendly
from makefuncs import set_target_library
from optplot import AnalyzePymoo

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
parser1.add_argument('--target_lib_file', default='../data/hc_3M_metaClusterBasis_thresh3.npy', type=str) # file for reading target functions
parser1.add_argument('--m', default=3, type=int) #number of total monomers
parser1.add_argument('--n_input_samples', default=40, type=int) #Number of values to titrate the input monomer species. Values spaced evenly on a log10 scale
parser1.add_argument('--acc_opt', default="outer", type=str)
parser1.add_argument('--w_opt', default="inner", type=str)
FLAGS_tune, __ = parser1.parse_known_args()

# Settings for Differential Evolution optimizer within TuneK (i.e. for a single call of loss(K))
parser3 = argparse.ArgumentParser()
parser3.add_argument('--maxiter_O', default=3, type=int)
parser3.add_argument('--popsize_O', default=15, type=int)
parser3.add_argument('--polish_O', default=1, type=int)
parser3.add_argument('--workers_O', default=1, type=int) # default is to use 1 worker (not paralleized). -1 uses all available workers!
FLAGS_optsetts, __ = parser3.parse_known_args()
tune_dict = FLAGS_tune.__dict__
tune_dict['opt_settings_outer'] = make_opt_settings(FLAGS_optsetts.__dict__)
FOO = TuneK(**tune_dict)

# Settings for Differential Evolution optimizer over binding affinity K
parser2 = argparse.ArgumentParser()
parser2.add_argument('--maxiter_K', default=2, type=int)
parser2.add_argument('--popsize_K', default=3, type=int)
parser2.add_argument('--polish_K', default=1, type=int)
parser2.add_argument('--workers_K', default=1, type=int) # default is to use 1 worker (not paralleized). -1 uses all available workers!
FLAGS_diffev, __ = parser2.parse_known_args()
opt_setts_K = make_opt_settings(FLAGS_diffev.__dict__)


def my_loss(x):
    return FOO.loss_k(x)

def do_opt(opt_setts):

    problem = FunctionalProblem(
        n_var=len(FOO.param_lb),
        objs=my_loss,
        xl=FOO.param_lb,
        xu=FOO.param_ub,
        constr_ieq=[])

    termination = SingleObjectiveDefaultTermination(
        x_tol=1e-6,
        cv_tol=0.0,
        f_tol=1e-6,
        nth_gen=5,
        n_last=20,
        n_max_gen=opt_setts['maxiter'])

    algorithm = DE(
        pop_size=opt_setts['popsize'],
        sampling=LHS(),
        variant="DE/rand/1/bin",
        CR=0.3,
        dither="vector",
        jitter=False)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=12,
        save_history=True,
        verbose=True)

    k_opt = res.X
    k_opt_loss = res.F

    # make plots and check the solution
    print('\n## Now running/plotting final optimal values... ##')
    FOO.loss_k(k_opt, final_run=True)

    analyzeOpt = AnalyzePymoo(res, FOO.output_dir, FOO.Knames)
    analyzeOpt.make_plots()


######
if __name__ == '__main__':
    t0 = time()
    do_opt(opt_setts_K)
    print('# Total run time = {}'.format(sec2friendly(time()-t0)))
