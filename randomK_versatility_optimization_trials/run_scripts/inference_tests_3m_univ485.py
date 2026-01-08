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
parser0.add_argument('--n_rounds', default=1, type=int)
parser0.add_argument('--n_params_per_round', default=1, type=int)
parser0.add_argument('--n_weights_per_round', default=100, type=int)
FLAGS_run, __ = parser0.parse_known_args()

parser1 = argparse.ArgumentParser()
## Settings for TuneK
parser1.add_argument('--base_dir', default='../results/infer_485/a-outer_w-inner_allWeights', type=str) # base directory for output
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
parser2.add_argument('--polish_K', default=1, type=int)
parser2.add_argument('--workers_K', default=1, type=int) # default is to use 1 worker (not paralleized). -1 uses all available workers!
parser2.add_argument('--nstarts_K', default=2, type=int) # number of outer optimizations to be restarted for learning K
parser2.add_argument('--nxsurface_K', default=100, type=int) # number gridpoints for loss surface
FLAGS_diffev, __ = parser2.parse_known_args()
opt_setts_K = make_opt_settings(FLAGS_diffev.__dict__)

def do_opt(opt_setts, foo, K_true=[], seed=None):
    n_var = len(foo.param_lb)

    problem = FunctionalProblem(
        n_var=n_var,
        objs=foo.loss_k,
        xl=foo.param_lb,
        xu=foo.param_ub,
        constr_ieq=[])

    termination = SingleObjectiveDefaultTermination(
        x_tol=1e-6,
        cv_tol=0.0,
        f_tol=1e-6,
        nth_gen=max(1, int(opt_setts['maxiter']/4)),
        n_last=20,
        n_max_gen=opt_setts['maxiter'])

    algorithm = DE(CR=0.9,
        pop_size=opt_setts['popsize']*n_var)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        save_history=True,
        verbose=True)

    k_opt = res.X
    k_opt_loss = res.F

    # make plots and check the solution
    print('\n## Now running/plotting final optimal values... ##')
    foo.loss_k(k_opt, final_run=True)

    analyzeOpt = AnalyzePymoo([res], foo.Knames, truth=K_true)
    analyzeOpt.make_plots(foo.output_dir)
    return res

def opt_wrapper(opt_setts, n_rounds=10, n_params_per_round=1, n_weights_per_round=100):

    datadir = '../data/sims_3m'

    # read in relevant simulation files
    M = FLAGS_tune.m

    # dimer curves: discretization X (# monomers and dimers) X (# parameters)
    S = np.load(os.path.join(datadir, 'S_all_{}M_1000k.npy'.format(M)))

    # weights: (# random weights) X (# dimers)
    weights = np.load(os.path.join(datadir, 'out_weights_{}M_LHSsample_1000k.npy'.format(M))).T

    # output: discretization X parameters (K,a0) X weights
    output = np.load(os.path.join(datadir, 'output_{}M_LHSsample_1000k.npy'.format(M)))
    output = np.moveaxis(output, (0,1,2), (2,0,1)) # parameters (K,a0) X weights X discretization

    # load K, a0
    params = np.load(os.path.join(datadir, 'param_sets_{}M_1000k.npy'.format(M)))
    params = np.log10(params)
    Ksim = params[:,:-(M-1)] # samples X K values
    a0sim = params[:,-(M-1):] # samples X acc index

    # first, fit target functions with known true K,a,theta
    for c in range(n_rounds):
        FOO = TuneK(**tune_dict)
        # FOO.algorithm = DE(CR=0.9)

        # draw random param index
        ip = np.random.randint(Ksim.shape[0], size=n_params_per_round)
        iw = np.random.randint(weights.shape[0], size=n_weights_per_round)
        print('ip', ip)
        print('iw', iw)

        K = Ksim[ip] # binding affinity
        a0 = np.squeeze(a0sim[ip]) # accessory monomer concentration
        theta = np.atleast_2d(weights[iw]) # dimer weights
        F = np.atleast_2d(output[ip, iw]) #get function

        # set the target
        FOO.set_target(F) #wipes other targets and focuses on this one (interpolates, then re-discretizes)

        # set the TRUTH (only used for plotting!!!!)
        FOO.truth = {'K': K, 'a0': a0, 'theta': theta}

        # Evaluate loss surface
        # FOO.loss_surface_k(K)

        # Evaluate inner convergence quality for true K
        FOO.loss_k(K, n_starts=2, final_run=True, plot_surface=True, extra_nm='TrueK')
        print('True values are...')
        print('a0:',a0)
        print('theta:',theta)

        dump_dict = {'ip': ip, 'iw': iw, 'truth': FOO.truth}
        dump_data(dump_dict, fname=os.path.join(FOO.output_dir,'experiment_info.pkl'))

        # next, perform full optimization over K and a, theta
        res = do_opt(opt_setts, FOO, seed=0, K_true=np.squeeze(K))

        # finally, evaluate inference (do we converge to true K, a, theta)?
        print('True values are...')
        print('K:',K)
        print('a0:',a0)
        print('theta:',theta)

        FOO.truth = {}
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
