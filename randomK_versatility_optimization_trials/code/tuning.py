import os, sys

import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

import functools
import numpy as np
import pandas as pd
import scipy.stats
import itertools
import multiprocessing
from scipy.optimize import brute, differential_evolution

from pymoo.problems.functional import FunctionalProblem
from pymoo.optimize import minimize
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
from pymoo.core.population import Population
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.nelder_mead import NelderMead
from pymoo.operators.sampling.lhs import LHS
from pymoo.core.problem import starmap_parallelized_eval
from opt_utils import *

# %load_ext watermark

from pdb import set_trace as bp
import time

# If you need to install eqtk, you can uncomment and run the cell below.
# Or check out https://eqtk.github.io/getting_started/eqtk_installation.html.
# It can take several seconds to load the package into python.
# Additionally, the first time you run the equilibrium solver (eqtk.solve)
# may take longer than subsequent calls as the program compiles the code with Numba
#!{sys.executable} -m pip install --upgrade eqtk
import eqtk

#load some utility functions
# pwd = os.path.abspath('..')
# sys.path.append(os.path.join(pwd, 'code/'))
from utilities import *
from makefuncs import *

plt.rcParams.update({'font.size': 22, 'legend.fontsize': 12,
                'legend.facecolor': 'white', 'legend.framealpha': 0.8,
                'legend.loc': 'upper left', 'lines.linewidth': 4.0})

default_colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

class TuneK:
    def __init__(self,
                    base_dir = '../optimization_results',
                    target_lib_name = 'SinCos',
                    start = "both", # start libraries as "on", "off", or "both"
                    n_switches = 2, # number of switches for bump library
                    n_switch_points = 5, # number of locations to place the switches
                    target_lib_file = None,
                    target_lib_names_file = None,
                    acc_opt = 'inner',
                    w_opt = 'inner',
                    single_beta = False,
                    scale_type = "global", # "per-dimer", "per-target"
                    scale_bounds = (-14,14), # in log10 space
                    opt_settings_outer = {}, # use default settings for differential evolution
                    m = 3,
                    n_input_samples = 30, #discretization of first monomer...time complexity scales linearly w.r.t. this parameter
                    n_accessory_samples = 75, # number of latin-hypercube samples from the space of initial accessory concentrations...time complexity scales linearly w.r.t. this parameter
                    dim_input = 1, # dimension of input
                    floor_dimers = True,
                    input_lb = -3,
                    input_ub = 3,
                    acc_lb = -3,
                    acc_ub = 3,
                    param_lb = -10, #-5,
                    param_ub = 7, # 7,
                    id_dimer = None,
                    plot_inner_opt = True,
                    make_plots = True,
                    polish = False,
                    lsq_linear_method = 'bvls',
                    lsq_bounds = (0, np.Inf),
                    dimer_eps=1e-16,
                    log_errors=True,
                    no_rescaling=True,
                    nxsurface=10,
                    id_target=None,
                    grid_dir=None,
                    randomizeK=False,
                    id_K=None,
                    K_list_file='K_list.npy',
                    inner_opt_seed=99,
                    abort_early=False,
                    **kwargs):
        """
        Run simulations for dimer networks of size m and input titration size t
        with k different parameter universes.

        Parameters
        ----------
        m : int.
            Number of monomer species.
        n_input_samples : int. Default 10.
            Number of values to titrate the input monomer species.
            Values spaced evenly on a log10 scale
        n_accessory_samples : int. Default 10.
            Number of values to titrate the accessory monomer species concentrations.
            This is performed with either LatinHypercube sampling
            OR
            denotes the maximum evaluations of an optimization over accessory concentrations.
            Values spaced evenly on a log10 scale.
        input_lb : int. Default -3
            lower bound for titrating the input monomer species. log10 scale
        input_ub : int. Default 3
            upper bound for titrating the input monomer species. log10 scale
        param_lb : int. Default -6
            Lower bounds for binding affinity parameters.
        param_ub : int. Default 6
            Upper bounds for binding affinity parameters.
        base_dir : string. Default 'results'
            relative path for saving figure.
        target_lib_name : string. Default 'SinCos'
            Name of specially designed library to which we will fit.
        """
        self.base_dir = base_dir
        self.m = m
        self.n_input_samples = n_input_samples
        self.n_accessory_samples = n_accessory_samples
        self.dim_input = dim_input
        self.n_acc = m - dim_input
        self.input_lb = input_lb
        self.input_ub = input_ub
        self.acc_lb = acc_lb
        self.acc_ub = acc_ub
        self.floor_dimers = floor_dimers # set small dimer concentrations to a limit...e.g. anything below 1e-3 gets set to 1e-3.

        self.abort_early = abort_early # use to run self.outer_opt fewer times for randomK experiments that don't need final plots.
        self.randomizeK = randomizeK
        self.id_K = id_K
        self.K_list_file = K_list_file # list of K's to try (will use the one indexed by id_K...). only active if id_K is not None.
        self.inner_opt_seed = inner_opt_seed

        self.param_lb = param_lb
        self.param_ub = param_ub
        self.scale_bounds = scale_bounds
        self.opt_settings_outer = opt_settings_outer
        self.polish = polish

        self.id_dimer = id_dimer #Set to an integer to choose which dimer output to consider. If None, find_best_beta will try them all.

        self.dimer_eps = dimer_eps # threshold below which to set to 0 for equilibrium levels.
        self.log_errors = log_errors # if True, computes errors w.r.t. target function as |log(f_target) - log(f_approx)|^2
        self.no_rescaling = no_rescaling # sets all betas = 1 or 0

        self.lsq_bounds = lsq_bounds #(0, np.Inf)

        self.make_plots = make_plots
        self.plot_inner_opt = plot_inner_opt
        self.nxsurface = nxsurface # number of grid points per axis on inner optimization plots

        self.truth = {'K': [], 'a0': [], 'theta': []} # useful for plotting inferences vs truth

        self.setup()

        self.single_beta = single_beta
        self.scale_type = scale_type
        if self.scale_type in ["per-dimer", "global"]:
            self.one_scale = True
        elif self.scale_type == "per-target":
            self.one_scale = False
        else:
            raise('Scale Type not recognized')
        self.acc_opt = acc_opt
        self.w_opt = w_opt
        self.lsq_linear_method = lsq_linear_method

        self.make_output_dir()

        try:
            target_lib_file = target_lib_file.format(self.m)
        except:
            pass

        self.f_targets = set_target_library(n_input_samples=n_input_samples, target_lib_name=target_lib_name, target_lib_file=target_lib_file, target_lib_names_file=target_lib_names_file, n_switches=n_switches, n_switch_points=n_switch_points, start=start)
        if id_target is not None:
            self.f_targets = self.f_targets[id_target].reshape(1,-1)

        if self.log_errors:
            self.f_targets = np.log10(self.f_targets)

        self.n_targets = self.f_targets.shape[0]
        self.f_targets_max_sq = [1] #np.max(self.f_targets, axis=1)**2
        self.f_targets_max_sq[self.f_targets_max_sq==0] = 1 #avoid divide by zero

        if self.make_plots:
            self.plot_targets(output_dir=self.output_dir)

        self.set_opts()
        try:
            self.brute_info_dict, (self.K_sorted, self.c0_sorted, self.mse_sorted) = self.get_brute_K(grid_dir)
            self.set_opts()
        except:
            print('COULD NOT LOAD GRID DATA from', grid_dir)
            print('Currently in ', os.getcwd())

    def get_brute_K(self, grid_dir):
        curve_path = os.path.join(grid_dir, '{}m_S_all.npy'.format(self.m))

        # I think the first m belong are monomers?
        jacob = np.load(curve_path, allow_pickle=True) # 30 x dimers x N

        ### NOTE: the first entries belong to K, then the last few entries belong to accessory a.
        param_path = os.path.join(grid_dir, '{}m_K_A_param_sets.npy'.format(self.m))
        params = np.load(param_path, allow_pickle=True) # 30 x dimers x N
        params = np.log10(params)
        param_ub = np.max(params, 0)
        param_lb = np.min(params, 0)
        print('Grid upper bounds:', param_ub)
        print('Grid lower bounds:', param_lb)

        ub_stack = np.hstack((self.param_ub,self.acc_ub_list))
        lb_stack = np.hstack((self.param_lb,self.acc_lb_list))
        print('Optimization upper bounds:', ub_stack)
        print('Optimization lower bounds:', lb_stack)

        ub_violate = ub_stack < param_ub
        lb_violate = lb_stack > param_lb
        if any(ub_violate):
            print('Grid violates upper bounds!', ub_violate)
        if any(lb_violate):
            print('Grid violates lower bounds!', lb_violate)

        # for i_dimer in range(10): #range(jacob.shape[1]):
        mse_vec = np.zeros(jacob.shape[2])
        for i_param in range(jacob.shape[2]):
            # guessing that the first N belong to the monomers
            dimers = np.log10(jacob[:,-self.n_dimers:,i_param])
            theta_star, mse_total, mse_list = self.simple_loss(dimers)
            mse_vec[i_param] = mse_total

        #order the curves by increasing mse!
        inds = np.argsort(mse_vec)

        mse_sorted = mse_vec[inds]
        params_sorted = params[inds]

        K_sorted = params_sorted[:,:self.n_dimers]
        K_sorted[K_sorted < self.param_lb] = self.param_lb[0]
        K_sorted[K_sorted > self.param_ub] = self.param_ub[0]

        c0_sorted = params_sorted[:,self.n_dimers:]
        c0_sorted[c0_sorted < self.acc_lb] = self.acc_lb
        c0_sorted[c0_sorted > self.acc_ub] = self.acc_ub

        # print raw params
        print('Raw best grid params are...')
        print('K:', K_sorted[0])
        print('c0:', c0_sorted[0])
        print('MSE = ', mse_sorted[0])
        print('Re-check MSE:', self.simple_loss(self.g1(c0_sorted[0], K_sorted[0])))

        # re-order each param entry so that it obeys the descending constraint (for uniqueness)
        for j in range(K_sorted.shape[0]):
            K_sorted[j], new_inds = sort_K_ascending(K_sorted[j], self.m, n_input=self.dim_input)
            c0_inds = new_inds[self.acc_monomer_ind] - self.dim_input
            c0_sorted[j] = c0_sorted[j,c0_inds]

        # store optimal params
        # check `make_opt_output_list` to see when this shape fails. should be fine for now.
        K = K_sorted[0]
        c0 = c0_sorted[0]
        opt_list = [DotDict({'X': c0})]
        info_dict = self.make_opt_output_list(opt_list, K)[0]
        info_dict['K'] = K

        print('Printing best grid element (after re-ordering)...')
        print('K:', K_sorted[0])
        print('c0:', c0_sorted[0])
        print('MSE = ', mse_sorted[0])
        # double check stuff:
        print('Re-check MSE:', self.simple_loss(self.g1(c0, K)))

        return info_dict, (K_sorted, c0_sorted, mse_sorted)

    def plot_targets(self, output_dir, fits=None):
        if self.log_errors: # convert to regular units, since plot_targets plots things in log-coordinates
            f_targets = 10**self.f_targets
            if fits is not None:
                fits = 10**fits
        plot_targets(output_dir, self.input_concentration, f_targets, fits)

    def set_target(self, F):
        '''Overwrite existing targets'''
        self.f_targets = interp_target(self.n_input_samples, F)
        self.n_targets = self.f_targets.shape[0]
        self.f_targets_max_sq = 1 #np.max(self.f_targets, axis=1)**2

    def mse(self, f_true, f_pred):
        '''Mean Squared Error'''
        return np.mean( (f_true - f_pred)**2 )

    def make_output_dir(self):
        # t = time.localtime()
        # timestamp = time.strftime('%m%d%Y%H%M%S', t)
        # base_nm = self.make_experiment_name()
        # self.output_dir = os.path.join(self.base_dir, timestamp + '_' + base_nm)
        self.output_dir = self.base_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def make_experiment_name(self):
        foo = {'acc': self.acc_opt, 'weights': self.w_opt}
        mystr = '_'.join([key + '-' + str(foo[key]) for key in foo])
        return mystr

    def setup(self):

        self.Knames = np.array(make_Kij_names(n_input=self.dim_input, n_accesory=self.n_acc))
        self.N = make_nXn_stoich_matrix(self.m)
        self.num_rxns = self.N.shape[0]
        self.M0_min = [self.input_lb]*self.dim_input + [0] * (self.n_acc)
        self.M0_max = [self.input_ub]*self.dim_input + [0] * (self.n_acc)
        self.num_conc = [self.n_input_samples]*self.dim_input + [1] * (self.n_acc)

        self.n_dimers = len(self.Knames)
        self.C0 = make_C0_grid(self.m, M0_min=self.M0_min, M0_max=self.M0_max, num_conc=self.num_conc)
        self.input_concentration = self.C0[:,:self.dim_input] # input monomer

        self.num_params = self.num_rxns + (self.n_acc)

        self.acc_monomer_ind = np.arange(self.dim_input,self.m)

        #Parameter bounds for sampling and for titration
        if self.param_lb is None:
            self.param_lb = [-6]*self.num_rxns + [-3]*(self.n_acc)

        if self.param_ub is None:
            self.param_ub = [6]*self.num_rxns + [3]*(self.n_acc)

        self.acc_lb_list = [self.acc_lb]*(self.n_acc)
        self.acc_ub_list = [self.acc_ub]*(self.n_acc)
        # self.acc_list = sample_concentrations(self.n_acc, self.n_accessory_samples, self.acc_lb_list, self.acc_ub_list, do_power=False) # 75 (number of draws) x 2 (number of accessories)

        self.param_lb = [self.param_lb]*self.num_rxns
        self.param_ub = [self.param_ub]*self.num_rxns


    def diff(self, x, ilow, ihigh):
        return x[ihigh] - x[ilow]

    def optimize_binding(self, popsize=15, maxiter=10, seed=None, nstarts=1, polish=False, plot_surface=False, do_constraints=True, dothreading=False, make_plots=True, **kwargs):
        res_list = []

        for ns in range(nstarts):

            constr_iq = []
            if do_constraints:
                ## TODO: NEED TO CHECK THIS and ignore all input homodimers (need to get their indices)
                ind_list = get_diag_inds(n_input=self.dim_input, n_accesory=self.n_acc, m = self.m)[self.dim_input:] # ignore input homodimers
                constr_iq = [functools.partial(self.diff, ilow=ind_list[j-1], ihigh=ind_list[j])
                                for j in range(1,len(ind_list)) ]

            n_var = len(self.param_lb)
            problem = FunctionalProblem(
                n_var=n_var,
                objs=self.loss_k,
                xl=self.param_lb,
                xu=self.param_ub,
                constr_ieq=constr_iq)

            if dothreading:
                # the number of processes to be used
                n_process = multiprocessing.cpu_count()
                pool = multiprocessing.Pool(n_process)

                print('Running on {} CPUs'.format(n_process))

                problem.runner = pool.starmap
                problem.func_eval = starmap_parallelized_eval

            termination = SingleObjectiveDefaultTermination(
                x_tol=1e-6,
                cv_tol=0.0,
                f_tol=1e-6,
                nth_gen=max(1, int(maxiter/4)),
                n_last=20,
                n_max_gen=maxiter)

            if self.id_K is not None:
                np.random.seed(self.id_K)
                if self.randomizeK:
                    K0 = np.random.uniform(low=self.param_lb, high=self.param_ub, size=(self.num_rxns))
                else:
                    K0 = np.log10(np.load(self.K_list_file)[self.id_K])

                K0 = sort_K_ascending(K0, self.m, n_input=self.dim_input)[0]
                print('idK = ', self.id_K, 'yields K:', K0)
                pop = Population.new("X", K0.reshape(1,-1)) #, "F", F)
                algorithm = DE(pop_size=popsize, sampling=pop)
            else:
                try:
                    K_unique = pd.DataFrame(self.K_sorted).drop_duplicates().to_numpy()
                    if K_unique.shape[0] >= popsize:
                        X = K_unique[:popsize]
                    else:
                        X = self.K_sorted[:popsize]
                    # F = self.mse_sorted[:popsize]
                    pop = Population.new("X", X) #, "F", F)
                    # algorithm = GA(pop_size=popsize, sampling=pop)
                    algorithm = DE(pop_size=popsize, sampling=pop, CR=0.9)
                except:
                    print('UNABLE to initialize at brute force results')
                    # algorithm = GA(pop_size=popsize)
                    algorithm = DE(pop_size=popsize, CR=0.9)
            # algorithm = DE(CR=0.9, pop_size=popsize, sampling=pop)
            # algorithm = NelderMead(x0=X[0])


        # try to add brute force grid to results
        # try:
        #     # res =
        #     res_list += [res]
        # except:
        #     print('Unable to add brute force solution to result list.')

            # make plots and check the solution
            if nstarts==1:
                extra_nm = ''
            else:
                extra_nm = 'FinalK_{}'.format(ns)

            if self.abort_early:
                self.loss_k(K0, final_run=True, plot_surface=plot_surface, nxsurface=self.nxsurface, extra_nm=extra_nm)
                return

            res = minimize(
                problem,
                algorithm,
                termination,
                seed=ns,
                save_history=True,
                verbose=True)
            print('Optimization time:', res.exec_time, 'seconds')
            if polish:
                result = scipy.optimize.minimize(problem.objs[0],
                                  res.X,
                                  method='trust-constr',
                                  bounds=np.array([problem.xl, problem.xu]).T)
                res.X = result.x
                res.F = result.fun

            res_list += [res]

            k_opt = res.X
            k_opt_loss = res.F

            ## Compare to grid_K
            # dimers = self.g1(self.c0)
            try:
                opt_diff = k_opt_loss - self.mse_sorted[0]
                if opt_diff > 0:
                    print('UH OH----Optimization performed WORSE than its grid-based initialization by amount', opt_diff)
            except:
                pass

            print('\n## Now running/plotting final optimal values... ##')
            self.loss_k(k_opt, final_run=True, plot_surface=plot_surface, nxsurface=self.nxsurface, extra_nm=extra_nm)
            analyzeOpt = AnalyzePymoo([res], self.Knames, truth=self.truth['K'])
            percentile_list = [0,1] #[0, 1, 10, 50, 100]
            analyzeOpt.write_info(os.path.join(self.output_dir, extra_nm))
            if make_plots:
                analyzeOpt.make_plots(os.path.join(self.output_dir, extra_nm), percentile_list=percentile_list)

            if self.abort_early:
                return res

            # make some plots with the optimal K
            output_dir = os.path.join(self.output_dir, extra_nm)
            info_dict = self.outer_opt(k_opt, make_plots=False, verbose=False)[0] # returns list of optimization results for K
            info_dict['K'] = k_opt
            print(info_dict)
            self.plot_many_fits(output_dir, [info_dict])

            n_examples = 10
            for j in range(len(percentile_list)-1):
                k_examples = analyzeOpt.sample_X_from_grid(p_low=percentile_list[j], p_high=percentile_list[j+1], n=n_examples)
                param_list = []
                for K in k_examples:
                    output_list = self.outer_opt(K, make_plots=False, verbose=False) # returns list of optimization results for K
                    info_dict = output_list[0] # just use the first one
                    info_dict['K'] = K
                    param_list.append(info_dict)

                # make plots of this list of parameters
                output_dir = os.path.join(self.output_dir, 'PercentileSims_run{}/Percentile{}'.format(ns,percentile_list[j+1]))
                self.plot_many_fits(output_dir, param_list)

        # plot multi-start summary
        new_plot_dirname = os.path.join(self.output_dir,'Final_Summary')
        ap = AnalyzePymoo(res_list, self.Knames, truth=self.truth['K'])
        ap.write_info(new_plot_dirname)
        if make_plots:
            ap.make_plots(new_plot_dirname, percentile_list=percentile_list)
        for j in range(len(percentile_list)-1):
            k_examples = ap.sample_X_from_grid(p_low=percentile_list[j], p_high=percentile_list[j+1], n=n_examples)
            param_list = []
            for K in k_examples:
                output_list = self.outer_opt(K, make_plots=False, verbose=False) # returns list of optimization results for K
                info_dict = output_list[0] # just use the first one
                info_dict['K'] = K
                param_list.append(info_dict)

            # make plots of this list of parameters
            output_dir = os.path.join(self.output_dir, 'PercentileSims_All/Percentile{}'.format(percentile_list[j+1]))
            self.plot_many_fits(output_dir, param_list)

            # do percentile sims
            for n in range(k_examples.shape[0]):
                self.loss_k(k_examples[n], final_run=True, nxsurface=self.nxsurface,
                plot_surface=plot_surface,
                extra_nm='PercentileSims_All/Percentile{}/example{}'.format(percentile_list[j+1],n))

        return res

    def loss_surface_k(self, K, a_list=[], mse_list=[], nx=100, extra_nm=''):
        '''When fitting 1 accessory concentrations per-target, we plot the
        conditional loss surface for 1 target, assuming optimal choice of
        accessory for the other targets.'''
        figdir = os.path.join(self.output_dir, extra_nm)
        os.makedirs(figdir, exist_ok=True)

        a0 = np.linspace(self.acc_lb_list, self.acc_ub_list, nx).T
        xv, yv = np.meshgrid(*a0)

        if self.acc_opt=='inner':
            n_loops = self.n_targets
            c0_acc_full = np.copy(a_list[0])
        else:
            n_loops = 1

        for k in range(n_loops):
            if n_loops > 1:
                k_name = '_target{}'.format(k)
            else:
                k_name = ''

            zs = np.zeros((nx,nx))
            for i in range(nx):
                for j in range(nx):
                    # treat xv[i,j], yv[i,j]
                    c0_acc = np.array([xv[i,j], yv[i,j]])
                    if self.acc_opt=='inner' and self.w_opt=='inner' and not(self.one_scale):
                        _, mse_total = self.inner_opt(c0_acc, K, j_target=k)
                    if self.acc_opt=='inner' and self.w_opt=='inner' and self.one_scale:
                        c0_acc_full[k] = c0_acc
                        _, mse_total, mse_list = self.inner_opt(c0_acc_full.reshape(-1), K)
                    elif self.acc_opt=='inner' and self.w_opt=='outer':
                        c0_acc_full[k] = c0_acc
                        _, mse_total, mse_list = self.inner_opt(c0_acc_full.reshape(-1), K)
                    elif self.acc_opt=='outer' and self.w_opt=='inner':
                        _, mse_total, mse_list = self.inner_opt(c0_acc, K)
                    zs[i,j] = mse_total

            for nm in ['', '_log']:
                fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [10,10])
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
                if nm=='_log':
                    zfoo = np.log10(zs)
                else:
                    zfoo = zs
                cf = axs.contourf(a0[0], a0[1], zfoo)
                axs.set_xlabel('a_0')
                axs.set_ylabel('a_1')
                if len(self.truth['a0']):
                    axs.axvline(x=self.truth['a0'][0], color='red', linestyle='--', linewidth=1)
                    axs.axhline(y=self.truth['a0'][1], color='red', linestyle='--', linewidth=1)
                for c in range(len(a_list)):
                    aopt = a_list[c]
                    if aopt.ndim>1:
                        aopt = aopt[k]
                    mse = mse_list[c]
                    if nm=='_log':
                        mse = np.log10(mse)
                    axs.scatter(aopt[0], aopt[1], s=500, c=default_colors[c+1], marker='+', label='Run {} (MSE {})'.format(c, round(mse,3)))
                    axs.legend()
                cbar = fig.colorbar(cf)
                cbar.ax.set_ylabel('MSE' + nm)

                # axs.scatter(self.truth['a0'][0], self.truth['a0'][1], s=1000, c='red', marker="*")
                fig.savefig(os.path.join(figdir, 'inner_loss_surface{}{}.pdf'.format(k_name, nm)), format='pdf')
                plt.close()


    def loss_k(self, K, n_starts=1, final_run=False, verbose=False, normalize_plot=False, plot_surface=False, extra_nm='Running', nxsurface=100):
        self.n_starts = n_starts
        if extra_nm=='':
            use_extra_nm = False
        else:
            use_extra_nm = True

        output_list = self.outer_opt(K, verbose=verbose, make_plots=self.plot_inner_opt, use_extra_nm=use_extra_nm)
        c = -1
        for info_dict in output_list:
            c += 1
            if len(output_list)==1:
                dir_str = extra_nm
            else:
                dir_str = '{}/run{}'.format(extra_nm, c)
            mse_best = self.one_loss_k(K, **info_dict,
                            final_run=final_run,
                            verbose=verbose,
                            normalize_plot=normalize_plot,
                            output_dir=os.path.join(self.output_dir, dir_str))

        if plot_surface:
            a0_list = [d['c0_acc_best'] for d in output_list]
            mse_list = [d['mse_best'] for d in output_list]
            self.loss_surface_k(K, a_list=a0_list, mse_list=mse_list, extra_nm=extra_nm, nx=nxsurface)
        return mse_best

    def one_loss_k(self, K, mse_best=None, mse_list_best=None, c0_acc_best=None, theta_best=None,
                    final_run=False, verbose=False, normalize_plot=False,
                    output_dir=None):
        if self.make_plots or final_run:
            os.makedirs(output_dir, exist_ok=True)

        if verbose or final_run:
            print('log(K):', K)
            print('MSE:', mse_best)
            print('MSE per target:', mse_list_best)
            print('Accessory concentrations:', c0_acc_best)
            print('theta:', theta_best)

        if final_run:
            if self.make_plots:
                outdir = os.path.join(output_dir, 'optimization_results')
                os.makedirs(outdir, exist_ok=True)
            ncols = min(6, self.n_targets)
            N_subs = int(np.ceil(self.n_targets / ncols))
            j = -1

            N_plot_dimers = min(10, self.n_dimers)
            if N_plot_dimers == self.n_dimers:
                dimer_inds = np.arange(N_plot_dimers)
            else:
                dimer_inds = np.random.choice(np.arange(self.n_dimers), size=N_plot_dimers, replace=False)

            f_hat_list = []
            mse_list = []

            if self.make_plots:
                for n in range(N_subs):
                    fig, axs = plt.subplots(nrows=3, ncols=ncols, figsize = [ncols*7,21], squeeze=False)
                    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
                    for cc in range(ncols):
                        j += 1
                        if j >= self.n_targets:
                            continue
                        c0_acc, theta = self.get_params(c0_acc_best, theta_best, j)
                        axs[0,cc].plot(self.input_concentration, self.f_targets[j], label='Target', color='black')
                        f_hat = self.predict(K, c0_acc, theta)
                        f_hat_list.append(f_hat)
                        axs[0,cc].plot(self.input_concentration, f_hat, '--', color=default_colors[0])
                        axs[0,cc].set_xlabel('[Input Monomer]')
                        mse_j = self.mse(f_hat, self.f_targets[j])
                        mse_list += [mse_j]
                        axs[0,cc].set_title('Fitting target-{}: MSE {}'.format(j, round(mse_j, 3)))
                        axs[0,cc].set_xscale('log')
                        axs[0,cc].legend()

                        # compute dimers
                        dimer_names = self.Knames[dimer_inds]
                        output_dimers = self.g1(c0_acc, K)[:,dimer_inds]
                        max_od = np.max(output_dimers, axis=0)

                        if normalize_plot:
                            axs[1,cc].plot(self.input_concentration, output_dimers/max_od, label=dimer_names)
                            axs[1,cc].set_title('[Output Dimers (Max-Normalized)]')
                            y = theta[dimer_inds]*max_od
                        else:
                            axs[1,cc].plot(self.input_concentration, output_dimers, label=dimer_names)
                            axs[1,cc].set_title('[Output Dimers]')
                            if self.log_errors:
                                axs[1,cc].set_ylim([-3,3])
                            else:
                                axs[1,cc].set_yscale('log')
                            y = theta[dimer_inds]

                        axs[1,cc].set_xscale('log')
                        axs[1,cc].set_xlabel('[Input Monomer]')
                        axs[1,cc].legend()

                        x = np.flipud(np.argsort(y))
                        y_sorted = y[x]
                        xseq = np.arange(len(x))
                        axs[2,cc].bar(xseq, y_sorted, color=default_colors[x])
                        axs[2,cc].set_xticks(xseq)
                        axs[2,cc].set_xticklabels(dimer_names[x])
                        if normalize_plot:
                            axs[2,cc].set_title('Normalized Dimer Weights')
                        else:
                            axs[2,cc].set_title('Dimer Weights')
                            axs[2,cc].set_yscale('log')

                    fig.suptitle('Mean overall MSE = {}'.format(round(np.mean(mse_list),3)))
                    if self.make_plots:
                        fig.savefig(os.path.join(outdir, 'plot{}.pdf'.format(n)), format='pdf')
                    plt.close()
            else:
                for j in range(self.n_targets):
                    c0_acc, theta = self.get_params(c0_acc_best, theta_best, j)
                    f_hat = self.predict(K, c0_acc, theta)
                    f_hat_list.append(f_hat)
                    mse_j = self.mse(f_hat, self.f_targets[j])
                    mse_list += [mse_j]

            # plot output dimers
            if self.make_plots:
                if self.acc_opt=='outer':
                    fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [10,10])
                    output_dimers = self.g1(c0_acc, K)
                    axs.plot(output_dimers)
                    axs.set_title('Output Dimers')
                    axs.set_yscale('log')
                    fig.savefig(os.path.join(output_dir, 'output_dimers.pdf'), format='pdf')
                    plt.close()

                # plot accessory values
                a_dict = {'Learnt': np.array(c0_acc_best).reshape(-1, self.n_acc)}
                if len(self.truth['a0']):
                    a_dict['True'] = np.array(self.truth['a0']).reshape(-1, self.n_acc)
                fig, axs = plt.subplots(nrows=1, ncols=len(a_dict), figsize = [len(a_dict)*12,10], squeeze=False)
                plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)
                c = -1
                for nm, val in a_dict.items():
                    c += 1
                    ax = axs[0,c]
                    sns.heatmap(ax=ax, data=val, cmap="viridis", vmin=self.acc_lb, vmax=self.acc_ub)
                    ax.set_xticklabels(np.arange(2,self.m+1))
                    ax.set_ylabel('Target Function Index')
                    ax.set_xlabel('Accessory Monomer Index')
                    ax.set_title('{} Accessory Monomer Concentrations (log10)'.format(nm))
                fig.savefig(os.path.join(output_dir, 'accessory_concentrations.pdf'), format='pdf')
                plt.close()

                ## plot optimal K
                mask = np.tril(np.ones(self.m),-1) # creating mask
                K_dict = {'Learnt': make_K_matrix(K, self.m)}
                if len(self.truth['K']):
                    K_dict['True'] = make_K_matrix(self.truth['K'], self.m)
                fig, axs = plt.subplots(nrows=1, ncols=len(K_dict), figsize = [len(a_dict)*10,10], squeeze=False)
                c = -1
                for nm, val in K_dict.items():
                    c += 1
                    ax = axs[0,c]
                    # plotting a triangle correlation heatmap
                    sns.heatmap(ax=ax, data=val, cmap="viridis", mask=mask, vmin=self.param_lb[0], vmax=self.param_ub[0])
                    ax.set_xticklabels(np.arange(1,self.m+1))
                    ax.set_yticklabels(np.arange(1,self.m+1))
                    ax.set_title('{} Binding Affinity Matrix (log10)'.format(nm))
                fig.savefig(os.path.join(output_dir, 'logK.pdf'), format='pdf')
                plt.close()

            # save results
            f_fit = np.array(f_hat_list)
            Linf = np.max( np.abs(f_fit - self.f_targets), axis=1 )

            out = {'MSE': mse_best,
                    'MSE-per-target': mse_list_best,
                    'Linf': Linf,
                    'a': c0_acc_best,
                    'theta': theta_best,
                    'logK': K,
                    'K': np.float_power(10, K),
                    'f_fit': f_fit,
                    'f_target': self.f_targets}

            dump_data(out, os.path.join(output_dir, 'model_info.pkl'), to_dict=False)

        return mse_best

    def try_lsq_linear(self, x, y, scale=None, fit=True):
        assert x.shape[0]==y.shape[0]
        if fit:
            try:
                foo = scipy.optimize.lsq_linear(x, y, bounds=self.lsq_bounds, method=self.lsq_linear_method)
            except:
                print('BVLS tolerance not met. Switching to TRF solver.')
                foo = scipy.optimize.lsq_linear(x, y, bounds=self.lsq_bounds, method='trf')
        else:
            fun = y.squeeze() - scale*x.squeeze()
            foo = DotDict(fun=fun, x=scale)

        return foo

    def find_best_beta(self, x, y, scale=None, fit=True, id_beta=None):
        n_betas = x.shape[1]
        if scale is None:
            scale = [None for j in range(n_betas)]
        elif len(scale)==1:
            if scale[0] == 1:
                scale = 1
            scale = [scale for j in range(n_betas)]

        if id_beta is not None:
            # assign beta vector
            beta = np.zeros(n_betas)
            foo_best = self.try_lsq_linear(x[:,id_beta].reshape(-1,1), y, scale=scale[id_beta], fit=fit)
            beta[id_beta] = foo_best.x
            foo_best.x = beta
            return foo_best


        foo_list = [self.try_lsq_linear(x[:,j].reshape(-1,1), y, scale=scale[j], fit=fit) for j in range(n_betas)]
        j_best = 0
        mse_best = np.Inf
        for j in range(n_betas):
            mse_j = np.sum(foo_list[j].fun**2)
            if mse_j <= mse_best:
                j_best = j
                mse_best = mse_j
                foo_best = foo_list[j]
        # assign beta vector
        beta = np.zeros(n_betas)
        beta[j_best] = foo_best.x
        foo_best.x = beta
        return foo_best

    def lsq_linear_wrapper(self, x, y, scale=None, one_scale=False):
        if self.single_beta:
            if self.no_rescaling:
                scale = [1]
            foo = self.find_best_beta(x, y, scale=scale, fit=not(one_scale), id_beta=self.id_dimer)
        else:
            foo = self.try_lsq_linear(x, y)
        return foo

    def simple_loss(self, dimers_all, apply_power_theta=True, error_only=False):
        if self.acc_opt=='inner' and self.w_opt=='inner' and self.one_scale and self.single_beta:
            theta_star = np.zeros((self.n_targets, dimers_all.shape[1]))
            mse_list = [0 for j in range(self.n_targets)]
            for j in range(self.n_targets):
                if self.n_scales > 0:
                    scale = c0_acc[-self.n_scales:]
                    if apply_power_theta:
                        scale = np.float_power(10, scale)
                else:
                    scale = [1]
                foo = self.lsq_linear_wrapper(x=dimers_all, y=self.f_targets[j], scale=scale, one_scale=True)
                theta_star[j] = foo.x
                mse_list[j] = np.sum(foo.fun**2) / self.n_input_samples / self.f_targets_max_sq[j]
            mse_total = np.mean(mse_list) # average over targets
            if error_only:
                return mse_total
            else:
                return theta_star, mse_total, mse_list

        elif self.acc_opt=='inner' and self.w_opt=='inner' and not(self.one_scale):
            pass
            # dimers = self.g1(c0_acc, K) # 40 x 9
            # foo = self.lsq_linear_wrapper(x=dimers, y=self.f_targets[j_target])
            # theta_star_j = foo.x
            # mse_j = np.sum(foo.fun**2) / self.n_input_samples / self.f_targets_max_sq[j_target]
            # mse_j /= self.n_targets # average over targets
            # if error_only:
            #     return mse_j
            # else:
            #     return theta_star_j, mse_j
        elif self.acc_opt=='inner' and self.w_opt=='outer':
            # dimers_all = []
            # for j in range(self.n_targets):
            #     i_low = j*(self.n_acc)
            #     i_high = i_low + (self.n_acc)
            #     dimers_j = self.g1(c0_acc[i_low:i_high], K) # 40 x 9
            #     dimers_all += [dimers_j]
            targets_all = self.f_targets.reshape(-1) #(40*n,)
            # dimers_all = np.vstack(dimers_all)
            foo = self.lsq_linear_wrapper(x=dimers_all, y=targets_all)
            theta_star = foo.x
            mse_total = np.sum(foo.fun**2) / self.n_input_samples / np.sum(self.f_targets_max_sq) / self.n_targets

            # compute listed MSEs
            mse_list = [0 for j in range(self.n_targets)]
            for j in range(self.n_targets):
                i_low = j*(self.n_input_samples)
                i_high = i_low + self.n_input_samples
                mse_list[j] = np.sum(foo.fun[i_low:i_high]**2) / self.n_input_samples / self.f_targets_max_sq[j]
            if error_only:
                return mse_total
            else:
                return theta_star, mse_total, mse_list
        elif self.acc_opt=='outer' and self.w_opt=='inner':
            # dimers = self.g1(c0_acc, K) # 40 x 9
            theta_star = np.zeros((self.n_targets, dimers_all.shape[1]))
            mse_list = [0 for j in range(self.n_targets)]
            for j in range(self.n_targets):
                foo = self.lsq_linear_wrapper(x=dimers_all, y=self.f_targets[j])
                theta_star[j] = foo.x
                mse_list[j] = np.sum(foo.fun**2) / self.n_input_samples / self.f_targets_max_sq[j]
            mse_total = np.mean(mse_list)
            if error_only:
                return mse_total
            else:
                return theta_star, mse_total, mse_list
        else:
            pass


    def inner_opt(self, c0_acc, K, j_target=None, error_only=False, apply_power_theta=True):
        '''This function computes optimal linear weights and the associated errors incurred for these optimal weights.'''
        if self.acc_opt=='inner' and self.w_opt=='inner' and self.one_scale and self.single_beta:

            dimers_all = []
            for j in range(self.n_targets):
                i_low = j*(self.n_acc)
                i_high = i_low + (self.n_acc)
                dimers_j = self.g1(c0_acc[i_low:i_high], K) # 40 x 9
                dimers_all += [dimers_j]
            # targets_all = self.f_targets.reshape(-1) #(40*n,)
            dimers_all = np.array(dimers_all)

            theta_star = np.zeros((self.n_targets, dimers_j.shape[1]))
            mse_list = [0 for j in range(self.n_targets)]
            for j in range(self.n_targets):
                if self.n_scales > 0:
                    scale = c0_acc[-self.n_scales:]
                    if apply_power_theta:
                        scale = np.float_power(10, scale)
                else:
                    scale = [1]
                foo = self.lsq_linear_wrapper(x=dimers_all[j], y=self.f_targets[j], scale=scale, one_scale=True)
                theta_star[j] = foo.x
                mse_list[j] = np.sum(foo.fun**2) / self.n_input_samples / self.f_targets_max_sq[j]
            mse_total = np.mean(mse_list) # average over targets
            if error_only:
                return mse_total
            else:
                return theta_star, mse_total, mse_list

        elif self.acc_opt=='inner' and self.w_opt=='inner' and not(self.one_scale):
            dimers = self.g1(c0_acc, K) # 40 x 9
            foo = self.lsq_linear_wrapper(x=dimers, y=self.f_targets[j_target])
            theta_star_j = foo.x
            mse_j = np.sum(foo.fun**2) / self.n_input_samples / self.f_targets_max_sq[j_target]
            mse_j /= self.n_targets # average over targets
            if error_only:
                return mse_j
            else:
                return theta_star_j, mse_j
        elif self.acc_opt=='inner' and self.w_opt=='outer':
            dimers_all = []
            for j in range(self.n_targets):
                i_low = j*(self.n_acc)
                i_high = i_low + (self.n_acc)
                dimers_j = self.g1(c0_acc[i_low:i_high], K) # 40 x 9
                dimers_all += [dimers_j]
            targets_all = self.f_targets.reshape(-1) #(40*n,)
            dimers_all = np.vstack(dimers_all)
            foo = self.lsq_linear_wrapper(x=dimers_all, y=targets_all)
            theta_star = foo.x
            mse_total = np.sum(foo.fun**2) / self.n_input_samples / np.sum(self.f_targets_max_sq) / self.n_targets

            # compute listed MSEs
            mse_list = [0 for j in range(self.n_targets)]
            for j in range(self.n_targets):
                i_low = j*(self.n_input_samples)
                i_high = i_low + self.n_input_samples
                mse_list[j] = np.sum(foo.fun[i_low:i_high]**2) / self.n_input_samples / self.f_targets_max_sq[j]
            if error_only:
                return mse_total
            else:
                return theta_star, mse_total, mse_list
        elif self.acc_opt=='outer' and self.w_opt=='inner':
            dimers = self.g1(c0_acc, K) # 40 x 9
            theta_star = np.zeros((self.n_targets, dimers.shape[1]))
            mse_list = [0 for j in range(self.n_targets)]
            for j in range(self.n_targets):
                foo = self.lsq_linear_wrapper(x=dimers, y=self.f_targets[j])
                theta_star[j] = foo.x
                mse_list[j] = np.sum(foo.fun**2) / self.n_input_samples / self.f_targets_max_sq[j]
            mse_total = np.mean(mse_list)
            if error_only:
                return mse_total
            else:
                return theta_star, mse_total, mse_list
        else:
            pass

    def set_inner_problems(self, K):
        self.inner_problem_list = []
        if self.acc_opt=='inner' and self.w_opt=='inner' and not(self.one_scale):
            for j_target in range(self.n_targets):
                f_obj = functools.partial(self.inner_opt, K=K, j_target=j_target, error_only=True)
                problem = FunctionalProblem(
                    n_var=self.n_var,
                    objs=f_obj,
                    xl=self.acc_lb_list,
                    xu=self.acc_ub_list)
                self.inner_problem_list.append(problem)

        elif self.acc_opt=='inner':
            f_obj = functools.partial(self.inner_opt, K=K, error_only=True)
            xl = [self.acc_lb]*(self.n_acc)*self.n_targets
            xu = [self.acc_ub]*(self.n_acc)*self.n_targets
            if self.w_opt=='inner' and self.one_scale and self.single_beta and not(self.no_rescaling):
                xl += [self.scale_bounds[0]]*self.n_scales
                xu += [self.scale_bounds[1]]*self.n_scales
            problem = FunctionalProblem(
                n_var=self.n_var,
                objs=f_obj,
                xl=xl,
                xu=xu)
            self.inner_problem_list.append(problem)

        elif self.acc_opt=='outer' and self.w_opt=='inner':
            f_obj = functools.partial(self.inner_opt, K=K, error_only=True)
            problem = FunctionalProblem(
                n_var=self.n_var,
                objs=f_obj,
                xl=self.acc_lb_list,
                xu=self.acc_ub_list)
            self.inner_problem_list.append(problem)
        else:
            pass

    def f_min_outer(self, problem, truth, plot_dirname, verbose=True, make_plots=True, seed=None, make_new_dir=True):
        opt_list = minimize_wrapper(
            problem,
            self.algorithm,
            self.termination,
            n_starts=self.n_starts,
            save_history=True,
            polish=self.polish,
            verbose=verbose,
            truth=truth,
            seed=seed,
            writedata=not(self.abort_early),
            plot_analyses=make_plots,
            makenewdir=make_new_dir,
            plot_dirname=plot_dirname)
        return opt_list

    def outer_opt(self, K, verbose=True, make_plots=True, use_extra_nm=True):
        seed = self.inner_opt_seed
        self.set_inner_problems(K)
        if self.acc_opt=='inner':
            if self.w_opt=='inner' and not(self.one_scale):
                opt_list = []
                for j in range(self.n_targets):
                    try:
                        truth = self.truth['a0'][j]
                    except:
                        truth = self.truth['a0']
                    problem = self.inner_problem_list[j]
                    opt_list_j = self.f_min_outer(problem, truth=truth, plot_dirname=os.path.join(self.output_dir,'inner_opt_j{}'.format(j)), verbose=verbose, make_plots=make_plots, seed=seed)
                    opt_list.append(opt_list_j)
                output_list = self.make_opt_output_list(opt_list, K)
            elif self.w_opt=='outer' or self.one_scale:
                problem = self.inner_problem_list[0]
                if use_extra_nm:
                    plot_dirname = os.path.join(self.output_dir,'inner_opt')
                else:
                    plot_dirname = self.output_dir
                opt_list = self.f_min_outer(problem, truth=self.truth['a0'], plot_dirname=plot_dirname, verbose=verbose, make_plots=make_plots, seed=seed, make_new_dir=use_extra_nm)
                output_list = self.make_opt_output_list(opt_list, K)
        elif self.acc_opt=='outer' and self.w_opt=='inner':
            problem = self.inner_problem_list[0]
            if use_extra_nm:
                plot_dirname = os.path.join(self.output_dir,'inner_opt')
            else:
                plot_dirname = self.output_dir
            opt_list = self.f_min_outer(problem, truth=self.truth['a0'], plot_dirname=plot_dirname, verbose=verbose, make_plots=make_plots, seed=seed, make_new_dir=use_extra_nm)
            output_list = self.make_opt_output_list(opt_list, K)
        else:
            pass
        return output_list

    def make_opt_output_list(self, opt_list, K):
        output_list = []
        if self.acc_opt=='inner':
            if self.w_opt=='inner' and not(self.one_scale):
                n_opts = len(opt_list[0])
                for n in range(n_opts):
                    mse_best = 0
                    mse_list_best = []
                    c0_acc_best = np.zeros((self.n_targets, self.n_acc))
                    theta_best = np.zeros((self.n_targets, self.n_dimers))
                    for j in range(self.n_targets):
                        opt = opt_list[j][n]
                        c0_acc_best_j = opt.X
                        # rerun the best run to get more details
                        theta_best_j, mse_best_j = self.inner_opt(c0_acc_best_j, K, j)
                        mse_list_best.append(mse_best_j)
                        mse_best += mse_best_j
                        c0_acc_best[j,:] = c0_acc_best_j
                        theta_best[j,:] = theta_best_j

                    info_dict = {'mse_best': mse_best,
                                'mse_list_best': mse_list_best,
                                 'c0_acc_best': c0_acc_best,
                                 'theta_best': theta_best}
                    output_list.append(info_dict)
            elif self.w_opt=='outer' or self.one_scale:
                for opt in opt_list:
                    c0_acc_best = opt.X
                    # rerun the best run to get more details
                    theta_best, mse_total_best, mse_list_best = self.inner_opt(c0_acc_best, K)
                    if self.w_opt=='inner' and self.one_scale:
                        # convert c0_acc_best to a shaped array
                        if self.n_scales > 0:
                            c0_acc_best = c0_acc_best[:-self.n_scales].reshape(self.n_targets,-1)
                        else:
                            c0_acc_best = c0_acc_best.reshape(self.n_targets,-1)
                    else:
                        # convert c0_acc_best to a shaped array
                        c0_acc_best = c0_acc_best.reshape(self.n_targets,-1)

                    info_dict = {'mse_best': mse_total_best,
                                 'mse_list_best': mse_list_best,
                                 'c0_acc_best': c0_acc_best,
                                 'theta_best': theta_best}
                    output_list.append(info_dict)
        elif self.acc_opt=='outer' and self.w_opt=='inner':
            for opt in opt_list:
                c0_acc_best = opt.X
                # rerun the best run to get more details
                theta_best, mse_total_best, mse_list_best = self.inner_opt(c0_acc_best, K)
                info_dict = {'mse_best': mse_total_best,
                             'mse_list_best': mse_list_best,
                             'c0_acc_best': c0_acc_best,
                             'theta_best': theta_best}
                output_list.append(info_dict)
        else:
            pass

        return output_list


    def set_opts(self):

        self.termination = SingleObjectiveDefaultTermination(
                            x_tol=1e-6,
                            cv_tol=0.0,
                            f_tol=1e-10,
                            nth_gen=5,
                            n_last=20,
                            n_max_gen=self.opt_settings_outer['maxiter'])

        if self.acc_opt=='inner' and self.w_opt=='outer':
            self.n_var = self.n_targets * (self.n_acc)
        elif self.acc_opt=='inner' and self.one_scale:
            if self.scale_type=="global":
                n_scales = 1
            elif self.scale_type=="per-dimer":
                n_scales = self.n_dimers
            if self.no_rescaling:
                n_scales = 0
            self.n_var = self.n_targets * (self.n_acc) + n_scales
            self.n_scales = n_scales
        else:
            self.n_var = self.n_acc

        popsize = self.opt_settings_outer['popsize']
        # self.algorithm = GA(pop_size=popsize)
        # self.algorithm = DE(CR=0.9, pop_size=popsize)
        try:
            self.algorithm = NelderMead(x0=self.c0_sorted[0])
            # self.algorithm = NelderMead() #=self.c0_sorted[0])
            # pop = Population.new("X", X) #, "F", F)
            # self.algorithm = GA(pop_size=popsize, sampling=pop, eliminate_duplicates=False)
            print('Successfully initialized Inner Opt at grid opt.')
        except:
            self.algorithm = GA(pop_size=popsize)
            print('Unable to initialize Inner Opt at grid opt. Defaulting to random init.')


    def g1(self, c0_acc, K, apply_power_K=True, apply_power_c0=True):
        # for 1d -> 1d predictions, we have each row of C0 being the same EXCEPT in its first column,
        # where we modulate the input monomor concentration over a pre-defined range.
        # Note: evaluation of g1 scales linearly with number of rows in C0...eqtk must treat each row of C0 independently.

        if apply_power_K:
            K = np.float_power(10, K)
        if apply_power_c0:
            c0_acc = np.float_power(10, c0_acc)

        C0 = self.C0.copy()
        C0[:,self.acc_monomer_ind] = c0_acc # each row should get c0_acc
        sols = eqtk.solve(c0=C0, N=self.N, K=K)

        dimers = sols[:,self.m:]

        # dimers = thresh2eps(dimers, eps=self.dimer_eps)
        if self.floor_dimers:
            dimers[dimers < 10**self.input_lb] = 10**self.input_lb

        if self.log_errors:
            dimers = np.log10(dimers)


        return dimers

    def g2(self, dimers, theta):
        f_out = dimers @ theta
        return f_out

    def predict(self, K, c0_acc, theta):
        dimers = self.g1(c0_acc, K)

        f_hat = self.g2(dimers, theta)
        return f_hat

    def predict_many(self, param_list):

        fits = []
        for p in param_list:
            K = p['K']
            c0_acc_best = p['c0_acc_best']
            theta_best = p['theta_best']

            fits_p = []
            for j in range(self.n_targets):
                c0_acc, theta = self.get_params(c0_acc_best, theta_best, j)
                f_hat = self.predict(K, c0_acc, theta)
                fits_p.append(f_hat)

            fits.append(np.array(fits_p))
        foo = np.atleast_3d(np.array(fits)).transpose(1,0,2) # N targets x N param sets x Input Discretization
        return foo

    def get_params(self, c0_acc, theta, j):
        if c0_acc.ndim==1 and theta.ndim==2:
            return c0_acc, theta[0]
        # NOTE: there may be more bugs here? check whether c0_acc is getting pulled correctly with right shape when this function is called
        if self.acc_opt=='inner':
            c0_acc = c0_acc[j]
        if self.w_opt=='inner':
            theta = theta[j]
        return c0_acc, theta

    def plot_many_fits(self, output_dir, param_list):
        os.makedirs(output_dir, exist_ok=True)

        # generate list of fits
        fits = self.predict_many(param_list)

        # plot fits
        if self.log_errors: # convert to regular units, since plot_targets plots things in log-coordinates
            f_targets = 10**self.f_targets
            fits = 10**fits

        self.plot_targets(output_dir, fits)
