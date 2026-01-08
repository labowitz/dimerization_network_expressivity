import os, sys
sys.path.append('../code')
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from pymoo.optimize import minimize
import scipy.optimize
from utilities import *
from pdb import set_trace as bp
from time import time

plt.rcParams.update({'font.size': 22, 'legend.fontsize': 12,
                'legend.facecolor': 'white', 'legend.framealpha': 0.8,
                'legend.loc': 'upper left', 'lines.linewidth': 4.0})

default_colors = np.array(plt.rcParams['axes.prop_cycle'].by_key()['color'])

def thresh2eps(x, eps=1e-10):
    x[np.abs(x)<=eps] = eps
    return x

def minimize_wrapper(problem, algorithm, termination, seed=None, save_history=True,
                    verbose=True,
                    n_starts=2,
                    plot_analyses=True,
                    plot_dirname='optplots',
                    polish=False,
                    report_times=False,
                    makenewdir=True,
                    writedata=True,
                    truth = []):

    t0 = time()
    opt_list = []
    for n in range(n_starts):
        opt = minimize(
            problem,
            algorithm,
            termination,
            seed=seed,
            save_history=save_history,
            verbose=verbose)

        # optionally run a local optimizer to polish the optimization
        if polish:
            result = scipy.optimize.minimize(problem.objs[0],
                              opt.X,
                              method='trust-constr',
                              bounds=np.array([problem.xl, problem.xu]).T)
            opt.X = result.x
            opt.F = result.fun

        opt_list += [opt]

    if report_times:
        print('Minimization took {} seconds'.format(time() - t0))

    t0 = time()
    if makenewdir:
        new_plot_dirname = make_new_dir(plot_dirname)
    else:
        new_plot_dirname = plot_dirname
    ap = AnalyzePymoo(opt_list, truth=truth)
    if writedata:
        ap.write_info(new_plot_dirname)
    if plot_analyses:
        ap.make_plots(new_plot_dirname)
    if report_times:
        print('Analysis took {} seconds'.format(time() - t0))

    return opt_list


def plot_targets(output_dir, inputs, targets, fits=[], n_plots=4, input_bounds=[1e-3,1e3], output_bounds=[1e-3,1e3], label_fits=False, fit_label='fits'):
    os.makedirs(output_dir, exist_ok=True)
    n_targets = len(targets)
    nrows = min(n_plots, int(np.ceil(n_targets/n_plots)) )
    ncols = min(n_plots, int(np.ceil(n_targets/nrows)) )
    N_subs = int(np.ceil(n_targets / (nrows*ncols)))
    cc = -1
    for n in range(N_subs):
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize = [ncols*5,nrows*3], squeeze=False, sharex=True, sharey=True)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
        for i in range(nrows):
            for j in range(ncols):
                cc += 1
                if cc < n_targets:
                    axs[i,j].plot(inputs, targets[cc], label='target', color='black', linewidth=4)
                    try:
                        if label_fits:
                            axs[i,j].plot(inputs, fits[cc].T, '--', linewidth=2, label=fit_label)
                        else:
                            axs[i,j].plot(inputs, fits[cc].T, '--', color=default_colors[0], linewidth=2)
                    except:
                        pass
                    # axs[i,j].set_yscale('log')
                    # axs[i,j].set_xscale('log')
                    axs[i,j].set(xlim=input_bounds, ylim=output_bounds, xscale='log', yscale='log')

                    axs[i,j].legend()
                    axs[i,j].set_title('Target {}'.format(cc))
                    axs[-1,j].set_xlabel('[Input Monomer]')

        fig.savefig(os.path.join(output_dir,'target_plot{}.pdf'.format(n)), format='pdf')
        plt.close()

def plot_targets2(output_dir, inputs, targets, fits, jacob_fits, m_list, n_plots=4, input_bounds=[1e-3,1e3], output_bounds=[1e-3,1e3], label_fits=False, fit_label='fits', nm='mse', id_list=None):

    '''fits: (n_targets, n_inputs, n_runs, n_m_vals)'''
    linestyles = ['dotted', 'dashed', 'dashdot', 'dotted']

    if id_list is None:
        id_list = np.arange(fits.shape[0])
    else:
        fits = fits[id_list]
        targets = targets[id_list]
        jacob_fits = jacob_fits[id_list]


    nT, nI, nR, nM = fits.shape
    os.makedirs(output_dir, exist_ok=True)
    n_targets = len(targets)
    nrows = min(n_plots, int(np.ceil(n_targets/n_plots)) )
    ncols = min(n_plots, int(np.ceil(n_targets/nrows)) )
    N_subs = int(np.ceil(n_targets / (nrows*ncols)))
    cc = -1
    for n in range(N_subs):
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize = [ncols*5,nrows*3], squeeze=False, sharex=True, sharey=True)
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
        for i in range(nrows):
            for j in range(ncols):
                cc += 1
                if cc < n_targets:
                    axs[i,j].plot(inputs, targets[cc], label='target', color='black', linewidth=4)

                    # extract relevant fits
                    foo = fits[cc]
                    for tt in range(nM):
                        for ss in range(nR):
                            if ss==0:
                                label = 'M={}'.format(m_list[tt])
                                axs[i,j].plot(inputs, jacob_fits[cc,:,tt].T, 'x', color=default_colors[tt], linewidth=2, label=label+' grid')
                            else:
                                label = None
                            try:
                                axs[i,j].plot(inputs, fits[cc,:,ss,tt].T, color=default_colors[tt], linewidth=2, label=label, linestyle=linestyles[tt])
                            except:
                                pass

                    # axs[i,j].set_yscale('log')
                    # axs[i,j].set_xscale('log')
                    axs[i,j].set(xlim=input_bounds, ylim=output_bounds, xscale='log', yscale='log')
                    if i==(nrows-1) and j==(ncols-1):
                        axs[i,j].legend()
                    axs[i,j].set_title('Target {}'.format(id_list[cc]))
                    axs[-1,j].set_xlabel('[Input Monomer]')

        fig.savefig(os.path.join(output_dir,'target_{}_plot{}.pdf'.format(nm,n)), format='pdf')
        plt.close()

def plot_avg_err(output_dir, avg_err, jacob_avg_err, m_list, std=None, nm='mse', caption='Average MSE per target'):
    '''fits: (n_targets, n_inputs, n_runs, n_m_vals)'''
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = [10,10])
    ax.plot(m_list, avg_err, '-o', label='optimization')
    ax.plot(m_list, jacob_avg_err, '-o', label='grid')
    if std is not None:
        ax.errorbar(m_list, avg_err, yerr=std)
    ax.legend()
    ax.set_xlabel('Network size')
    ax.set_ylabel(caption)
    ax.set_title(caption)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    fig.savefig(os.path.join(output_dir,'avg_{}.pdf'.format(nm)), format='pdf')
    plt.close()

def plot_avg_err2(output_fname, mean, std=None, nm='mse', caption='Average MSE per target'):
    '''fits: (n_targets, n_inputs, n_runs, n_m_vals)'''
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = [12,12])
    ax.plot(mean, '-o')
    if std is not None:
        ax.errorbar(mean.index, mean, yerr=std)
    ax.set_xlabel('Network size')
    ax.set_title(caption)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    fig.savefig(output_fname, format='pdf')
    plt.close()

def plot_boxes(output_fname, df, xname, yname):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = [12,12])
    sns.boxplot(ax=ax, x=xname, y=yname, data=df, showfliers = False, color='gray')
    sns.stripplot(ax=ax, x=xname, y=yname, data=df, s=20, marker='D', color='r', alpha=0.8)
    ax.set_xlabel('Network size')
    ax.set_ylim([0, 1])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
    fig.savefig(output_fname, format='pdf')
    plt.close()

def plot_K_convergence(output_fname, df, xname, yname, hue):
    # make mean plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = [12,12])
    sns.lineplot(ax=ax, data=df.groupby(['m']).expanding().mean().reset_index(), x='level_1', y=yname, hue=hue)
    fig.savefig(output_fname+'_mean.pdf', format='pdf')
    plt.close()

    # make std plot
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize = [12,12])
    sns.lineplot(ax=ax, data=df.groupby(['m']).expanding().std().reset_index(), x='level_1', y=yname, hue=hue)
    fig.savefig(output_fname+'_std.pdf', format='pdf')
    plt.close()


def plot_target_err(output_dir, opt_err, jacob_err, m_list, nm='mse', caption='MSE per target'):
    N_targets = jacob_err.shape[0]
    os.makedirs(output_dir, exist_ok=True)
    for i in range(len(m_list)):
        m = m_list[i]
        print('Plotting target error for m=', m)

        print('Optimization results unavailable for IDs:',np.where(np.isnan(opt_err[:,0,i])))

        fig, ax = plt.subplots()
        inds = np.argsort(jacob_err[:,i])
        opt_diff = opt_err[:,0,i] - jacob_err[:,i]
        i_bad = np.nanargmax(opt_diff)
        print('Worst offender is id', i_bad, 'with difference', opt_diff[i_bad])
        print('Opt MSE:', opt_err[i_bad,0,i])
        print('Grid MSE:', jacob_err[i_bad,i])

        # inds = (opt_err[:,0,i] - jacob_err[:,i] > 1e-2)
        # N_targets = np.sum(inds)
        # announce id's for which Optimization was WORSE than its grid-based initialization
        print('Target IDs for which optimization was WORSE than its grid-based initialization by more than 1e-2...', np.where(opt_err[:,0,i] - jacob_err[:,i] > 1e-2))

        markerline1, stemlines1, baseline1 = ax.stem(np.arange(N_targets), jacob_err[inds,i], markerfmt='D', label="Grid")
        baseline1.set_linewidth(0)
        stemlines1.set_linewidth(0.2)
        markerline1.set_markerfacecolor('none')

        ax.scatter(np.arange(N_targets), opt_err[inds,0,i], marker='D', color='orange', label="Optimization")

        ax.set_ylabel(caption)
        ax.set_yscale('log')
        ax.set_title(caption + ' m = ' + str(m))
        ax.legend()
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
        fig.savefig(os.path.join(output_dir,'stem_{}_{}.pdf'.format(nm,m)), format='pdf')
        plt.close()

        # plot ECDF
        fig, ax = plt.subplots()
        sns.ecdfplot(jacob_err[:,i], label='Grid (N={})'.format(jacob_err.shape[0]))
        sns.ecdfplot(opt_err[:,0,i], label='Optimization (N={})'.format(np.sum(~np.isnan(opt_err[:,0,i]))))
        ax.legend()
        ax.set_xscale('log')
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=0.4)
        fig.savefig(os.path.join(output_dir,'ecdf_{}_{}.pdf'.format(nm,m)), format='pdf')
        plt.close()



class AnalyzePymoo:
    def __init__(self, opt_list,
                    xnames = None,
                    truth = [],
                    bounds = [-6,6],
                    **kwargs):

        self.opt_list = opt_list
        self.xnames = xnames
        self.bounds = bounds

        self.truth = truth

        self.extract_info()

    def extract_info(self):

        # get opt sequence
        self.X = []
        self.F = []
        self.Xall = []
        self.Fall = []
        self.n_params = self.opt_list[0].opt.get("X").shape[1]

        for n in range(len(self.opt_list)):
            history = self.opt_list[n].history
            self.X += [np.vstack([e.opt.get("X") for e in history])]
            self.F += [np.array([e.opt[0].F for e in history])]

            self.Xall += [np.vstack([e.pop.get("X") for e in history])]
            self.Fall += [np.vstack([e.pop.get("F") for e in history])]

        # get index for iterations
        self.n_evals = np.array([e.evaluator.n_eval for e in history]) # only need 1, and they are identical

        # convert to np array (1st coord indexes the independent optimization).
        self.X = np.array(self.X)
        self.F = np.array(self.F)

        self.Xall = np.array(self.Xall) # n_opt_runs X n_evals X n_params
        self.Fall = np.array(self.Fall) # n_opt_runs X n_evals X n_objs (=1)

        if self.X.ndim==2:
            self.X = np.expand_dims(self.X, 0)
            self.F = np.expand_dims(self.F, 0)

        if self.xnames is None:
            self.xnames = ['P{}'.format(c) for c in range(self.X.shape[-1])]

        # order all evals in terms of F
        inds = np.argsort(self.Fall.reshape(-1))
        self.Fall_ordered = self.Fall.reshape(-1)[inds]
        self.Xall_ordered = self.Xall.reshape(-1,self.Xall.shape[-1])[inds]

        self.Fmin = np.min(self.Fall_ordered)
        self.Fmax = np.max(self.Fall_ordered)

    def write_info(self, writedir):
        os.makedirs(writedir, exist_ok=True)
        dump = {'X': self.X,
                'F': self.F,
                # 'Xall': self.Xall,
                # 'Fall': self.Fall,
                'Xall_ordered': self.Xall_ordered,
                'Fall_ordered': self.Fall_ordered,
                'n_evals': self.n_evals,
                'truth': self.truth}
        dump_data(dump, os.path.join(writedir, 'opt_info.pkl'))


    def make_plots(self, plotdir, percentile_list=[0, 1, 10, 50, 100]):
        os.makedirs(plotdir, exist_ok=True)

        try:
            # write monitoring data to file
            self.write_info(plotdir)

            # compare across optimizations
            self.plot_loss(plotdir)

            self.compare_params(plotdir)

            # plot robustness in parameter space
            self.plot_robustness(plotdir, percentile_list)

            # plot per-optimization:
            for n in range(len(self.opt_list)):
                pdir = os.path.join(plotdir, 'run{}'.format(n))
                os.makedirs(pdir, exist_ok=True)
                self.plot_params(self.X[n], pdir)
        except:
            print('Failed to make plots')


    def plot_robustness(self, plotdir, percentile_list):

        #$ plot CDF
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = [8,8])
        sns.ecdfplot(data=self.Fall.T.squeeze(), ax=ax)

        ax.set_title('Empirical CDF of Network Expressivity')
        ax.set_ylabel('Proportion')
        ax.set_xlabel('Expressivity (target misfits)')

        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)
        fig.savefig(os.path.join(plotdir, 'F_cdf.pdf'), format='pdf')
        ax.set_xscale('log')
        fig.savefig(os.path.join(plotdir, 'F_cdf_log.pdf'), format='pdf')
        plt.close()

        ## plot bivariate scatter plots of parameter choices at different thresholds of quality
        # thresh_list = np.percentile(self.Fall_ordered, percentile_list)
        for j in range(len(percentile_list)-1):
            f_low, f_high = self.get_thresh_F(p_low=percentile_list[j], p_high=percentile_list[j+1])
            df = pd.DataFrame()
            for k in range(self.Fall.shape[0]):
                Xset_k = self.Xall[k,((self.Fall[k]<=f_high) & (self.Fall[k]>=f_low)).squeeze()]
                df_k = pd.DataFrame(Xset_k, columns=self.xnames)
                df_k['run'] = k
                df = pd.concat([df,df_k])
            df.reset_index(drop=True, inplace=True)
            try:
                fig = sns.pairplot(data=df, corner=True, hue='run')
                fig.set(xlim=self.bounds, ylim=self.bounds)
                fig.savefig(os.path.join(plotdir, 'X_bivariate_thresh{}.pdf'.format(percentile_list[j])), format='pdf')
                plt.close()
            except:
                print('UNABLE TO PRINT BIVARIATE PLOT. WHAT IS WRONG? IDK.')

    def sample_X_from_percentile(self, p_high=100, p_low=0, n=2):
        thresh_tup = np.percentile(self.Fall_ordered, [p_low, p_high])
        Xset = self.Xall_ordered[(self.Fall_ordered<=thresh_tup[1]) & (self.Fall_ordered>=thresh_tup[0])]

        Nmax = Xset.shape[0]
        N = min(Nmax, n)
        X = Xset[np.random.choice(Nmax, size=N, replace=False)]
        return X

    def get_thresh_F(self, p_high=100, p_low=0):
        f_low = self.Fmin + (p_low/100)*(self.Fmax-self.Fmin)
        f_high = self.Fmin + (p_high/100)*(self.Fmax-self.Fmin)
        return f_low, f_high

    def sample_X_from_grid(self, p_high=100, p_low=0, n=2):
        f_low, f_high = self.get_thresh_F(p_high, p_low)
        Xset = self.Xall_ordered[(self.Fall_ordered<=f_high) & (self.Fall_ordered>=f_low)]

        Nmax = Xset.shape[0]
        N = min(Nmax, n)
        X = Xset[np.random.choice(Nmax, size=N, replace=False)]
        return X

    def compare_params(self, plotdir, nm='a0'):

        N_plot_params = min(10, self.n_params)
        if N_plot_params == self.n_params:
            param_inds = np.arange(N_plot_params)
        else:
            param_inds = np.random.choice(np.arange(self.n_params), size=N_plot_params, replace=False)
        fig, axs = plt.subplots(nrows=1, ncols=N_plot_params, figsize = [N_plot_params*10,8])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)

        for n in range(len(self.opt_list)):
            for c in range(N_plot_params):
                j = param_inds[c]
                axs[c].plot(self.n_evals, self.X[n,:,j], label='Run {}'.format(n))
                axs[c].set_title("Parameter {}".format(j))
                axs[c].set_xlabel("Iters")

        for c in range(N_plot_params):
            j = param_inds[c]
            if len(self.truth):
                axs[c].axhline(y=self.truth[j], color='black', linestyle='--', label='True')
            axs[c].legend()

        # axs.set_title("Convergence")
        # axs.plot(self.n_evals, X, label=self.xnames)
        # axs.legend()
        fig.savefig(os.path.join(plotdir, 'X_convergence_comparison.pdf'), format='pdf')
        plt.close()

    def plot_loss(self, plotdir):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)

        for n in range(len(self.opt_list)):
            axs.plot(self.n_evals, self.F[n], "--", label="Run {}".format(n))

        axs.legend()
        axs.set_title("Convergence")
        fig.savefig(os.path.join(plotdir, 'fval_convergence.pdf'), format='pdf')
        plt.close()

    def plot_params(self, X, plotdir):
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)

        axs.set_title("Convergence")
        axs.plot(self.n_evals, X, label=self.xnames)

        if len(self.truth):
            for c in range(self.X.shape[-1]):
                axs.axhline(y=self.truth[c], color=default_colors[c], linestyle='--', label='True')

        axs.legend()
        fig.savefig(os.path.join(plotdir, 'X_convergence.pdf'), format='pdf')
        plt.close()


def analyze_convergence(inputdir):

    dirlist = ['inner_opt_{}'.format(c) for c in range(10)]
    X = []
    F = []
    Fstd = []
    Xstd = []
    for dir in dirlist:
        res = load_data(os.path.join(inputdir, dir, "opt_info.pkl"))
        X += [res['X']]
        F += [res['F']]
        Fstd += [np.std(res['F'],axis=0)]
        Xstd += [np.sum(np.std(res['X'],axis=0), axis=1)]
    n_evals = res['n_evals']

    X = np.vstack(X)
    F = np.vstack(F).squeeze()
    Fstd = np.hstack(Fstd).T
    Xstd = np.vstack(Xstd)

    # plot F optimization variancex
    df = pd.DataFrame(Fstd.T)
    df['iteration'] = n_evals
    df = df.melt(id_vars='iteration')
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
    sns.lineplot(ax=axs, data=df, x='iteration', y='value')
    fig.savefig(os.path.join(inputdir, 'Fstd_convergence.pdf'), format='pdf')
    plt.close()

    # plot X optimization variancex
    df = pd.DataFrame(Xstd.T)
    df['iteration'] = n_evals
    df = df.melt(id_vars='iteration')
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
    sns.lineplot(ax=axs, data=df, x='iteration', y='value')
    fig.savefig(os.path.join(inputdir, 'Xstd_convergence.pdf'), format='pdf')
    plt.close()

    # plot F optimization
    df = pd.DataFrame(F.T)
    df['iteration'] = n_evals
    df = df.melt(id_vars='iteration')
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)
    sns.lineplot(ax=axs, data=df, x='iteration', y='value')
    fig.savefig(os.path.join(inputdir, 'F_convergence.pdf'), format='pdf')
    plt.close()

    # plot X optimization
    dfX = pd.DataFrame()
    for p in range(X.shape[-1]):
        pname = 'A{}'.format(p+1)
        for iter in range(X.shape[1]):
            for rn in range(X.shape[0]):
                foo = pd.DataFrame([{'iteration':iter, 'rn': rn, 'parameter': pname, 'value': X[rn,iter,p]}])
                dfX = pd.concatenate(dfX, foo)
    dfX = dfX.reset_index()
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize = [14,8])
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.6, hspace=0.6)
    sns.lineplot(ax=axs, data=dfX, x='iteration', y='value', hue='parameter', ci='sd')
    fig.savefig(os.path.join(inputdir, 'X_convergence.pdf'), format='pdf')
    plt.close()
