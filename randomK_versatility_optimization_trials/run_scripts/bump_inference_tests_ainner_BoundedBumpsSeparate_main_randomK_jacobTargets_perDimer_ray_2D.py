import os, sys
import shutil
import time
import numpy as np
# from multiprocessing import Pool
import ray
from ray.util.multiprocessing import Pool
import pickle
import argparse
import pandas as pd
sys.path.append('../code')
from utilities import dict_combiner
from bump_inference_tests_ainner_ManyBumps import opt_wrapper
from opt_utils import plot_avg_err2, plot_boxes, plot_K_convergence
from pdb import set_trace as bp


# Plotting
import itertools
import pathlib
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.ticker as mticker
from matplotlib import rcParams
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# Plotting settings
rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.size":12,
    "axes.titlesize":12,
    "axes.labelsize":12,
    "xtick.labelsize":12,
    "ytick.labelsize":12,
    "savefig.dpi": 900,
    'figure.figsize': [6.0, 4.0],
    'figure.dpi': 72.0,
    'pdf.fonttype':42,
    'ps.fonttype':42,
})




parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='../results/manybumps1switch_dev2', type=str) # base directory for output
parser.add_argument('--grid_dir', default='../data/sims', type=str) # base directory for output
parser.add_argument('--target_lib_file', default='../data/voxel_averages/10M_voxel_averages.npy', type=str) # file for reading target functions
parser.add_argument("--dev_run", default=0, type=int)
parser.add_argument("--run_all", default=0, type=int)
parser.add_argument("--id", default=0, type=int)
parser.add_argument("--dothreading", default=0, type=int)
parser.add_argument("--make_plots", default=0, type=int)
parser.add_argument("--m", default=3, type=int)
parser.add_argument("--n_random_Ks", default=1, type=int)
parser.add_argument("--frac_targets", default=1, type=float)
FLAGS = parser.parse_args()

id_target = ['AND', 'NAND', 'OR', 'XOR', 'NOR', 'XNOR', 'ANOTB', 'AeqB', 'AneqB',
       'DoubleBandpass', 'ratio']

mydict = {
    "floor_dimers": [True],
    "dim_input": [2],
    "n_input_samples": [12], # used 30 for 1-D, using 12 x 12 for 2-D
    "grid_dir": [FLAGS.grid_dir],
    "nominal_base_dir": [FLAGS.base_dir],
    "target_lib_file": ['../data/2d/custom_2D_targets.npy'],
    "target_lib_names_file": ['../data/2d/custom_2D_target_names.npy'],
    "target_lib_name": ["2d_jacob"], # this option reads in targets from file
    "dimer_eps": [1e-16],
    "n_switches": [1],# 2],
    "n_switch_points": [3],
    "start": ["both"], #["on", "off", "both"],
    "id_target": id_target,
    "id_dimer": [i for i in range(int(FLAGS.m * (FLAGS.m + 1) / 2))],
    "m": [FLAGS.m],
    "acc_opt": ["inner"],
    "w_opt": ["inner"],
    "single_beta": [1],
    "scale_type": ["per-dimer"], #"per-target"],
    "plot_inner_opt": [0],
    "make_plots": [0],
    "maxiter_O": [20],
    "popsize_O": [100],
    "polish_O": [0],
    "maxiter_K": [1],
    "popsize_K": [1],
    "polish_K": [0],
    "nstarts_K": [1],
    "randomizeK": [True],
    "abort_early": [True],
    "id_K": [i for i in range(FLAGS.n_random_Ks)],
    "inner_opt_seed": [None]
}

if FLAGS.dev_run:
    # mydict['m'] = [3]
    mydict['maxiter_O'] = [2]
    mydict['maxiter_K'] = [1]
    mydict['popsize_O'] = [2]
    mydict['popsize_K'] = [1]
    mydict['polish_K'] = [0]
    mydict['polish_O'] = [0]


EXPERIMENT_LIST = dict_combiner(mydict)

MAX_ITERS = 2e6

def namemaker(x):

    foo = [x[k] for k in ['m', 'id_target', 'id_K', 'id_dimer']]
    dirname = 'm-{}_targetID-{}_KID-{}_dimerID-{}'.format(*foo)

    goo = [x[k] for k in ['maxiter_O', 'popsize_O', 'polish_O', 'maxiter_K', 'popsize_K', 'polish_K']]
    fname = 'maxiterO-{}_popsizeO-{}_polishO-{}_maxiterK-{}_popsizeK-{}_polishK-{}'.format(*goo)

    nm = os.path.join(fname, dirname)
    return nm

def plotter(master_file, remove_targets=False):
    # read in CSVs if possible, else create them from master file
    base_dir = os.path.split(master_file)[0]
    df_output_file = os.path.join(base_dir, 'summary.csv')
    df_best_output_file = os.path.join(base_dir, 'summary_bestDimerID.csv')
    try:
        #read in summary CSVs
        df = pd.read_csv(df_output_file)
        # df_best = pd.read_csv(df_best_output_file)

        ## remove 90% of targets for each m
        if remove_targets:
            frac_targets = 0.1
            for m in df.m.unique():
                n_targets = df.targetID[df.m==m].nunique()
                id_target = np.random.choice(n_targets, size=int(n_targets*frac_targets), replace=False)
                idx_drop = df[(df.m==m) & ~df.targetID.isin(id_target)].index
                df.drop(idx_drop, inplace=True)
    except:
        # make CSV's from master file
        with open(master_file, 'rb') as f:
            x = pickle.load(f)

        keep_keys = ['m', 'targetID', 'KID', 'dimerID']
        new_dict = {}
        df = pd.DataFrame()
        for key, value in x.items():
            atts = {}
            for g in key.split('_'):
                fooval = g.split('-')[1]
                try:
                    fooval = float(fooval)
                except:
                    pass
                atts[g.split('-')[0]] = fooval
            # atts = {g.split('-')[0]: float(g.split('-')[1]) for g in key.split('_')}

            foo = {k: atts[k] for k in keep_keys}
            foo['Linf'] = float(value['Linf'])
            df = pd.concat( [df, pd.DataFrame.from_dict([foo]) ])

        df['goodenough'] = df.Linf <= 1
        df.to_csv(df_output_file)

    df_best = pd.DataFrame()
    for m in df.m.unique():
        m_df = df.loc[df.m==m]
        for kID in m_df.KID.unique():
            m_df_kID = m_df.loc[m_df.KID==kID]
            # this is now a list of dimerIDs x target IDs
            for targetID in m_df_kID.targetID.unique():
                m_df_kID_targetID = m_df_kID.loc[m_df_kID.targetID==targetID]
                # this is now a list of dimerIDs
                Linf = m_df_kID_targetID.Linf.min()
                foo = {'m': m, 'KID': kID, 'targetID': targetID, 'Linf': Linf}
                df_best = pd.concat([df_best, pd.DataFrame.from_dict([foo])])
    df_best['goodenough'] = df_best.Linf <= 1
    # df_best.to_csv(df_best_output_file)

    # make plots
    # for each m, what fraction of expressible targets can be achieved using a random K, a tuned a, and choice of output dimer?
    df1 = df_best.groupby(['m','KID']).mean().reset_index()
    plot_fname = os.path.join(base_dir, 'plot_bestDimer.pdf')
    # plot_avg_err2(plot_fname, mean=df1.groupby(['m']).mean().goodenough, std=df1.groupby(['m']).std().goodenough, nm='Linf1', caption='Fraction of targets fit with Linf < 1.0')
    plot_boxes(plot_fname, df1, xname='m', yname='goodenough')
    plot_fname = os.path.join(base_dir, 'plot_bestDimer_convergence')
    try:
        plot_K_convergence(plot_fname, df1, xname='kID', yname='goodenough', hue='m')
    except:
        pass

    # for each m, what fraction of expressible targets can be achieved using a random K, a random output dimer, and a tuned a
    df2 = df.groupby(['m','KID', 'dimerID']).mean().reset_index()
    plot_fname = os.path.join(base_dir, 'plot_eachDimerSeparate.pdf')
    # plot_avg_err2(plot_fname, mean=df2.groupby(['m']).mean().goodenough, std=df2.groupby(['m']).std().goodenough, nm='Linf1', caption='Fraction of targets fit with Linf < 1.0')
    plot_boxes(plot_fname, df2, xname='m', yname='goodenough')
    plot_fname = os.path.join(base_dir, 'plot_eachDimerSeparate_convergence')
    try:
        plot_K_convergence(plot_fname, df2, xname='kID', yname='goodenough', hue='m')
    except:
        pass

def plotmaker(out_dir, dimer_of_interest):

    input_lb = -3
    input_ub = 3

    with open(os.path.join(out_dir,'model_info.pkl'), 'rb') as f:
        x = pickle.load(f)

    t = int(np.sqrt(x['f_fit'].shape[1]))
    f_target = x['f_target'] #.reshape(t,t)
    f_fit = x['f_fit'] #.reshape(t,t)
    f_dict = {'target': f_target, 'fit': f_fit}

    for nm, foo in f_dict.items():
        fig, axs = plt.subplots(nrows=1,ncols=1,figsize=(4.3,4))
        cmap = cm.get_cmap('viridis')
        x_points = np.logspace(input_lb, input_ub, t , endpoint=True)
        y = 10**foo.reshape((t,t)).T
        matrix = axs.pcolormesh(x_points, x_points, y,\
                                 cmap = cmap,norm=mpl.colors.LogNorm(vmin=10**input_lb,vmax=10**input_ub),\
                                 shading = 'nearest')
        print(np.max(y), np.min(y))

        axs.xaxis.set_ticks_position("bottom")
        axs.set_xscale('log')
        axs.set_yscale('log')
        axs.set_xlabel(f'Input M1 (unitless)')
        axs.set_ylabel(f'Input M2 (unitless)')
        axs.set_xlim([10**input_lb,10**input_ub])
        axs.set_ylim([10**input_lb,10**input_ub])
        axs.xaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))
        axs.yaxis.set_minor_locator(mticker.LogLocator(numticks=999, subs="auto"))

        # _ = axs[row,col].set_title(make_nXn_species_names(m=m)[m+dimer_of_interest])

        # Add colorbar
        divider = make_axes_locatable(axs)
        cax1 = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(matrix,cax=cax1,label='Log (Concentration) (unitless)')

        # Adjust subplot aspect ratio
        plt.subplots_adjust(left=0.1,
                            bottom=0.1,
                            right=0.9,
                            top=0.9,
                            wspace=0.7,
                            hspace=0.5)

        dimers_label = ''
        filename = f'responses{dimers_label}_{nm}'
        plt.savefig(str(pathlib.Path(out_dir,filename+'.pdf')),bbox_inches='tight',transparent=True)
        fig.patch.set_facecolor('white')
        plt.savefig(str(pathlib.Path(out_dir,filename+'.png')),dpi=900,bbox_inches='tight')
        plt.close()

def run_cleanup(output_dir, master_file):
    os.makedirs(output_dir, exist_ok=True)

    print('Beginning Cleanup...')

    if not os.path.exists(output_dir):
        print(output_dir, 'NOT YET CREATED. SKIPPING RUN CLEANUP.')
        return

    master = {}

    print('Starting with master of length', len(master))

    # look for completed runs, then consolidate their output and delete the original run data.
    for run_dir in os.listdir(output_dir):
        info_file = os.path.join(output_dir, run_dir, 'model_info.pkl')
        experiment_key = os.path.split(run_dir)[-1]
        try:
            with open(info_file, 'rb') as f:
                # read experiment info
                model_info = pickle.load(f)

                #save experiment info to master dict
                master[experiment_key] = model_info
        except:
            pass

    # write master to file
    with open(master_file, 'wb') as f_master:
        pickle.dump(master, f_master)

    print('Finishing with master of length', len(master))

    print('Done Cleanup...')
    return


def run_main(sett, dothreading=False, make_plots=False):
    t0 = time.time()
    sett['base_dir'] = os.path.join(sett['nominal_base_dir'], namemaker(sett))

    K_names = ['maxiter_K', 'popsize_K', 'polish_K', 'nstarts_K']
    sett_K = {k.split('_')[0]: sett[k] for k in K_names}
    sett_K['dothreading'] = dothreading
    sett_K['make_plots'] = make_plots

    tune_names = ['maxiter_O', 'popsize_O', 'polish_O']
    sett['opt_settings_outer'] = {k.split('_')[0]: sett[k] for k in tune_names}

    n_iters = sett_K['maxiter']*sett_K['popsize']*sett['opt_settings_outer']['maxiter']*sett['opt_settings_outer']['popsize']
    if n_iters <= MAX_ITERS:
        opt_wrapper(sett_K, tune_dict=sett)

    print('Main run took', (time.time()-t0)/60, 'minutes')

    plotmaker(sett['base_dir'], sett['id_dimer'])
    return

def run_wrapper(id):
    if id == 0:
        t0 = time.time()
        output_dir = os.path.join(FLAGS.base_dir, 'maxiterO-2_popsizeO-2_polishO-0_maxiterK-1_popsizeK-1_polishK-0')
        master_file = os.path.join(output_dir, 'master_file.pkl')
        while time.time() - t0 < 24*60*60:
            run_cleanup(output_dir, master_file)
            time.sleep(10)
    else:
        run_main(EXPERIMENT_LIST[id])

    return


if __name__ == "__main__":
    print('{} total experiments available'.format(len(EXPERIMENT_LIST)))
    if FLAGS.run_all:
        print('Running all experiments')
        df = pd.DataFrame()
        for sett in EXPERIMENT_LIST:
            print('Running id = ', sett['id_target'], 'of', len(EXPERIMENT_LIST))

            try:
                run_main(sett, dothreading=FLAGS.dothreading, make_plots=FLAGS.make_plots)
                status = "FINISHED"
            except:
                status = "FAILED"
                # print('The following setting failed:')
                # print(sett)
            sett["status"] = status
            sett_df = pd.DataFrame([sett])
            df = pd.concat([df, sett_df])

            print(df.sort_values(by="status"))
    else:
        try:
            output_dir = os.path.join(FLAGS.base_dir, 'maxiterO-2_popsizeO-2_polishO-0_maxiterK-1_popsizeK-1_polishK-0')
            master_file = os.path.join(output_dir, 'master_file.pkl')
            plotter(master_file)
        except:
            pass


        output_dir = os.path.join(FLAGS.base_dir, 'maxiterO-20_popsizeO-100_polishO-0_maxiterK-1_popsizeK-1_polishK-0')
        master_file = os.path.join(output_dir, 'master_file.pkl')
        if os.path.exists(master_file):
            plotter(master_file)

        t0 = time.time()
        pool = Pool(os.cpu_count()-1) # use 1 fewer cores than available so that it is still easy to use bash on the machine.
        pool.map(run_main, EXPERIMENT_LIST)
        print('All jobs ran in a total of', (time.time() - t0)/60/60, 'hours')

        print('Run finished! Running cleanup now...')
        output_dir = os.path.join(FLAGS.base_dir, 'maxiterO-2_popsizeO-2_polishO-0_maxiterK-1_popsizeK-1_polishK-0')
        master_file = os.path.join(output_dir, 'master_file.pkl')
        run_cleanup(output_dir, master_file)

        try:
            plotter(master_file)
        except:
            pass

        output_dir = os.path.join(FLAGS.base_dir, 'maxiterO-20_popsizeO-100_polishO-0_maxiterK-1_popsizeK-1_polishK-0')
        master_file = os.path.join(output_dir, 'master_file.pkl')
        run_cleanup(output_dir, master_file)

        try:
            plotter(master_file)
        except:
            pass
