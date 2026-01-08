import os, sys
import shutil
import time
import numpy as np
import pickle
import argparse
import pandas as pd
sys.path.append('../code')
from utilities import dict_combiner
from bump_inference_tests_ainner_ManyBumps import opt_wrapper
from pdb import set_trace as bp

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='../results/manybumps1switch_dev', type=str) # base directory for output
parser.add_argument('--grid_dir', default='../data/sims', type=str) # base directory for output
parser.add_argument('--target_lib_file', default='../data/voxel_averages/{}M_voxel_averages.npy', type=str) # file for reading target functions
parser.add_argument("--dev_run", default=0, type=int)
parser.add_argument("--run_all", default=0, type=int)
parser.add_argument("--id", default=0, type=int)
parser.add_argument("--dothreading", default=0, type=int)
parser.add_argument("--make_plots", default=0, type=int)
parser.add_argument("--m", default=3, type=int)
parser.add_argument("--n_random_Ks", default=10, type=int)
FLAGS = parser.parse_args()

mydict = {
    "grid_dir": [FLAGS.grid_dir],
    "nominal_base_dir": [FLAGS.base_dir],
    "target_lib_file": [FLAGS.target_lib_file],
    "target_lib_name": ["MetaClusters"], # this option reads in targets from file
    "dimer_eps": [1e-16],
    "n_switches": [1],# 2],
    "n_switch_points": [3],
    "start": ["both"], #["on", "off", "both"],
    "id_target": [i for i in range(691)],
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
    # "id_K": [i for i in range(10)],
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

def run_main(sett, dothreading, make_plots):
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

    return sett

def run_cleanup(run_dir, info_file, experiment_key, master_output_file, danger_to_read):

    time.sleep(10*np.random.rand())

    while os.path.exists(danger_to_read):
        time.sleep(10*np.random.rand())

    srf = open(danger_to_read, 'wb')
    try:
        with open(master_output_file, 'rb') as f_master:
            master = pickle.load(f_master)
    except:
        # only to initialize (first time)
        master = {}

    # read in run-specific model_info.pkl, then write it to master
    with open(info_file, 'rb') as f_run:
        #write the run to master
        master[experiment_key] = pickle.load(f_run)

    # write master to file
    with open(master_output_file, 'wb') as f_master:
        pickle.dump(master, f_master)

    # delete run-specific
    shutil.rmtree(run_dir)

    # delete safe to read file
    srf.close()
    os.remove(danger_to_read)

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
        settings = EXPERIMENT_LIST[FLAGS.id]

        for id_K in range(FLAGS.n_random_Ks):
            settings['id_K'] = id_K
            print('Running target id = ', settings['id_target'], 'with dimer id =', settings['id_dimer'])
            sett = run_main(settings, dothreading=FLAGS.dothreading, make_plots=FLAGS.make_plots)

            run_dir = sett['base_dir']
            info_file = os.path.join(run_dir, 'model_info.pkl')
            experiment_key = os.path.split(sett['base_dir'])[-1]
            danger_to_read = os.path.join(*os.path.split(sett['base_dir'])[:-1], 'master_results_is_being_modified.info')
            master_output_file = os.path.join(*os.path.split(sett['base_dir'])[:-1], 'master_results.pkl')

            # run_cleanup(run_dir, info_file, experiment_key, master_output_file, danger_to_read)
