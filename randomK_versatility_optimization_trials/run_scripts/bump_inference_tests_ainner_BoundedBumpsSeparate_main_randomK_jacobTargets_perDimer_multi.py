import os, sys
import shutil
import time
import numpy as np
from multiprocessing import Pool
import pickle
import argparse
import pandas as pd
sys.path.append('../code')
from utilities import dict_combiner
from bump_inference_tests_ainner_ManyBumps import opt_wrapper
from pdb import set_trace as bp

parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='../results/manybumps1switch_dev2', type=str) # base directory for output
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

# n_target_dict = {m: np.load(FLAGS.target_lib_file).shape[0] for m in range(3,13)}

mydict = {
    "grid_dir": [FLAGS.grid_dir],
    "nominal_base_dir": [FLAGS.base_dir],
    "target_lib_file": [FLAGS.target_lib_file],
    "target_lib_name": ["MetaClusters"], # this option reads in targets from file
    "dimer_eps": [1e-16],
    "n_switches": [1],# 2],
    "n_switch_points": [3],
    "start": ["both"], #["on", "off", "both"],
    "id_target": [i for i in range(np.load(FLAGS.target_lib_file.format(FLAGS.m)).shape[0])],
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

def run_main(sett, dothreading=False, make_plots=False):
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

    return

def run_cleanup(output_dir, master_file):
    os.makedirs(output_dir, exist_ok=True)

    print('Beginning Cleanup...')

    if not os.path.exists(output_dir):
        print(output_dir, 'NOT YET CREATED. SKIPPING RUN CLEANUP.')
        return

    try:
        with open(master_file, 'rb') as f_master:
            master = pickle.load(f_master)
    except:
        # only to initialize (first time)
        master = {}

    print('Starting with master of length', len(master))

    # look for completed runs, then consolidate their output and delete the original run data.
    to_delete = []
    for run_dir in os.listdir(output_dir):
        info_file = os.path.join(output_dir, run_dir, 'model_info.pkl')
        experiment_key = os.path.split(run_dir)[-1]
        try:
            with open(info_file, 'rb') as f:
                # read experiment info
                model_info = pickle.load(f)

                #save experiment info to master dict
                master[experiment_key] = model_info

                # remember to delete this experiment
                to_delete.append(os.path.join(output_dir, run_dir))
        except:
            pass

    # write master to file
    with open(master_file, 'wb') as f_master:
        pickle.dump(master, f_master)

    print('Finishing with master of length', len(master))

    # delete all accounted for files
    for del_dir in to_delete:
        shutil.rmtree(del_dir)

    print('Done Cleanup...')
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
        t0 = time.time()
        pool = Pool()
        # results = pool.map(run_main, EXPERIMENT_LIST)
        pool.map(run_main, EXPERIMENT_LIST)
        # results = pool.map(run_wrapper, range(len(EXPERIMENT_LIST)), chunksize=1)
        # chunksize 1 ensures ordered job submission (I think!)
        pool.close()
        pool.join()
        print('All jobs ran in a total of', (time.time() - t0)/60/60, 'hours')

        print('Run finished! Running cleanup now...')
        output_dir = os.path.join(FLAGS.base_dir, 'maxiterO-2_popsizeO-2_polishO-0_maxiterK-1_popsizeK-1_polishK-0')
        master_file = os.path.join(output_dir, 'master_file.pkl')
        run_cleanup(output_dir, master_file)
