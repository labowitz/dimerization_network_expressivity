import os, sys
import time
import numpy as np
import pickle
import shutil
import subprocess
import pandas as pd
from pdb import set_trace as bp

output_dir = "/groups/astuart/mlevine/dimer_computations_colab/results/BoundedBumps_randomK_jacobTarget_perDimer_9.0.0_devrun/maxiterO-2_popsizeO-2_polishO-0_maxiterK-1_popsizeK-1_polishK-0"
os.makedirs(output_dir, exist_ok=True)
log_file = os.path.join(output_dir, 'my.log')
master_file = os.path.join(output_dir, 'master_file.pkl')

sleep_secs = 3*60 # length of time (secs) to wait before trying to submit more jobs. Using 30min.

job_list = ['inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset37000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset36000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset35000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset34000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset33000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset32000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset31000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset30000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset29000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset28000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset27000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset26000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset25000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset24000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset23000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset22000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset21000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset20000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset19000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset18000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset17000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset16000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset15000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset14000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset13000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset12000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset11000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset10000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset9000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset8000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset7000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset6000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset5000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m10_offset0.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset8000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset7000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset6000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset5000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m5_offset0.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m4_offset4000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m4_offset3000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m4_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m4_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m4_offset0.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m3_offset2000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m3_offset1000.job',
'inference_HPC/randomK_jacobTargets_perDimer_autogen_devrun/m3_offset0.job']

def run_cleanup(master_file, output_dir):
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

run_cleanup(master_file, output_dir)
try:
    df = pd.read_csv(log_file)
except:
    df = pd.DataFrame(job_list,columns=['name'])
    df['SUBMITTED'] = 0
    df['id'] = np.arange(len(df))
    df.to_csv(log_file)

while any(df.SUBMITTED==0):

    run_cleanup(master_file, output_dir)

    one_job = df[df.SUBMITTED==0].iloc[0]
    cmd = ['sbatch', one_job['name']]
    status = 1
    while status!=0:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            # check for successful run and print the error
            status = proc.returncode
            out = proc.stdout
        except:
            status = np.random.choice([0,1])
            out = 'EXCEPTION'
        if status!=0:
            my_str = 'Job submission FAILED: {} {}'.format(out, cmd)
            my_str += '\n Will try again in {} mins'.format(sleep_secs/60)
            print(my_str)
            time.sleep(sleep_secs)
    new_str = 'Job submitted: {}'.format(' '.join(cmd))
    print(new_str)
    df.loc[df.id==one_job['id'], 'SUBMITTED'] = 1
    df.to_csv(log_file)
