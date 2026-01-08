import os, sys
import time
import numpy as np
import subprocess

job_directory = 'inference_HPC/randomK_jacobTargets_autogen'
os.makedirs(job_directory, exist_ok=True)

sbatch_str = """#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=24:00:00   # walltime
#SBATCH --array=0-1000      # how many tasks in the array
#SBATCH -J "randomK_jacobTargets_m{m}_{offset}"   # job name
#SBATCH --output=slurm/%x.%j.out
#SBATCH --error=slurm/%x.%j.err
#SBATCH --mem-per-cpu={mem}G

base_dir="/groups/astuart/mlevine/dimer_computations_colab/results/BoundedBumps_randomK_jacobTarget_8.1.0"

id=$(( {offset} + $SLURM_ARRAY_TASK_ID))
srun python bump_inference_tests_ainner_BoundedBumpsSeparate_main_randomK_jacobTargets.py --id $id --base_dir $base_dir --grid_dir badname --m {m}
"""

sleep_secs = 60*60 # length of time (secs) to wait before trying to submit more jobs. Using 1 hour.
max_targets = 691
num_Ks = 10
max_jobs = num_Ks*max_targets

n_per_batch_submission = 1000

mem_dict = {3: 5, 4: 10, 5: 20, 10: 20}
for m in [3, 4, 5, 10]:
    for offset in np.arange(0, max_jobs, 1000):
        job_file = os.path.join(job_directory, "m{}_offset{}.job".format(m, offset))

        with open(job_file, 'w') as fh:
            fh.writelines(sbatch_str.format(m=m, offset=offset, mem=mem_dict[m]))

        cmd = ['sbatch', job_file]
        status = 1
        while status!=0:
            proc = subprocess.run(cmd, capture_output=True, text=True)
            # check for successful run and print the error
            status = proc.returncode
            if status!=0:
                print('Job submission FAILED:', proc.stdout, cmd)
                print('Will try again in {} hrs'.format(sleep_secs/60/60))
                time.sleep(sleep_secs)
        print('Job submitted:', ' '.join(cmd))
