#!/bin/bash

# conda activate dimer_opt2

dev_run=1
n_random_Ks=3

for m in 3 5 10;
do
  echo $m
  nohup python bump_inference_tests_ainner_BoundedBumpsSeparate_main_randomK_jacobTargets_perDimer_ray.py --dev_run $dev_run --grid_dir badname --m $m --n_random_Ks $n_random_Ks > log_${m}.out 2> log_${m}.err &
done
