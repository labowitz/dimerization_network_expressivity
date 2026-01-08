#!/bin/bash

# conda activate dimer_opt2

dev_run=0
n_random_Ks=10
base_dir='../results/fullrun1'

for m in 3 4 5 10;
do
  echo $m
  nohup python bump_inference_tests_ainner_BoundedBumpsSeparate_main_randomK_jacobTargets_perDimer_ray.py --dev_run $dev_run --grid_dir badname --m $m --n_random_Ks $n_random_Ks --base_dir $base_dir > log_${m}.out 2> log_${m}.err &
done
