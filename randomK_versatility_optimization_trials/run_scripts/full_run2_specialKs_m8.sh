#!/bin/bash

# conda activate dimer_opt2

dev_run=0
randomizeK=0
frac_targets=0.1
base_dir='../results/specialK_m8'

for m in 8;
do
  echo $m
  nohup python bump_inference_tests_ainner_BoundedBumpsSeparate_main_randomK_jacobTargets_perDimer_ray.py --randomizeK $randomizeK --dev_run $dev_run --grid_dir badname --m $m --frac_targets $frac_targets --base_dir $base_dir > log_${m}.out 2> log_${m}.err &
done
