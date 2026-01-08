#!/bin/bash

# conda activate dimer_opt2

dev_run=0
randomizeK=1
n_Ks=50
frac_targets=0.1
base_dir='../results/fullrun2_0.1ofTargets'

for m in 2 3 4 5 6 7 8 9 10 11 12 13;
do
  echo $m
  nohup python bump_inference_tests_ainner_BoundedBumpsSeparate_main_randomK_jacobTargets_perDimer_ray.py --dev_run $dev_run --grid_dir badname --m $m --frac_targets $frac_targets --n_Ks $n_Ks --randomizeK $randomizeK --base_dir $base_dir > log_${m}.out 2> log_${m}.err &
done
