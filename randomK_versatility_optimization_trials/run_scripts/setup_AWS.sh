#!/bin/bash

# 1. Download and Install Anaconda
##downloading (you can change your edition by visting anaconda.com)
wget https://repo.anaconda.com/archive/Anaconda3-2023.03-Linux-x86_64.sh

#Installing
sh Anaconda3-2023.03-Linux-x86_64.sh -b
rm Anaconda3-2023.03-Linux-x86_64.sh

~/anaconda3/bin/conda init bash
~/anaconda3/bin/conda config --append channels conda-forge

## get code from github
git clone https://github.com/BenEmert/dimer_computations_colab.git
cd dimer_computations_colab
git checkout paper_v0

## set up virtual environment


# 2. Create virtual environment from requirements listed in the github repo
##Making a Virtual Environment for specific project (Recommended)

#For specific python version
~/anaconda3/bin/conda create -n dimer_opt2 --file requirements2.txt
#Activate your envirnement
~/anaconda3/bin/conda activate dimer_opt2

pip install pymoo==0.5.0
pip install --upgrade eqtk

# Now run!!
m=3
nohup python bump_inference_tests_ainner_BoundedBumpsSeparate_main_randomK_jacobTargets_perDimer_multi.py --dev_run 1 --grid_dir badname --m $m --n_random_Ks 2 > log_${m}.out 2> log_${m}.err &
