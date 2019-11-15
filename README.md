# MetaBO
Dear reviewers,

this is the source code accompanying the ICLR 2020 Submission: "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".

# Installation
We kindly ask you to clone this repository and run

conda env create -f environment.yml

to create a new conda environment named "metabo" with all python packages required to run the experiments.

# Contents
We provide:
 - Scripts to reproduce the results presented in the paper. These scripts are named evaluate_metabo_<experiment_name>.py.
   They load pre-trained network weights stored in /metabo/iclr2020/<experiment_name> to reproduce the results
   without the need of re-training neural acquisition functions.
 - Scripts to re-train the aforementioned neural acquisition functions. These scripts are named train_metabo_<experiment_name>.py.
 
# Copyright
Copyright (c) 2019

Copyright holder of the paper "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".

Submitted to ICLR 2020 for review.

All rights reserved.
