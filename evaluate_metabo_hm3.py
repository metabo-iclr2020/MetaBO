# Copyright (c) 2019
# Copyright holder of the paper "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".
# Submitted to ICLR 2020 for review.
# All rights reserved.

# ******************************************************************
# evaluate_metabo_hm3.py
# Reproduce results from MetaBO paper on Hartmann-3-Function
# For convenience, we provide the pretrained weights resulting from the experiments described in the paper.
# These weights can be reproduced using train_metabo_hm3.py
# ******************************************************************

import os
from metabo.eval.evaluate import eval_experiment
from metabo.eval.plot_results import plot_results
from metabo.policies.taf.generate_taf_data_hm3 import generate_taf_data_hm3
from gym.envs.registration import register, registry
from datetime import datetime

# set evaluation parameters
afs_to_evaluate = ["MetaBO", "TAF-ME", "TAF-RANKING", "EI", "Random"]
rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "metabo")
logpath = os.path.join(rootdir, "iclr2020", "gobfcts", "hm3", "M_50", "MetaBO-HM3-v0")
savepath = os.path.join(logpath, "eval", datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
n_workers = 10
n_episodes = 100

# evaluate all afs
for af in afs_to_evaluate:
    # set af-specific parameters
    if af == "MetaBO":
        features = ["posterior_mean", "posterior_std", "timestep", "budget", "x"]
        pass_X_to_pi = False
        n_init_samples = 0
        load_iter = 1988  # best ppo iteration during training, determined via metabo/ppo/util/get_best_iter_idx
        deterministic = True
        policy_specs = {}  # will be loaded from the logfiles
    elif af == "TAF-ME" or af == "TAF-RANKING":
        generate_taf_data_hm3(M=50)
        features = ["posterior_mean", "posterior_std", "incumbent", "timestep", "x"]
        pass_X_to_pi = True
        n_init_samples = 0
        load_iter = None  # does only apply for MetaBO
        deterministic = None  # does only apply for MetaBO
        policy_specs = {"TAF_datafile": os.path.join(rootdir, "policies", "taf", "taf_hm3_M_50_N_100.pkl")}
    else:
        features = ["posterior_mean", "posterior_std", "incumbent", "timestep"]
        pass_X_to_pi = False
        n_init_samples = 1
        load_iter = None  # does only apply for MetaBO
        deterministic = None  # does only apply for MetaBO
        if af == "EI":
            policy_specs = {}
        elif af == "Random":
            policy_specs = {}
        else:
            raise ValueError("Unknown AF!")

    # define environment
    env_spec = {
        "env_id": "MetaBO-HM3-v0",
        "D": 3,
        "f_type": "HM3-var",
        "f_opts": {"bound_translation": 0.1,
                   "bound_scaling": 0.1},
        "features": features,
        "T": 30,
        "n_init_samples": n_init_samples,
        "pass_X_to_pi": pass_X_to_pi,
        # parameters were determined offline via type-2-ML on a GP with 100 datapoints
        "kernel_lengthscale": [0.716, 0.298, 0.186],
        "kernel_variance": 0.83,
        "noise_variance": 1.688e-11,
        "use_prior_mean_function": False,
        "local_af_opt": True,
        "N_MS": 2000,
        "N_LS": 2000,
        "k": 5,
        "reward_transformation": "none",
    }

    # register gym environment
    if env_spec["env_id"] in registry.env_specs:
        del registry.env_specs[env_spec["env_id"]]
    register(
        id=env_spec["env_id"],
        entry_point="metabo.environment.metabo_gym:MetaBO",
        max_episode_steps=env_spec["T"],
        reward_threshold=None,
        kwargs=env_spec
    )

    # define evaluation run
    eval_spec = {
        "env_id": env_spec["env_id"],
        "env_seed_offset": 100,
        "policy": af,
        "logpath": logpath,
        "load_iter": load_iter,
        "deterministic": deterministic,
        "policy_specs": policy_specs,
        "savepath": savepath,
        "n_workers": n_workers,
        "n_episodes": n_episodes,
        "T": env_spec["T"],
    }

    # perform evaluation
    print("Evaluating {} on {}...".format(af, env_spec["env_id"]))
    eval_experiment(eval_spec)
    print("Done! Saved result in {}".format(savepath))
    print("**********************\n\n")

# plot (plot is saved to savepath)
print("Plotting...")
plot_results(path=savepath, logplot=True)
print("Done! Saved plot in {}".format(savepath))
