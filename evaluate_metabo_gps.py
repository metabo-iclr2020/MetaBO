# Copyright (c) 2019
# Copyright holder of the paper "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".
# Submitted to ICLR 2020 for review.
# All rights reserved.

# ******************************************************************
# evaluate_metabo_gps.py
# Reproduce results from MetaBO paper on GP-samples
# For convenience, we provide the pretrained weights resulting from the experiments described in the paper.
# These weights can be reproduced using train_metabo_gps.py
# ******************************************************************

import os
from metabo.eval.evaluate import eval_experiment
from metabo.eval.plot_results import plot_results
from gym.envs.registration import register, registry
from datetime import datetime

# set evaluation parameters
afs_to_evaluate = ["MetaBO", "EI", "Random"]
rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "metabo")
logpath = os.path.join(rootdir, "iclr2020", "gps", "MetaBO-GP-v0")
savepath = os.path.join(logpath, "eval", datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))
n_workers = 10
n_episodes = 100

# evaluate all afs
for af in afs_to_evaluate:
    # set af-specific parameters
    if af == "MetaBO":
        features = ["posterior_mean", "posterior_std", "timestep", "budget"]  # dimensionality agnostic
        pass_X_to_pi = False
        n_init_samples = 0
        load_iter = 1131  # best ppo iteration during training, determined via metabo/ppo/util/get_best_iter_idx
        deterministic = True
        policy_specs = {}  # will be loaded from the logfiles
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
        "env_id": "MetaBO-GP-v0",
        "D": 1,  # MetaBO is dimensionality agnostic and can be evaluated for any D
        "f_type": "GP",
        "f_opts": {"lengthscale_low": 0.1,
                   "lengthscale_high": 1.0,
                   "noise_var_low": 0.1,
                   "noise_var_high": 0.1,
                   "signal_var_low": 1.0,
                   "signal_var_high": 1.0},
        "features": features,
        "T": 50,
        "n_init_samples": n_init_samples,
        "pass_X_to_pi": pass_X_to_pi,
        # will be set individually for each new function to the sampled hyperparameters
        "kernel_lengthscale": None,
        "kernel_variance": None,
        "noise_variance": None,
        "use_prior_mean_function": True,
        "local_af_opt": True,
        "N_MS": 500,
        "N_LS": 500,
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
plot_results(path=savepath, logplot=False)
print("Done! Saved plot in {}".format(savepath))
