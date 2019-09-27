# Copyright (c) 2019
# Copyright holder of the paper "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".
# Submitted to ICLR 2020 for review.
# All rights reserved.

# ******************************************************************
# train_metabo_gps.py
# Train MetaBO on GP-samples
# The weights, stats, logs, and the learning curve are stored in metabo/log and can
# be evaluated using metabo/eval/evaluate.py
# ******************************************************************

import os
import multiprocessing as mp
from datetime import datetime
from metabo.policies.policies import NeuralAF
from metabo.ppo.ppo import PPO
from metabo.ppo.plot_learning_curve_online import plot_learning_curve_online
from gym.envs.registration import register

rootdir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "metabo")

# specifiy environment
env_spec = {
    "env_id": "MetaBO-GP-v0",
    "D": 5,
    "f_type": "GP",
    "f_opts": {"lengthscale_low": 0.1,
               "lengthscale_high": 1.0,
               "noise_var_low": 0.1,
               "noise_var_high": 0.1,
               "signal_var_low": 1.0,
               "signal_var_high": 1.0},
    "features": ["posterior_mean", "posterior_std", "timestep", "budget"],
    "T": 50,
    "n_init_samples": 0,
    "pass_X_to_pi": False,
    # will be set individually for each new function to the sampled hyperparameters
    "kernel_lengthscale": None,
    "kernel_variance": None,
    "noise_variance": None,
    "use_prior_mean_function": True,
    "local_af_opt": True,
    "N_MS": 5000,
    "N_LS": 5000,
    "k": 5,
    "reward_transformation": "neg_linear"  # true maximum not known
}

# specify PPO parameters
n_iterations = 2000
batch_size = 1200
n_workers = 10
arch_spec = 4 * [200]
ppo_spec = {
    "batch_size": batch_size,
    "max_steps": n_iterations * batch_size,
    "minibatch_size": batch_size // 20,
    "n_epochs": 4,
    "lr": 1e-4,
    "epsilon": 0.15,
    "value_coeff": 1.0,
    "ent_coeff": 0.01,
    "gamma": 0.98,
    "lambda": 0.98,
    "loss_type": "GAElam",
    "normalize_advs": True,
    "n_workers": n_workers,
    "env_id": env_spec["env_id"],
    "seed": 0,
    "env_seeds": list(range(n_workers)),
    "policy_options": {
        "activations": "relu",
        "arch_spec": arch_spec,
        "use_value_network": True,
        "t_idx": -2,
        "T_idx": -1,
        "arch_spec_value": arch_spec
    }
}

# register environment
register(
    id=env_spec["env_id"],
    entry_point="metabo.environment.metabo_gym:MetaBO",
    max_episode_steps=env_spec["T"],
    reward_threshold=None,
    kwargs=env_spec
)

# log data and weights go here, use this folder for evaluation afterwards
logpath = os.path.join(rootdir, "log", env_spec["env_id"], datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S"))

# set up policy
policy_fn = lambda observation_space, action_space, deterministic: NeuralAF(observation_space=observation_space,
                                                                            action_space=action_space,
                                                                            deterministic=deterministic,
                                                                            options=ppo_spec["policy_options"])

# do training
print("Training on {}.\nFind logs, weights, and learning curve at {}\n\n".format(env_spec["env_id"], logpath))
ppo = PPO(policy_fn=policy_fn, params=ppo_spec, logpath=logpath, save_interval=1)
# learning curve is plotted online in separate process
p = mp.Process(target=plot_learning_curve_online, kwargs={"logpath": logpath, "reload": True})
p.start()
ppo.train()
p.terminate()
plot_learning_curve_online(logpath=logpath, reload=False)
