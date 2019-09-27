# Copyright (c) 2019
# Copyright holder of the paper "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".
# Submitted to ICLR 2020 for review.
# All rights reserved.

# ******************************************************************
# plot_results.py
# Functionality for plotting performance of AFs.
# ******************************************************************

import os
from matplotlib import pyplot as plt
import pickle as pkl
import numpy as np
from metabo.eval.evaluate import Result  # for unpickling


def plot_results(path, logplot=False):
    fig, ax = plt.subplots(nrows=1, ncols=1)

    # collect results in savepath
    results = []
    for fn in os.listdir(path):
        if fn.startswith("result"):
            with open(os.path.join(path, fn), "rb") as f:
                result = pkl.load(f)
                results.append(result)

    env_id = results[0].env_id

    # do the plot
    for result in results:
        # prepare rewards_dict
        rewards_dict = {}
        for i, rew in enumerate(result.rewards):
            if isinstance(rew, tuple):
                t = rew[1]
                reward = rew[0]
            else:
                t = i % result.T + 1
                reward = rew

            if str(t) in rewards_dict:
                rewards_dict[str(t)].append(reward)
            else:
                rewards_dict[str(t)] = [reward]

        t_vec, loc, err_low, err_high = [], [], [], []
        for key, val in rewards_dict.items():
            t_vec.append(int(key))
            cur_loc = np.median(val)
            cur_err_low = np.percentile(val, q=70)
            cur_err_high = np.percentile(val, q=30)
            loc.append(cur_loc)
            err_low.append(cur_err_low)
            err_high.append(cur_err_high)

        t_vec, loc, err_low, err_high = np.array(t_vec), np.array(loc), np.array(err_low), np.array(err_high)
        # sort the arrays according to T
        sort_idx = np.argsort(t_vec)
        t_vec = t_vec[sort_idx]
        loc = loc[sort_idx]
        err_low = err_low[sort_idx]
        err_high = err_high[sort_idx]

        if not logplot:
            line = ax.plot(t_vec, loc, label=result.policy)[0]
            ax.fill_between(t_vec, err_low, err_high, alpha=0.2, facecolor=line.get_color())
        else:
            line = ax.semilogy(t_vec, loc, label=result.policy)[0]
            ax.fill_between(t_vec, err_low, err_high, alpha=0.2, facecolor=line.get_color())

    fig.suptitle(env_id)
    ax.grid(alpha=0.3)
    ax.set_xlabel("t", labelpad=0)
    ax.set_ylabel("simple regret")
    ax.legend()

    fig.savefig(fname=os.path.join(path, "plot.png"))
    plt.close(fig)

