# Copyright (c) 2019
# Copyright holder of the paper "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".
# Submitted to ICLR 2020 for review.
# All rights reserved.

# ******************************************************************
# util.py
# Utilities for ppo package.
# ******************************************************************

import numpy as np
import os
import pickle as pkl


def plot_learning_curve(ax, data, smoothing_range, color, plot_only_smoothed=False):
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    data_smooth = np.array([])
    for i in range(len(data)):
        if i >= smoothing_range:
            data_smooth = np.append(data_smooth,
                                    np.mean(data[i - smoothing_range:i + 1]))
        else:
            data_smooth = np.append(data_smooth, None)

    assert data.size == data_smooth.size
    if not plot_only_smoothed:
        ax.plot(data)
    ax.plot(data_smooth, color=color, lw=4)


def get_last_iter_idx(logpath):
    # returns the index of the last iteration stored in logpath
    files = os.listdir(logpath)
    last_iter = -np.inf
    for file in files:
        if file.startswith("stats_"):
            pos = file.find("_")
            iter = int(file[pos + 1:])
            if iter > last_iter:
                last_iter = iter

    return last_iter


def get_best_iter_idx(logpath):
    # returns the index of the best iteration (w.r.t. average step reward) stored in logpath
    last_iter = get_last_iter_idx(logpath)
    best_iter = -np.inf
    best_avg_step_rew = -np.inf
    with open(os.path.join(logpath, "stats_{:d}".format(last_iter)), "rb") as f:
        stats = pkl.load(f)
    for iter, avg_step_rew in enumerate(stats["avg_step_rews"]):
        if avg_step_rew > best_avg_step_rew:
            best_avg_step_rew = avg_step_rew
            best_iter = iter

    # consistency check
    with open(os.path.join(logpath, "stats_{:d}".format(best_iter)), "rb") as f:
        best_stats = pkl.load(f)
    assert best_stats["batch_stats"]["avg_step_reward"] == best_avg_step_rew

    return best_iter
