# Copyright (c) 2019
# Copyright holder of the paper "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".
# Submitted to ICLR 2020 for review.
# All rights reserved.

# ******************************************************************
# evaluate_metabo_adaboost.py
# Function to plot learning curves online from PPO-log output.
# ******************************************************************

from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl
import os
import time
from metabo.ppo.util import plot_learning_curve
import datetime


def plot_learning_curve_online(logpath, reload=True):
    plt.rc('text', usetex=False)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(13, 5))

    while True:
        time.sleep(.1)

        files = os.listdir(os.path.join(logpath))
        last_iter = -np.inf
        for file in files:
            if file.startswith("stats_"):
                pos = file.find("_")
                iter = int(file[pos + 1:])
                if iter > last_iter:
                    last_iter = iter
        if last_iter != -np.inf:
            with open(os.path.join(logpath, "params_{:d}".format(int(last_iter))), "rb") as f:
                params = pkl.load(f)
            with open(os.path.join(logpath, "stats_{:d}".format(int(last_iter))), "rb") as f:
                stats = pkl.load(f)
        else:
            continue

        n_steps_to_go = 0
        avg_step_rew = np.array(stats["avg_step_rews"])
        avg_init_rew = np.array(stats["avg_init_rews"])
        avg_term_rew = np.array(stats["avg_term_rews"])
        perc = stats["perc"]
        n_steps_to_go += (1 - stats["perc"] / 100) * params["max_steps"]
        if stats["perc"] < 100:
            sps = stats["batch_stats"]["sps"]
            n_workers = params["n_workers"]
        else:
            sps = 0
            n_workers = 0

        x = np.arange(avg_step_rew.size)

        eta_sec = n_steps_to_go // sps if sps != 0 else (0 if perc >= 100 else None)
        eta = datetime.timedelta(seconds=eta_sec)
        fig.suptitle(
            "Env: {}, Date: {}\n n_workers_running: {}, {:.0f}sps, topt: {:.0f}s, {:.2f}%, ETA = {}".format(
                params["env_id"], os.path.basename(logpath), n_workers, sps, stats["t_optim"], perc, eta))

        ax.cla()
        ax.plot(x, avg_step_rew, color="r", alpha=0.5, label="avg. step reward")
        ax.plot(x, avg_init_rew, color="g", alpha=0.5, label="avg. initial reward")
        ax.plot(x, avg_term_rew, color="b", alpha=0.5, label="avg. final reward")
        plot_learning_curve(ax, avg_step_rew, color="r", smoothing_range=25, plot_only_smoothed=True)
        plot_learning_curve(ax, avg_init_rew, color="g", smoothing_range=25, plot_only_smoothed=True)
        plot_learning_curve(ax, avg_term_rew, color="b", smoothing_range=25, plot_only_smoothed=True)
        ax.set_xlabel("Iteration")
        ax.grid()
        ax.legend()
        fig.savefig(os.path.join(logpath, "learning_curves.png"))

        if not reload:
            break

        time.sleep(5)
