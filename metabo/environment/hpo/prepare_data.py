# Copyright (c) 2019
# Copyright holder of the paper "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".
# Submitted to ICLR 2020 for review.
# All rights reserved.

# ******************************************************************
# prepare_data.py
# Process the raw datasets from the HPO experiments for usage in MetaBO training and evaluation.
# Processed data is stored in metabo/environment/hpo/processed/<model>/objectives.pkl
#
# Due to licensing issues, the datasets used in this experiment cannot be shipped with the MetaBO package.
# However, you can download the datasets yourself from https://github.com/nicoschilling/ECML2016
# Put the folders "svm" and "adaboost" from this repository into metabo/environment/hpo/data
#
# The gp-hyperparameters for each dataset were estimated offline using type-2-ML on a GP with 100 datasets and
# stored in metabo/environment/hpo/processed/<model>/gp_hyperparameters.pkl
# ******************************************************************

import os
import numpy as np
import pickle as pkl


def prepare_hpo_data(model, datapath):
    # read in data
    if model == "svm":
        param_nos_to_extract = [3, 4]
    elif model == "adaboost":
        param_nos_to_extract = [0, 1]
    else:
        raise ValueError("Unknown model!")
    data_dict = {}

    try:
        for file in os.listdir(datapath):
            if not os.path.isfile(os.path.join(datapath, file)):
                continue
            with open(os.path.join(datapath, file), "r") as f:
                cur_X = None
                cur_Y = None
                for line in f:
                    cur_x = np.zeros((1, len(param_nos_to_extract)))
                    cur_y = np.zeros((1, 1))
                    fields = line.split()
                    found_params = len(param_nos_to_extract) * [False]
                    for field in fields:
                        if ":" in field:
                            colon_pos = field.find(":")
                            param_no = int(field[:colon_pos])
                            param_val = float(field[colon_pos + 1:])
                            if param_no in param_nos_to_extract:
                                param_index_in_x = param_nos_to_extract.index(param_no)
                                found_params[param_index_in_x] = True
                                cur_x[0, param_index_in_x] = param_val
                        else:
                            objective_val = float(field)
                            cur_y[0, 0] = objective_val
                    if all(entry is True for entry in found_params):
                        cur_X = np.concatenate((cur_X, cur_x), axis=0) if cur_X is not None else cur_x
                        cur_Y = np.concatenate((cur_Y, cur_y), axis=0) if cur_Y is not None else cur_y
            data_dict[file] = {"X": cur_X,
                               "Y": cur_Y}
    except FileNotFoundError:
        raise FileNotFoundError("Could not find HPO datasets. Please download the datasets from "
                                "https://github.com/nicoschilling/ECML2016 and put the content of data into metabo/environment/hpo/data")

    if not len(data_dict) == 50:
        raise ValueError(
            "Number of datasets incorrect. Please download the datasets from "
            "https://github.com/nicoschilling/ECML2016 and put the content of data into metabo/environment/hpo/data")

    # normalize domain to unitsquare
    # find domain bounds
    elt = next(iter(data_dict.values()))
    X = elt["X"]
    domain = np.zeros((len(param_nos_to_extract), 2))
    for i in range(len(param_nos_to_extract)):
        domain[i, 0] = np.min(X[:, i])
        domain[i, 1] = np.max(X[:, i])
    # normalize
    for key, val in data_dict.items():
        val["X"] = (val["X"] - domain[:, 0]) / np.ptp(domain, axis=1)

    # store data
    this_path = os.path.dirname(os.path.realpath(__file__))
    savepath = os.path.join(this_path, "processed", model)
    os.makedirs(savepath, exist_ok=True)
    with open(os.path.join(savepath, "objectives.pkl"), "wb") as f:
        pkl.dump(data_dict, f)
