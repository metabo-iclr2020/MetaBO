# Copyright (c) 2019
# Copyright holder of the paper "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".
# Submitted to ICLR 2020 for review.
# All rights reserved.

# ******************************************************************
# util.py
# Utilities for the MetaBO framework.
# ******************************************************************

import numpy as np


def create_uniform_grid(domain, N_samples_dim):
    D = domain.shape[0]
    x_grid = []
    for i in range(D):
        x_grid.append(np.linspace(domain[i, 0], domain[i, 1], N_samples_dim))
    X_mesh = np.meshgrid(*x_grid)
    X = np.vstack(X_mesh).reshape((D, -1)).T

    return X, X_mesh


def scale_from_unit_square_to_domain(X, domain):
    # X contains elements in unit square, stretch and translate them to lie domain
    return X * domain.ptp(axis=1) + domain[:, 0]


def scale_from_domain_to_unit_square(X, domain):
    # X contains elements in domain, translate and stretch them to lie in unit square
    return (X - domain[:, 0]) / domain.ptp(axis=1)


def get_cube_around(X, diam, domain):
    assert X.ndim == 1
    assert domain.ndim == 2
    cube = np.zeros(domain.shape)
    cube[:, 0] = np.max((X - 0.5 * diam, domain[:, 0]), axis=0)
    cube[:, 1] = np.min((X + 0.5 * diam, domain[:, 1]), axis=0)
    return cube
