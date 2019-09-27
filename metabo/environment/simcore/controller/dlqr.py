# Copyright (c) 2019
# Copyright holder of the paper "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".
# Submitted to ICLR 2020 for review.
# All rights reserved.

import numpy as np
from scipy.linalg import solve_discrete_are

import metabo.environment.simcore.controller.base_controller as cont


class Dlqr(cont.Controller):
    """
    Time-discrete linear-quadratic regulator

    """

    def __init__(self, environment, q, r, xr=None, ur=None, ctrl_dt=None):
        self.nx = environment.param.nx
        self.nu = environment.param.nu
        cont.Controller.__init__(self)

        if xr is None:
            xr = np.zeros(self.nx)

        if ur is None:
            ur = np.zeros(self.nu)

        # rest position and u0 is currently set to zero
        if ctrl_dt is None:
            ad, bd = environment.linearize(xr, ur)
        else:
            ad, bd = environment.linearize(xr, ur, dt=ctrl_dt)

        # get the solution of the discrete riccati equation
        p = np.array(solve_discrete_are(ad, bd, q, r))

        # calculate feedback gain
        self._K = np.dot(np.linalg.inv(
            np.array(r + np.dot(bd.T.dot(p), bd), dtype=float)),
            np.dot(bd.T.dot(p), ad))

    def calc_input(self, state, xr):
        self.output = -self._K @ (state - xr)
        return self.output
