# Copyright (c) 2019
# Copyright holder of the paper "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".
# Submitted to ICLR 2020 for review.
# All rights reserved.

class Controller:
    """
    Base class for control strategy dependent on the used environment
    """

    def __init__(self):
        self.output = 0
        pass

    def calc_input(self, state, xr):
        raise NotImplementedError
