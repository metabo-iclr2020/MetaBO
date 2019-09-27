# Copyright (c) 2019
# Copyright holder of the paper "Meta-Learning Acquisition Functions for Transfer Learning in Bayesian Optimization".
# Submitted to ICLR 2020 for review.
# All rights reserved.

# ******************************************************************
# policies.py
# Implementation of the MetaBO neural AF as well as benchmark AFs.
# ******************************************************************

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from metabo.policies.mlp import MLP
from scipy.stats import norm
import pickle as pkl
import GPy


class NeuralAF(nn.Module):
    """
    Base class for MetaBO-Policies. Subclasses have to implement init_structure() and forward().

    SHAPES:
    forward()
     states: (N_batch, N_grid, N_features)
     logits: (N_batch, N_grid)
     values: (N_batch, )
    act(): only one action/value at a time in self.act()
     state: (N_grid, N_features)
     action: ()
     value: ()
    predict_vals_logps_ents()
     states: (N_batch, N_grid, N_features)
     actions: (N_batch, )
     values: (N_batch, )
     logprobs: (N_batch, )
     entropies: (N_batch, )
    """

    def __init__(self, observation_space, action_space, deterministic, options):
        super(NeuralAF, self).__init__()
        self.N_features = None  # has to be set in init_structure()
        self.deterministic = deterministic

        # initialize the network structure
        self.init_structure(observation_space=observation_space, action_space=action_space, options=options)

        # initialize weights
        self.apply(self.init_weights)

    def init_structure(self, observation_space, action_space, options):
        self.N_features = observation_space.shape[1]

        # activation function
        if options["activations"] == "relu":
            f_act = F.relu
        elif options["activations"] == "tanh":
            f_act = torch.tanh
        else:
            raise NotImplementedError("Unknown activation function!")

        # policy network
        self.policy_net = MLP(d_in=self.N_features, d_out=1, arch_spec=options["arch_spec"], f_act=f_act)

        # value network
        if "use_value_network" in options and options["use_value_network"]:
            self.use_value_network = True
            self.value_net = MLP(d_in=2, d_out=1, arch_spec=options["arch_spec_value"], f_act=f_act)
            self.t_idx = options["t_idx"]
            self.T_idx = options["T_idx"]
        else:
            self.use_value_network = False

    def forward(self, states):
        assert states.dim() == 3
        assert states.shape[-1] == self.N_features

        # policy network
        logits = self.policy_net.forward(states)
        logits.squeeze_(2)

        # value network
        if self.use_value_network:
            tT = states[:, [0], [self.t_idx, self.T_idx]]
            values = self.value_net.forward(tT)
            values.squeeze_(1)
        else:
            values = torch.zeros(states.shape[0]).to(logits.device)

        return logits, values

    def af(self, state):
        state = torch.from_numpy(state[None, :].astype(np.float32))
        with torch.no_grad():
            out = self.forward(state)
        af = out[0].to("cpu").numpy().squeeze()

        return af

    def act(self, state):
        # here, state is assumed to contain a single state, i.e. no batch dimension
        state = state.unsqueeze(0)  # add batch dimension
        out = self.forward(state)
        logits = out[0]
        value = out[1]
        if self.deterministic:
            action = torch.argmax(logits)
        else:
            distr = Categorical(logits=logits)
            # to sample the action, the policy uses the current PROCESS-local random seed, don't re-seed in pi.act
            action = distr.sample()

        return action.squeeze(0), value.squeeze(0)

    def predict_vals_logps_ents(self, states, actions):
        assert actions.dim() == 1
        assert states.shape[0] == actions.shape[0]
        out = self.forward(states)
        logits = out[0]
        values = out[1]

        distr = Categorical(logits=logits)
        logprobs = distr.log_prob(actions)
        entropies = distr.entropy()

        return values, logprobs, entropies

    def set_requires_grad(self, requires_grad):
        for p in self.parameters():
            p.requires_grad = requires_grad

    @staticmethod
    def num_flat_features(x):
        return np.prod(x.size()[1:])

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Linear:
            m.weight.data.normal_(mean=0.0, std=0.01)
            m.bias.data.fill_(0.0)


class UCB():
    def __init__(self, feature_order, kappa, D=None, delta=None):
        self.feature_order = feature_order
        self.kappa = kappa
        self.D = D
        self.delta = delta
        assert not (self.kappa == "gp_ucb" and self.D is None)
        assert not (self.kappa == "gp_ucb" and self.delta is None)
        np.random.seed(0)  # make UCB behave deterministically

    def act(self, state):
        state = state.numpy()
        ucbs = self.af(state)
        action = np.random.choice(np.flatnonzero(ucbs == ucbs.max()))
        value = 0.0

        action = torch.tensor([action], dtype=torch.int64)
        value = torch.tensor([value])
        return action.squeeze(0), value.squeeze(0)

    def af(self, state):
        mean_idx = self.feature_order.index("posterior_mean")
        means = state[:, mean_idx]
        std_idx = self.feature_order.index("posterior_std")
        stds = state[:, std_idx]
        if self.kappa == "gp_ucb":
            timestep_idx = self.feature_order.index("timestep")
            timesteps = state[:, timestep_idx] + 1  # MetaBO timesteps start at 0
        else:
            timesteps = None

        kappa = self.compute_kappa(timesteps)
        ucbs = means + kappa * stds
        return ucbs

    def compute_kappa(self, timesteps):
        # https: // arxiv.org / pdf / 0912.3995.pdf
        # https: // arxiv.org / pdf / 1012.2599.pdf
        if self.kappa == "gp_ucb":
            assert timesteps is not None
            nu = 1
            tau_t = 2 * np.log(timesteps ** (self.D / 2 + 2) * np.pi ** 2 / (3 * self.delta))
            kappa = np.sqrt(nu * tau_t)
        else:
            assert timesteps is None
            kappa = self.kappa
        return kappa

    def set_requires_grad(self, flag):
        pass


class EI():
    def __init__(self, feature_order):
        self.feature_order = feature_order

    def act(self, state):
        state = state.numpy()
        eis = self.af(state)
        action = np.random.choice(np.flatnonzero(eis == eis.max()))
        value = 0.0

        action = torch.tensor([action], dtype=torch.int64)
        value = torch.tensor([value])
        return action.squeeze(0), value.squeeze(0)

    def af(self, state):
        mean_idx = self.feature_order.index("posterior_mean")
        means = state[:, mean_idx]
        std_idx = self.feature_order.index("posterior_std")
        stds = state[:, std_idx]
        incumbent_idx = self.feature_order.index("incumbent")
        incumbents = state[:, incumbent_idx]

        mask = stds != 0.0
        eis, zs = np.zeros((means.shape[0],)), np.zeros((means.shape[0],))
        zs[mask] = (means[mask] - incumbents[mask]) / stds[mask]
        pdf_zs = norm.pdf(zs)
        cdf_zs = norm.cdf(zs)
        eis[mask] = (means[mask] - incumbents[mask]) * cdf_zs + stds[mask] * pdf_zs
        return eis

    def set_requires_grad(self, flag):
        pass


class PI():
    def __init__(self, feature_order, xi):
        self.feature_order = feature_order
        self.xi = xi

    def act(self, state):
        state = state.numpy()
        pis = self.af(state)
        action = np.random.choice(np.flatnonzero(pis == pis.max()))
        value = 0.0

        action = torch.tensor([action], dtype=torch.int64)
        value = torch.tensor([value])
        return action.squeeze(0), value.squeeze(0)

    def af(self, state):
        mean_idx = self.feature_order.index("posterior_mean")
        means = state[:, mean_idx]
        std_idx = self.feature_order.index("posterior_std")
        stds = state[:, std_idx]
        incumbent_idx = self.feature_order.index("incumbent")
        incumbents = state[:, incumbent_idx]

        mask = stds != 0.0
        pis, zs = np.zeros((means.shape[0],)), np.zeros((means.shape[0],))
        zs[mask] = (means[mask] - (incumbents[mask] + self.xi)) / stds[mask]
        cdf_zs = norm.cdf(zs)
        pis[mask] = cdf_zs
        return pis

    def set_requires_grad(self, flag):
        pass


class TAF():
    # implements the Transfer Acquisition Function from Wistuba et. al., Mach Learn (2018)
    # https://rd.springer.com/content/pdf/10.1007%2Fs10994-017-5684-y.pdf
    def __init__(self, datafile, mode="me", rho=None):
        self.datafile = datafile
        self.models_source = []  # will be filled in self.generate_source_models()
        self.generate_source_models()
        self.mode = mode
        self.rho = rho
        if self.mode == "me":
            assert self.rho is None
        elif self.mode == "ranking":
            assert self.rho > 0
        else:
            raise ValueError("Unknown TAF-mode!")

    def generate_source_models(self):
        with open(self.datafile, "rb") as f:
            data = pkl.load(f)
        self.data = data

        self.D = data["D"]
        self.M = data["M"]
        for i in range(self.M):
            self.models_source.append(self.train_gp(X=data["X"][i], Y=data["Y"][i],
                                                    kernel_lengthscale=data["kernel_lengthscale"][i],
                                                    kernel_variance=data["kernel_variance"][i],
                                                    noise_variance=data["noise_variance"][i],
                                                    use_prior_mean_function=data["use_prior_mean_function"][i]))

    def act(self, state, X_target, model_target):
        state = state.numpy()
        tafs = self.af(state, X_target, model_target)
        action = np.random.choice(np.flatnonzero(tafs == tafs.max()))
        value = 0.0

        action = torch.tensor([action], dtype=torch.int64)
        value = torch.tensor([value])
        return action.squeeze(0), value.squeeze(0)

    def train_gp(self, X, Y, kernel_lengthscale, kernel_variance, noise_variance, use_prior_mean_function):
        kernel = GPy.kern.RBF(input_dim=self.D,
                              variance=kernel_variance,
                              lengthscale=kernel_lengthscale,
                              ARD=True)

        if use_prior_mean_function:
            mf = GPy.core.Mapping(self.D, 1)
            mf.f = lambda X: np.mean(Y, axis=0)[0] if Y is not None else 0.0
            mf.update_gradients = lambda a, b: 0
            mf.gradients_X = lambda a, b: 0
        else:
            mf = None

        normalizer = False

        gp = GPy.models.gp_regression.GPRegression(X, Y,
                                                   noise_var=noise_variance,
                                                   kernel=kernel,
                                                   mean_function=mf,
                                                   normalizer=normalizer)
        gp.Gaussian_noise.variance = noise_variance
        gp.rbf.lengthscale = kernel_lengthscale
        gp.rbf.variance = kernel_variance

        return gp

    def af(self, state, X_target, model_target):
        # gather predictions of target gp
        mean_idx = 0
        means_target = state[:, mean_idx]
        std_idx = 1
        stds_target = state[:, std_idx]
        incumbent_idx = std_idx + self.D + 1
        incumbents_target = state[:, incumbent_idx]

        # gather predicitions of source gps
        xs = state[:, std_idx + 1:std_idx + 1 + self.D]
        means_source, stds_source = [], []
        for i in range(self.M):
            cur_means, cur_vars = self.models_source[i].predict_noiseless(xs)
            cur_stds = np.sqrt(cur_vars)
            means_source.append(cur_means)
            stds_source.append(cur_stds)
        means_source = np.concatenate(means_source, axis=1)
        stds_source = np.concatenate(stds_source, axis=1)

        # compute weights
        if self.mode == "me":  # product of experts
            beta = 1 / (self.M + 1)
            weights = [beta * stds_source[:, i] ** (-2) for i in range(self.M)]
            weights.append(beta * stds_target ** (-2))
            weights = np.array(weights).T
        elif self.mode == "ranking":  # ranking-based
            t = X_target.shape[0] if X_target is not None else 0

            # Epanechnikov quadratic kernel
            def kern(a, b, rho):
                def gamma(x):
                    gamma = 3 / 4 * (1 - x ** 2) if x <= 1 else 0.0
                    return gamma

                kern = gamma(np.linalg.norm(a - b) / rho)
                return kern

            # compute ranking-based meta-features
            chi = [np.zeros((t ** 2,)) for _ in range(self.M + 1)]
            for k in range(self.M + 1):
                for i in range(t):
                    xi = X_target[i, :].reshape(1, self.D)
                    mu_k_i, _ = self.models_source[k].predict_noiseless(xi) if k < self.M \
                        else model_target.predict_noiseless(xi)
                    for j in range(t):
                        xj = X_target[j, :].reshape(1, self.D)
                        mu_k_j, _ = self.models_source[k].predict_noiseless(xj) if k < self.M \
                            else model_target.predict_noiseless(xj)
                        chi[k][j + i * t] = 1 / (t * (t - 1)) if mu_k_i.item() > mu_k_j.item() else 0.0

            # compute weights
            weights = []
            for i in range(self.M + 1):
                weights.append(kern(chi[i], chi[self.M + 1 - 1], self.rho))

            weights = np.array(weights)
            weights = np.tile(weights, (xs.shape[0], 1))

        # compute EI(x) of target model
        mask = stds_target != 0.0
        eis_target, zs = np.zeros((means_target.shape[0],)), np.zeros((means_target.shape[0],))
        zs[mask] = (means_target[mask] - incumbents_target[mask]) / stds_target[mask]
        pdf_zs = norm.pdf(zs)
        cdf_zs = norm.cdf(zs)
        eis_target[mask] = (means_target[mask] - incumbents_target[mask]) * cdf_zs + stds_target[mask] * pdf_zs

        # compute predicted improvements of source models
        incumbents_source = []
        for i in range(self.M):
            if X_target is None:
                cur_incumbent = incumbents_target[0]
            else:
                cur_incumbent = np.max(self.models_source[i].predict_noiseless(X_target)[0])
            incumbents_source.append(cur_incumbent)
        incumbents_source = np.array(incumbents_source)
        Is_source = means_source - incumbents_source
        Is_source[Is_source < 0.0] = 0.0

        # compute TAF
        source_af = np.sum((weights[:, :-1] * Is_source), axis=1)
        target_af = weights[:, -1] * eis_target
        weight_sum = np.sum(weights, axis=1)
        taf = (source_af + target_af) / weight_sum

        return taf

    def set_requires_grad(self, flag):
        pass
