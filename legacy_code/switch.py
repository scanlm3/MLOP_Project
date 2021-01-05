import tensorly as tl
import numpy as np
from numpy.linalg import norm
from PerformanceMetrics import MSE
import math
import time
from itertools import permutations
from CPDMWUTime import sketching_weight, sketch_indices, update_factors, update_weights
from Utils import lookup, proxr, sample_fibers, sampled_kr


def error(X, estimate, norm_x):
    F = estimate[0].shape[1]
    PP = tl.kruskal_to_tensor((np.ones(F), estimate))
    err_e = (np.linalg.norm(X[..., :] - PP[..., :]) ** 2) / norm_x
    return err_e


def adaIteration(configuration):
    (
        X,
        b0,
        n_mb,
        A,
        B,
        C,
        eta,
        Gt,
        norm_x,
        iterations,
        start_time,
        errors,
    ) = configuration
    estimate = [A, B, C]

    # setup parameters
    dim = len(X.shape)
    dim_vec = X.shape

    F = estimate[0].shape[1]

    mu = 0

    for it in range(iterations):
        # randomly permute the dimensions
        block_vec = np.random.permutation(dim)

        d_update = block_vec[0]

        # sampling fibers and forming the X_[d] = H_[d] A_[d]^t least squares
        [tensor_idx, factor_idx] = sample_fibers(n_mb, dim_vec, d_update)
        tensor_idx = tensor_idx.astype(int)

        cols = [tensor_idx[:, x] for x in range(len(tensor_idx[0]))]
        X_sample = X[tuple(cols)]
        X_sample = X_sample.reshape(
            (int(X_sample.size / dim_vec[d_update]), dim_vec[d_update])
        )

        # perform a sampled khatrirao product
        A_unsel = []
        for i in range(d_update):
            A_unsel.append(estimate[i])
        for i in range(d_update + 1, dim):
            A_unsel.append(estimate[i])
        H = np.array(sampled_kr(A_unsel, factor_idx))

        # compute the gradient
        g = (1 / n_mb) * (
            estimate[d_update] @ (H.transpose() @ H + mu * np.eye(F))
            - X_sample.transpose() @ H
            - mu * estimate[d_update]
        )

        # compute the accumlated gradient
        Gt[d_update] = np.abs(np.square(g)) + Gt[d_update]

        eta_adapted = np.divide(eta, np.sqrt(Gt[d_update]))
        d = d_update
        estimate[d_update] = estimate[d_update] - np.multiply(eta_adapted, g)

        estimate[d_update] = proxr(estimate[d_update], d)
        e = error(X, estimate, norm_x)
        t = time.time() - start_time
        errors[t] = ("ada", e)
        print("ada", it, e)

    return (configuration, estimate, errors)


def cpdmwuiteration(configuration):
    (
        X,
        F,
        sketching_rates,
        lamb,
        eps,
        eta,
        A,
        B,
        C,
        norm_x,
        X_unfold,
        iterations,
        start_time,
        errors,
    ) = configuration
    weights = np.array([1] * len(sketching_rates)) / (len(sketching_rates))

    dim_1, dim_2, dim_3 = X.shape

    I = np.eye(F)

    sketching_rates_selected = {}
    for i in range(iterations):
        s = sketching_weight(sketching_rates, weights)

        # Solve Ridge Regression for A,B,C
        A, B, C = update_factors(A, B, C, X_unfold, I, lamb, s, F)

        # Update weights
        p = np.random.binomial(n=1, p=eps)
        if p == 1 and len(sketching_rates) > 1:
            update_weights(
                A,
                B,
                C,
                X_unfold,
                I,
                norm_x,
                lamb,
                weights,
                sketching_rates,
                F,
                eta,
                eps,
            )
        now = time.time()
        e = error(X, [A, B, C], norm_x)
        t = time.time() - start_time
        errors[t] = ("cpdmwu", e)
        print("cpdmwu", i, e)
    return (configuration, [A, B, C], errors)
