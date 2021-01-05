import numpy as np
from numpy.linalg import norm
import math
import time
from itertools import permutations
import random
import tensorly as tl
import tensorly.tenalg as tl_alg
from numpy.linalg import pinv, norm
from Utils import error, sample_fibers, sampled_kr, weightsStr

from timer import Timer


"""
Picks a sketching rate from weights
"""
def sketching_weight(sketching_rate, weights):
    r = random.uniform(0, sum(weights))
    total_sum = 0

    for i, w in enumerate(weights):
        total_sum += w
        if total_sum > r:
            return sketching_rate[i][0], sketching_rate[i][1]
    return sketching_rate[-1][0], sketching_rate[-1][1]


"""
Picks the indices of the columns to sketch
"""
def sketch_indices(s, total_col):
    return np.random.choice(
        range(total_col), size=int(s * total_col), replace=False, p=None
    )

"""
Solves the ALS subproblems
"""
def update_factors(A, B, C, X_unfold, Id, lamb, s, rank):
    # Update A
    dim_1, dim_2 = X_unfold[0].shape
    idx = sketch_indices(s, dim_2)
    M = (tl_alg.khatri_rao([A, B, C], skip_matrix=0).T)[:, idx]
    A = (lamb * A + X_unfold[0][:, idx] @ M.T) @ pinv(M @ M.T + lamb * Id)

    # Update B
    dim_1, dim_2 = X_unfold[0].shape
    idx = sketch_indices(s, dim_2)
    M = (tl_alg.khatri_rao([A, B, C], skip_matrix=1).T)[:, idx]
    B = (lamb * B + X_unfold[1][:, idx] @ M.T) @ pinv(M @ M.T + lamb * Id)

    # Update C
    dim_1, dim_2 = X_unfold[1].shape
    idx = sketch_indices(s, dim_2)
    M = (tl_alg.khatri_rao([A, B, C], skip_matrix=2).T)[:, idx]
    C = (lamb * C + X_unfold[2][:, idx] @ M.T) @ pinv(M @ M.T + lamb * Id)

    return A, B, C


Gt = []


def AdaIteration(X, X_unfold, A_mat, B_mat, C_mat, b0, eta, F, errors, n_mb, norm_x):
    global Gt

    dim_vec = X.shape
    dim = len(X.shape)
    if Gt == []:
        Gt = [b0 for _ in range(dim)]
    A = [A_mat, B_mat, C_mat]
    mu = 0
    for i in range(1):
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
            A_unsel.append(A[i])
        for i in range(d_update + 1, dim):
            A_unsel.append(A[i])
        H = np.array(sampled_kr(A_unsel, factor_idx))

        # compute the gradient
        g = (1 / n_mb) * (
            A[d_update] @ (H.transpose() @ H + mu * np.eye(F))
            - X_sample.transpose() @ H
            - mu * A[d_update]
        )

        # compute the accumlated gradient
        Gt[d_update] = np.abs(np.square(g)) + Gt[d_update]

        eta_adapted = np.divide(eta, np.sqrt(Gt[d_update]))

        d = d_update

        A[d_update] = A[d_update] - np.multiply(eta_adapted, g)
    return A[0], A[1], A[2], errors

"""
Updated the weights for MWU
"""
def update_weights(
    X,
    A,
    B,
    C,
    X_unfold,
    Id,
    norm_x,
    lamb,
    weights,
    sketching_rates,
    rank,
    eta_cpd,
    eps,
    b0,
    eta_ada,
    F,
):
    dim_1, dim_2, dim_3 = X.shape
    old_error = error(X_unfold[0], norm_x, A, B, C)
    for i, w in enumerate(weights):
        start = time.time()
        s, grad = sketching_rates[i]
        if grad:
            A_new, B_new, C_new, _ = AdaIteration(
                X, X_unfold, A, B, C, b0, eta_ada, F, {}, int(s * dim_1 ** 2), norm_x
            )
        else:
            A_new, B_new, C_new = update_factors(A, B, C, X_unfold, Id, lamb, s, rank)
        total_time = time.time() - start
        weights[i] *= np.exp(
            -eta_cpd
            / eps
            * (error(X_unfold[0], norm_x, A_new, B_new, C_new) - old_error)
            / (total_time)
        )

    weights /= np.sum(weights)


def decompose(X, F, sketching_rates, lamb, eps, eta_cpd, Hinit, max_time, b0, eta_ada):
    weights = np.array([1] * (len(sketching_rates))) / (len(sketching_rates))
    global Gt
    Gt = []
    print(sketching_rates)

    dim_1, dim_2, dim_3 = X.shape
    A, B, C = Hinit[0], Hinit[1], Hinit[2]

    X_unfold = [tl.unfold(X, m) for m in range(3)]

    norm_x = norm(X)
    I = np.eye(F)

    PP = tl.kruskal_to_tensor((np.ones(F), [A, B, C]))
    e = np.linalg.norm(X - PP) ** 2 / norm_x

    NRE_A = {}
    prev_e = e
    sketching_rates_selected = {}
    now = time.time()
    itr = 1

    weights_record = {}
    weights_record[0] = list(weights)

    timer = Timer()
    while timer.elapsed() < max_time:
        s, grad = sketching_weight(sketching_rates, weights)
        if not grad:
            # Solve Ridge Regression for A,B,C
            A, B, C = update_factors(A, B, C, X_unfold, I, lamb, s, F)
        else:
            A, B, C, _ = AdaIteration(
                X, X_unfold, A, B, C, b0, eta_ada, F, None, int(s * dim_1 ** 2), norm_x
            )

        # Update weights
        updated = False
        p = np.random.binomial(n=1, p=eps)
        if p == 1 and len(sketching_rates) > 1:
            print("updating_weights:")
            weights_t = time.time()
            update_weights(
                X,
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
                eta_cpd,
                eps,
                b0,
                eta_ada,
                F,
            )
            print(f"Weights updated in {time.time() - weights_t}:")
            print(weightsStr(weights, sketching_rates))
            updated = True
        timer.pause()
        elapsed = timer.elapsed()
        if updated:
            weights_record[elapsed] = list(weights)
        e = error(X_unfold[0], norm_x, A, B, C)
        NRE_A[elapsed] = (e, grad, s)

        sketching_rates_selected[elapsed] = s

        print(
            f"Iteration Summary:\nIteration: {itr}\nAlgorithim: {'adagrad' if grad else 'sketched ALS'}\nSketching Rate: {s}\nError: {e}\nNormalized Error: {e/norm_x}\nChange in error: {prev_e - e}\nRelative Change in error: {(prev_e - e)/prev_e}\nElapsed: {elapsed}"
        )
        print()
        itr += 1
        prev_e = e
        timer.resume()
    return (A, B, C, NRE_A, weights_record)
