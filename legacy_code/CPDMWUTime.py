import numpy as np
from tqdm import tqdm
import random
import tensorly as tl
import time
from numpy.linalg import pinv, norm
import tensorly.tenalg as tl_alg
from Utils import error


def sketching_weight(sketching_rate, weights):
    r = random.uniform(0, sum(weights))
    total_sum = 0

    for i, w in enumerate(weights):
        total_sum += w
        if total_sum > r:
            return sketching_rate[i]

    return weights[-1]


def sketch_indices(s, total_col):
    return np.random.choice(
        range(total_col), size=int(s * total_col), replace=False, p=None
    )


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


def update_weights(
    A, B, C, X_unfold, Id, norm_x, lamb, weights, sketching_rates, rank, nu, eps
):
    for i, w in enumerate(weights):
        start = time.time()
        s = sketching_rates[i]
        A_new, B_new, C_new = update_factors(A, B, C, X_unfold, Id, lamb, s, rank)
        total_time = time.time() - start
        weights[i] *= np.exp(
            -nu
            / eps
            * (
                error(X_unfold[0], norm_x, A_new, B_new, C_new)
                - error(X_unfold[0], norm_x, A, B, C)
            )
            / (total_time)
        )

    weights /= np.sum(weights)
    return


def CPDMWUTime(
    X, F, sketching_rates, lamb, eps, nu, Hinit, max_time, sample_interval=0.5
):
    weights = np.array([1] * len(sketching_rates)) / (len(sketching_rates))

    dim_1, dim_2, dim_3 = X.shape
    A, B, C = Hinit[0], Hinit[1], Hinit[2]

    X_unfold = [tl.unfold(X, m) for m in range(3)]

    norm_x = norm(X)
    I = np.eye(F)

    PP = tl.kruskal_to_tensor((np.ones(F), [A, B, C]))
    error = np.linalg.norm(X - PP) ** 2 / norm_x

    NRE_A = {0: error}

    start = time.time()

    sketching_rates_selected = {}
    now = time.time()
    itr = 1
    with tqdm(position=0) as pbar:
        while now - start < max_time:
            s = sketching_weight(sketching_rates, weights)

            # Solve Ridge Regression for A,B,C
            A, B, C = update_factors(A, B, C, X_unfold, I, lamb, s, F)

            # Update weights
            p = np.random.binomial(n=1, p=eps)
            if p == 1 and len(sketching_rates) > 1:
                update_weights(
                    A, B, C, X_unfold, I, norm_x, lamb, weights, sketching_rates, F, nu, eps
                )
            now = time.time()
            PP = tl.kruskal_to_tensor((np.ones(F), [A, B, C]))
            error = np.linalg.norm(X - PP) ** 2 / norm_x
            elapsed = now - start
            NRE_A[elapsed] = error
            sketching_rates_selected[elapsed] = s
            pbar.set_description(
                "iteration: {}  t: {:.5f}  s: {}   error: {:.5f}  rates: {}".format(
                    itr, elapsed, s, error, sketching_rates
                )
            )
            itr += 1
            pbar.update(1)

    return A, B, C, NRE_A, sketching_rates_selected
