import tensorly as tl
import numpy as np
from numpy.linalg import norm
from PerformanceMetrics import MSE
import math
import time
from itertools import permutations

from Utils import lookup, proxr, sample_fibers, sampled_kr


def AdaCPDTime(X, b0, n_mb, max_time, A_init, sample_interval=500, eta=1):
    A = A_init

    # setup parameters
    dim = len(X.shape)
    dim_vec = X.shape

    F = A[0].shape[1]

    PP = tl.kruskal_to_tensor((np.ones(F), A))

    err_e = ((np.linalg.norm(X[..., :] - PP[..., :]) ** 2)) / norm(X)

    NRE_A = {0: err_e}

    start = time.time()

    mu = 0

    Gt = [b0 for _ in range(dim)]

    nextSample = sample_interval

    while time.time() - start < max_time:
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
        # print(A[d_update])
        A[d_update] = A[d_update] - np.multiply(eta_adapted, g)

        A[d_update] = proxr(A[d_update], d)

        t = time.time()

        PP = tl.kruskal_to_tensor((np.ones(F), A))
        error = np.linalg.norm(X - PP) ** 2 / norm(X)
        NRE_A[t - start] = error

    return (NRE_A, A)
