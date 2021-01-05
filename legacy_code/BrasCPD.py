import tensorly as tl
import numpy as np
from PerformanceMetrics import MSE
import math
import time

from Utils import lookup, proxr, sample_fibers, sampled_kr, error
from itertools import permutations


def BrasCPD(X, b0, n_mb, max_it, A_init, A_gt):
    A = A_init

    # setup parameters
    dim = len(X.shape)
    dim_vec = X.shape

    F = A[0].shape[1]

    PP = tl.kruskal_to_tensor((np.ones(F), A))

    err_e = (np.linalg.norm(X[..., :] - PP[..., :]) ** 2) / X.size

    NRE_A = [err_e]

    MSE_A = []

    mse = 10 ** 10

    for x in list(permutations(range(3), 3)):
        m = (1.0 / 3.0) * (
            MSE(A[x[0]], A_gt[0]) + MSE(A[x[1]], A_gt[1]) + MSE(A[x[2]], A_gt[2])
        )
        mse = min(mse, m)
    MSE_A.append(mse)

    tic = time.time()

    time_A = [time.time() - tic]

    for it in range(1, int(math.ceil(max_it))):
        # step size
        alpha = b0 / (n_mb * (it) ** (10 ** -6))

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
        for i in range(d_update - 1):
            A_unsel.append(A[i])
        for i in range(d_update + 1, dim):
            A_unsel.append(A[i])
        H = np.array(sampled_kr(A_unsel, factor_idx))

        alpha_t = alpha
        d = d_update
        A[d_update] = A[d_update] - alpha_t * (
            A[d_update] @ H.transpose() @ H - (X_sample.transpose() @ H)
        )

        A[d_update] = proxr(A[d_update], d)

        if it % math.ceil((X.shape[0] ** 2 / n_mb)) == 0:
            time_A.append(time.time() - tic)
            err = error(tl.unfold(X, 0), tl.norm(X), A[0], A[1], A[2])
            NRE_A.append(err)

            mse = 10 ** 10

            for x in list(permutations(range(3), 3)):
                m = (1.0 / 3.0) * (
                    MSE(A[x[0]], A_gt[0])
                    + MSE(A[x[1]], A_gt[1])
                    + MSE(A[x[2]], A_gt[2])
                )
                mse = min(mse, m)
            MSE_A.append(mse)

            print("MSE = {}, NRE = {}".format(mse, err))

    return (time_A, NRE_A, MSE_A, A)
