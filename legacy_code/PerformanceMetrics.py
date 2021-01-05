import tensorly.tenalg as tl_alg
import numpy as np
import numpy.linalg as np_lin


def cost(X0, norm_x, A):
    X_bar = A[0] @ (tl_alg.khatri_rao(A, skip_matrix=0).T)
    return np_lin.norm(X0 - X_bar) / norm_x


def MSE(X, Xt):
    X1 = X @ np.diag(np.divide(1, np.sqrt(np.sum(np.square(X), axis=0))))

    # Xt = Xt*diag(1./(sqrt(sum(Xt.^2))+eps));

    X2 = Xt @ np.diag(
        np.divide(1, np.sqrt(np.sum(np.square(Xt), axis=0) + np.finfo(float).eps))
    )

    M = np.transpose(X1) @ X2

    MSE_col = np.max(M)
    MSE = np.abs(np.mean(2 - 2 * MSE_col))

    return MSE
