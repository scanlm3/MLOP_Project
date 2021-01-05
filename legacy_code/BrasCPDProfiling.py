import tensorly.random as tl_ran
from BrasCPD import BrasCPD
import numpy as np
import matplotlib.pyplot as plt
from Utils import save_trial_data


# Rank
Fs = [10]

# number of mttrks
iter_mttkrp = 30

# tensor size
I_vec = [300]

num_trials = 2

# number of fibers
bs = [18]

alphas = [0.1]


for i0 in range(len(Fs)):
    F = Fs[i0]
    for i1 in range(len(I_vec)):
        for i2 in range(len(bs)):
            for i3 in range(len(alphas)):
                fig, ax = plt.subplots(2, 1)
                for trial_num in range(num_trials):
                    size = I_vec[i1]

                    # input tensor X
                    X = np.zeros((size, size, size))

                    print(
                        "==========================================================================="
                    )
                    print(
                        "Running at trial {} : I equals {} and F equals {}".format(
                            trial_num, size, F
                        )
                    )
                    print(
                        "==========================================================================="
                    )

                    I = [size] * 3

                    # generate true latent factors
                    A = []
                    for i in range(3):
                        A.append(np.random.random((I[i], F)))

                    A_gt = A

                    # form the tensor
                    for k in range(I[2]):
                        X[:, :, k] = A[0] @ np.diag(A[2][k, :]) @ np.transpose(A[1])

                    # initialize the latent factors
                    Hinit = [np.random.random((I[x], F)) for x in range(3)]

                    b0 = alphas[i3]
                    n_mb = bs[i2]

                    max_it = (I[0] * I[1] / n_mb) * iter_mttkrp

                    A_init = Hinit
                    tol = np.finfo(float).eps ** 2

                    time_A, NRE_A, MSE_A, A = BrasCPD(X, b0, n_mb, max_it, A_init, A_gt)

                    params = {
                        "b0": b0,
                        "mttkrp": iter_mttkrp,
                        "A_init": A_init,
                        "size": size,
                        "F": F,
                        "n_mb": n_mb,
                    }

                    save_trial_data("brascpd", X, A_gt, time_A, MSE_A, NRE_A, A, params)
