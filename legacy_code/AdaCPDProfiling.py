import numpy as np
import matplotlib.pyplot as plt
from AdaCPD import AdaCPD
from Utils import save_trial_data


# Rank
Fs = [100]

# number of mttrks
iter_mttkrp = 100

# tensor size
I_vec = [300]

num_trials = 1

bs = [18]

b0s = [0.1]

for i0 in range(len(Fs)):
    F = Fs[i0]
    for i1 in range(len(I_vec)):
        for i2 in range(len(bs)):
            for i3 in range(len(b0s)):
                fig, ax = plt.subplots(2, 1)
                for trial_num in range(num_trials):
                    size = I_vec[i1]

                    # input tensor X
                    X = np.zeros((size, size, size))

                    print(
                        "==========================================================================="
                    )
                    print(
                        "Running AdaCPD at trial {} : I equals {} and F equals {}".format(
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
                    for k in range(I[0]):
                        X[:, :, k] = A[0] @ np.diag(A[2][k, :]) @ np.transpose(A[1])

                    # initialize the latent factors
                    Hinit = []
                    for d in range(3):
                        Hinit.append(np.random.random((I[d], F)))

                    b0 = b0s[i3]
                    n_mb = bs[i2]

                    A_init = Hinit
                    tol = np.finfo(float).eps ** 2

                    time_A, NRE_A, MSE_A, A = AdaCPD(X, b0, n_mb, iter_mttkrp, A_init)

                    params = {
                        "b0": b0,
                        "mttkrp": iter_mttkrp,
                        "A_init": A_init,
                        "size": size,
                        "F": F,
                        "n_mb": n_mb,
                    }

                    save_trial_data("adacpd", X, A_gt, time_A, MSE_A, NRE_A, A, params)
