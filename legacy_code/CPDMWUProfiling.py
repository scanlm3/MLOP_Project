import numpy as np
from CPDMWU import CPD_MWU
from Utils import save_trial_data
import tensorly as tl
import tensorly.kruskal_tensor as tl_kruskal

# Rank
Fs = [100]


# tensor size
I_vec = [200]

num_trials = 10

for i0 in range(len(Fs)):
    F = Fs[i0]
    for i1 in range(len(I_vec)):
        for trial_num in range(num_trials):
            size = I_vec[i1]

            # input tensor X
            X = np.zeros((size, size, size))

            print(
                "==========================================================================="
            )
            print(
                "Running CPD-MWU at trial {} : I equals {} and F equals {}".format(
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
            Hinit = []
            for d in range(3):
                Hinit.append(np.random.random((I[d], F)))

            sketching_rates = list(np.linspace(10 ** (-3), 10 ** (-1), 4)) + [1]

            lamb = 0.01
            shape = (300, 300, 300)
            nu = 2

            # CPD_MWU(X, F, sketching_rates, lamb, eps, nu, Hinit, mttkrps=30):
            A, B, C, error, res_time = CPD_MWU(
                X,
                F,
                sketching_rates,
                lamb,
                np.finfo(float).eps ** 2,
                nu,
                Hinit,
                mttkrps=30,
            )

            print(error)

            params = {}
            # def save_trial_data(algo, X, GT, timing, MSE, NRE, A, params):
            save_trial_data("CPDMWU", X, A_gt, res_time, None, error, [A, B, C], params)
