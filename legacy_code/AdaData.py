import numpy as np
import matplotlib.pyplot as plt
from AdaCPD import AdaCPD
from Utils import save_trial_data
from joblib import Parallel, delayed
import multiprocessing
import csv
import os

# Rank
Fs = [50]

# number of mttrks
iter_mttkrp = 30

# tensor size
sizes = [200]
# sizes = [100]

num_trials = 2

bs = [5, 10, 20, 30, 40, 50, 75, 150, 200]
# b0s = [.5]
b0s = [0.25]


def RunTest(conf):
    F, size, n_mb, b0, trial = conf
    # input tensor X
    X = np.zeros((size, size, size))

    print("===========================================================================")
    print(
        "Running AdaCPD at trial {} : I equals {} and F equals {}".format(
            trial, size, F
        )
    )
    print("===========================================================================")

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

    A_init = Hinit
    tol = np.finfo(float).eps ** 2

    time_A, NRE_A, A = AdaCPD(X, b0, n_mb, iter_mttkrp, A_init)

    fig = plt.figure()
    ax = plt.gca()
    ax.scatter([x for x in range(len(NRE_A))], NRE_A)
    ax.set_yscale("log")
    plt.ylabel("Cost")
    plt.xlabel("MTTRKS")

    plt.title(
        "AdaCPD with \n"
        "Rank={}, Size={}, fibers sampled={}, b0={}, trial={}".format(
            F, size, n_mb, b0, trial
        )
    )

    plt.savefig("AdaCostF{}I{}nmb{}b0{}trial{}.png".format(F, size, n_mb, b0, trial))

    plt.close()

    total = sum(time_A)

    return {
        "Tensor Rank": F,
        "Decomposition Rank": F,
        "Size": size,
        "b0": b0,
        "Number of Fibers": n_mb,
        "Trial Number": trial,
        "Cost": NRE_A[-1],
        "Total Time": total,
    }


arrangements = []

there = os.path.exists("Adadata.csv")

with open("Adadata.csv", "a") as file:

    fieldnames = [
        "Tensor Rank",
        "Decomposition Rank",
        "Size",
        "b0",
        "Number of Fibers",
        "Trial Number",
        "Cost",
        "Total Time",
    ]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    if not there:
        writer.writeheader()
    for F in Fs:
        for size in sizes:
            for n_mb in bs:
                for b0 in b0s:
                    for trial in range(num_trials):
                        arrangements.append((F, size, n_mb, b0, trial))

    num_cores = int(multiprocessing.cpu_count() - 2)
    datas = Parallel(n_jobs=num_cores, verbose=100)(
        delayed(RunTest)(i) for i in arrangements
    )

    for data in datas:
        writer.writerow(data)
