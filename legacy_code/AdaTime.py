import numpy as np
import matplotlib.pyplot as plt
from AdaCPD import AdaCPD
from Utils import save_trial_data
from joblib import Parallel, delayed
import multiprocessing
import csv
import os
from cycler import cycler
from datetime import datetime
import pickle

# Rank
Fs = [50]

time = 60

# tensor size
sizes = [500]
# sizes = [10,20,30,40]

num_trials = 20

# bs = [5,10,20,30,40,50,75,150,200]
bs = [140000]
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

    time_A, NRE_A, A = AdaCPD(X, b0, n_mb, time, A_init)

    cost_diff = np.diff(NRE_A)
    time_diff = np.diff(time_A)

    heristic = np.divide(cost_diff, time_diff)


    fig = plt.figure()
    ax = plt.gca()
    ax.scatter([x for x in range(5, len(heristic))], heristic[5:])
    ax.set_yscale("symlog")
    plt.ylabel("Improvement")
    plt.xlabel("MTTRK")

    plt.title(
        "AdaCPD with \n"
        "Rank={}, Size={}, fibers sampled={}, b0={}, trial={}".format(
            F, size, n_mb, b0, trial
        )
    )

    # plt.savefig('adaPlots/AdaHeristicF{}I{}nmb{}b0{}trial{}.png'.format(F, size, n_mb, b0, trial))

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
        "Results": heristic,
    }


arrangements = []
for F in Fs:
    for size in sizes:
        for n_mb in bs:
            for b0 in b0s:
                for trial in range(num_trials):
                    arrangements.append((F, size, n_mb, b0, trial))

there = os.path.exists("Adadata.csv")

num_cores = int(multiprocessing.cpu_count() - 2)
# datas = Parallel(n_jobs=num_cores, verbose=100)(delayed(
# RunTest)(i)for i in arrangements)

for i in arrangements:
    print(i)
    RunTest(i)
quit()

now = datetime.now()
dt_string = now.strftime("%d%m%Y%H%M%S")

pickle.dump(datas, open("Adadata{}.dat".format(dt_string), "wb"))

fig, (ax, ax2) = plt.subplots(2, figsize=(5, 10))

ax.set_yscale("symlog")
ax2.set_yscale("log")
ax.set(xlabel="MTTKR", ylabel="Average Improvment")
ax2.set(xlabel="MTTKR", ylabel="Improvment Standard Deviation")

F, n_mb, b0 = 50, 100, 0.25


plt.rc(
    "axes",
    prop_cycle=(
        cycler(
            "color",
            [
                "r",
                "g",
                "b",
                "c",
                "m",
                "y",
                "k",
                "LIGHTSALMON",
                "DEEPPINK",
                "DARKORANGE",
                "REBECCAPURPLE",
            ],
        )
    ),
)


for s in sizes:
    results = []
    for r in datas:
        if r["Size"] == s:
            results.append(r["Results"])
    results = np.array(results)
    average = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    ax.scatter(
        [x for x in range(5, len(average))], average[5:], label="size {}".format(s)
    )
    ax2.scatter([x for x in range(5, len(std))], std[5:], label="size {}".format(s))

ax.legend(loc="best")
ax2.legend(loc="best")

fig.suptitle("AdaCPD with rank={}, fibers sampled={}, b0={}".format(F, n_mb, b0))
plt.savefig("AdaHeristic.png")

plt.close()
