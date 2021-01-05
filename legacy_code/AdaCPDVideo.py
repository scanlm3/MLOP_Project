import numpy as np
import matplotlib.pyplot as plt
from AdaCPD import AdaCPD
from Utils import save_trial_data, videoToTensor
import cv2
import multiprocessing
import csv
from joblib import Parallel, delayed
import os
from cycler import cycler
import pickle

X = videoToTensor("videoplayback.mp4")

Fs = [1, 10, 20, 40, 60, 80, 100]


def run(F):
    Hinit = []
    for d in range(3):
        Hinit.append(np.random.random((X.shape[d], F)))

    iters = 30
    # X, b0, n_mb, mttrks, A_init, eta=1
    time_A, NRE_A, A = AdaCPD(X, 0.1, 50, 30, Hinit, None)

    avg = sum(time_A) / len(time_A)

    return {
        "Decomposition Rank": F,
        "Cost": NRE_A,
        "Average MTTRK Time": avg,
        "Time": time_A,
    }


num_cores = int(multiprocessing.cpu_count() - 2)
datas = Parallel(n_jobs=num_cores, verbose=100)(delayed(run)(F) for F in Fs)

there = os.path.exists("AdaVideoData.csv")

fig, (ax, ax2) = plt.subplots(2, figsize=(5, 10))

ax.set_yscale("symlog")
ax2.set_yscale("symlog")
ax.set(xlabel="MTTKR", ylabel="Cost")
ax2.set(xlabel="MTTKR", ylabel="Improvement")

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


for F in Fs:
    results = []
    for r in datas:
        if r["Decomposition Rank"] == F:
            results.append(r["Cost"])
    print(results)
    average = np.mean(results, axis=0)
    std = np.std(results, axis=0)
    ax.scatter(
        [x for x in range(5, len(average))], average[5:], label="rank {}".format(F)
    )
    ax2.scatter([x for x in range(5, len(std))], std[5:], label="rank {}".format(F))

ax.legend(loc="best")
ax2.legend(loc="best")

fig.suptitle("CPDMWU on Video Tensor")

with open("AdaVideoData.csv", "a") as file:
    fieldnames = ["Decomposition Rank", "Cost", "Average MTTRK Time", "Time"]
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    if not there:
        writer.writeheader()
    for data in datas:
        writer.writerow(data)
        Err_A = data["Cost"]
        r = data["Decomposition Rank"]
        points = Err_A
        ax.scatter(
            [x for x in range(5, len(points))], points[5:], label="Rank {}".format(r)
        )


plt.title("AdaCPD on video tensor".format(X.shape))
plt.savefig("AdaVideoCost.png")
plt.close()
