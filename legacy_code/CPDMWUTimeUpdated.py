from CPDMWUTime import CPDMWUTime
from Utils import save_trial_data, createTensor, initDecomposition
import time
import pickle
import os
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from math import sqrt, log
import random
import sys

N = 10

fiberPropotions = np.linspace(0.001, 1, N)

Rank = 50

Sizes = [((50,30,20), 20), ((100,100,100), 60), ((200,200,200), 180), ((300,300,300), 300), ((400,400,400), 900)]

Trials = 5

arrangements = []

c_eps = float(sys.argv[1])
c_eta = float(sys.argv[2])

dir = str(time.time())
os.mkdir(dir)

# for c_eta in c_etas:
#     for trial in range(Trials):
#         for Size, maxtime in Sizes:
#             arrangements.append(
#                 (fiberPropotions, Size, trial, Rank, maxtime, dir, c_eps, c_eta)
#             )

for fiberPropotion in fiberPropotions:
    for trial in range(Trials):
        for Size,maxtime in Sizes:
            arrangements.append(([fiberPropotion], Size, trial, Rank, maxtime, dir, c_eps, c_eta))


def saveCPDTimeTrial(
    X,
    fiberPropotion,
    lamb,
    eps,
    eta,
    Size,
    trial,
    Rank,
    max_time,
    error,
    rates,
    c_eps,
    c_eta,
    dir,
):
    filename = "{}/CPDResults_{}_{}_{}_{}_{}_{}_{}.dat".format(
        dir, fiberPropotion, Size, trial, Rank, max_time, c_eps, c_eta
    )
    results = {
        "fiberPropotion": fiberPropotion,
        "Size": Size,
        "trial": trial,
        "Rank": Rank,
        "lamb": lamb,
        "eps": eps,
        "eta": eta,
        "max_time": max_time,
        "error": error,
        "rates": rates,
        "c_eps": c_eps,
        "c_eta": c_eta,
    }
    with open(filename, "wb") as f:
        pickle.dump(results, f)


def runTest(conf):
    print("running")
    fiberPropotion, Size, trial, Rank, max_time, dir, c_eps, c_eta = conf
    random.seed

    eps = 1 / (c_eps * len(fiberPropotion))
    eta = sqrt(2 * log(len(fiberPropotion)) / c_eta)
    lamb = 0.001

    if eps > 1:
        print("eps")
        return

    # Create tensor
    X = createTensor(Size, Rank)

    # init starting
    A_init = initDecomposition(Size, Rank)

    b0 = 0.25
    # X, F, sketching_rates, lamb, eps, eta, Hinit, max_time ,sample_interval=.5

    print(
        "Running trial with the following:\n\tProporion of Fibers = {}\n\tSize = {}\n\tRank = {}\n\tTrialNumber = {}\n\tMax Time = {}".format(
            fiberPropotion, Size, Rank, trial, maxtime
        )
    )
    print("\tc_eps = {}\n\teps={}\n\tc_eta={}\n\teta={}".format(c_eps, eps, c_eta, eta))
    _, _, _, error, rates = CPDMWUTime(
        X, Rank, fiberPropotion, lamb, eps, eta, A_init, max_time, sample_interval=0.5
    )
    saveCPDTimeTrial(
        X,
        fiberPropotion,
        lamb,
        eps,
        eta,
        Size,
        trial,
        Rank,
        max_time,
        error,
        rates,
        c_eps,
        c_eta,
        dir,
    )


for i in arrangements:
    runTest(i)
