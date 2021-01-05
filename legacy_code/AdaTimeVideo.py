from AdaCPDTime import AdaCPDTime
from Utils import save_trial_data, createTensor, initDecomposition, videoToTensor
import time
import pickle
import os
from joblib import Parallel, delayed
import multiprocessing

fiberPropotions = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1]

Ranks = [100, 150, 200]

Sizes = [600]

Trials = 1

arrangements = []

maxtime = 1200

dir = str(time.time())
os.mkdir(dir)
for Rank in Ranks:
    for fiberPropotion in fiberPropotions:
        for Size in Sizes:
            for trial in range(Trials):
                arrangements.append((fiberPropotion, Size, trial, Rank, maxtime, dir))


def saveAdaTimeTrial(X, fiberPropotion, Size, trial, Rank, b0, max_time, error, dir):
    filename = "{}/AdaResultsVideo_{}_{}_{}_{}_{}_{}.dat".format(
        dir, fiberPropotion, Size, trial, Rank, b0, max_time
    )
    results = {
        "fiberPropotion": fiberPropotion,
        "Size": Size,
        "trial": trial,
        "Rank": Rank,
        "b0": b0,
        "max_time": max_time,
        "error": error,
    }
    with open(filename, "wb") as f:
        pickle.dump(results, f)


def runTest(conf):
    fiberPropotion, Size, trial, Rank, max_time, dir = conf
    print(
        "Running video trial with the following:\n\tProporion of Fibers = {}\n\tSize = {}\n\tRank = {}\n\tTrialNumber = {}\n\tMax Time = {}".format(
            fiberPropotion, Size, Rank, trial, maxtime
        )
    )
    numberOfFibers = Size ** 2
    FibersSampled = max(int(numberOfFibers * fiberPropotion), 1)

    # Create tensor
    X = videoToTensor("600Test.mp4")

    # init starting
    A_init = initDecomposition(Size, Rank)

    b0 = 0.25

    error, _ = AdaCPDTime(
        X, b0, FibersSampled, max_time, A_init, sample_interval=0.5, eta=1
    )
    saveAdaTimeTrial(X, fiberPropotion, Size, trial, Rank, b0, max_time, error, dir)


for i in arrangements:
    runTest(i)
