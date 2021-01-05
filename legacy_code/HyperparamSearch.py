from CPDMWUTime import CPDMWUTime
from Utils import save_trial_data, createTensor, initDecomposition, videoToTensor
import time
import pickle
import os
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from math import sqrt, log
import random

random.seed(0)

fiberPropotion = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1]

Sizes = [200]

Trials = 5

arrangements = []

maxtime = 120


def runTest(conf):
    fiberPropotion, Size, trial, Rank, max_time, nu, eps, lamb = conf
    print(
        "Running trial with the following:\n\tProporion of Fibers = {}\n\tSize = {}\n\tRank = {}\n\tTrialNumber = {}\n\tMax Time = {}\n\tnu = {}\n\teps={}".format(
            fiberPropotion, Size, Rank, trial, maxtime, nu, eps
        )
    )

    # Create tensor
    X = createTensor(Size, Rank)

    # init starting
    A_init = initDecomposition(Size, Rank)

    # X, F, sketching_rates, lamb, eps, eta, Hinit, max_time ,sample_interval=.5

    _, _, _, error, rates = CPDMWUTime(
        X, Rank, fiberPropotion, lamb, eps, nu, A_init, max_time, sample_interval=0.5
    )
    return sum([error[x] for x in error.keys() if x > 5]) / len(
        [error[x] for x in error.keys() if x > 5]
    )


cn_low = 0.2
cn_high = 100

cnu_low = 1
cnu_high = 1000
lamb = 0.001

Size = 200

Rank = 50

time = 60
while True:
    print(cn_low, cn_high, cnu_low, cnu_high)
    eps = 1 / (cn_low * len(fiberPropotion))
    nu = sqrt(2 * log(len(fiberPropotion)) / ((cnu_high + cnu_low) / 2))

    total_error_low = 0
    for i in range(5):
        conf = fiberPropotion, Size, i, Rank, time, nu, eps, lamb
        total_error_low += runTest(conf)

    eps = 1 / (cn_high * len(fiberPropotion))

    total_error_high = 0
    for i in range(5):
        conf = fiberPropotion, Size, i, Rank, time, nu, eps, lamb
        total_error_high += runTest(conf)

    if total_error_high > total_error_low:
        cn_high = (cn_high + cn_low) / 1.1
    else:
        cn_low = (cn_high + cn_low) / 3

    print(cn_low, cn_high, cnu_low, cnu_high)

    eps = 1 / (((cn_low + cn_high) / 2) * len(fiberPropotion))
    nu = sqrt(2 * log(len(fiberPropotion)) / cnu_low)

    total_error_low = 0
    for i in range(5):
        conf = fiberPropotion, Size, i, Rank, time, nu, eps, lamb
        total_error_low += runTest(conf)

    nu = sqrt(2 * log(len(fiberPropotion)) / cnu_high)

    total_error_high = 0
    for i in range(5):
        conf = fiberPropotion, Size, i, Rank, time, nu, eps, lamb
        total_error_high += runTest(conf)

    if total_error_high > total_error_low:
        cnu_high = (cnu_high + cnu_high) / 1.5
    else:
        cnu_low = (cnu_high + cnu_high) / 3
