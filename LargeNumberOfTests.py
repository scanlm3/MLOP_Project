"""
Runs different tensor with different version of the algorithim
"""
from Utils import save_trial_data, createTensor, initDecomposition, videoToTensor, error
import time
import pickle
import os
import numpy as np
import numpy.linalg as linalg
import collections
import matplotlib.pyplot as plt
from math import sqrt, log
import tensorly as tl
from CPCombined import *
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pickle

Size = 300
Rank = 100
max_time = 90
trials = 100

errors_sketechedALS_1 = []
errors_sketechedALS = []
errors_both = []
errors_adagrad_1 = []
errors_adagrad_5 = []
errors_adagrad_MWU = []

b0 = 1
eta_ada = 1

lamb = 0.0001

proprtions = np.linspace(0.1, 1, num=5)

eta_cpd = 0.0003

eps = 0.2

for i in range(trials):
    X = createTensor(Size, Rank)

    A_init = initDecomposition(Size, Rank)

    A, B, C = A_init[0], A_init[1], A_init[2]

    norm_x = linalg.norm(X)

    eps = 0.15
    sketching_rates = [
        (p, False) for p in proprtions
    ]  # Sketched ALS with multiple sketching rates

    _, _, _, NRE_A, weights = decompose(
        X,
        Rank,
        sketching_rates,
        lamb,
        eps,
        eta_cpd,
        A_init,
        max_time,
        b0,
        eta_ada,
    )

    errors_sketechedALS.append(NRE_A[list(NRE_A.keys())[-1]])
    eps = 0.08
    sketching_rates = [(p, True) for p in proprtions] + [
        (p, False) for p in proprtions
    ]  # both algorithims

    _, _, _, NRE_A, weights = decompose(
        X,
        Rank,
        sketching_rates,
        lamb,
        eps,
        eta_cpd,
        A_init,
        max_time,
        b0,
        eta_ada,
    )

    errors_both.append(NRE_A[list(NRE_A.keys())[-1]])

    eps = .15
    sketching_rates = [
        (p, True) for p in proprtions
    ]  # CPDMWU with adagrad for all sketching rates

    _, _, _, NRE_A, weights = decompose(
        X,
        Rank,
        sketching_rates,
        lamb,
        eps,
        eta_cpd,
        A_init,
        max_time,
        b0,
        eta_ada,
    )

    errors_adagrad_MWU.append(NRE_A[list(NRE_A.keys())[-1]])


def stats(data):
    nums = [d[0] for d in data]

    return np.average(nums), np.median(nums), np.var(nums), np.std(nums)

print(stats(errors_sketechedALS))
print(stats(errors_both))
print(stats(errors_adagrad_MWU))