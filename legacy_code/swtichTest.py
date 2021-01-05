from switch import adaIteration, error, cpdmwuiteration
from Utils import save_trial_data, createTensor, initDecomposition, videoToTensor
import time
import pickle
import os
import numpy as np
import numpy.linalg as linalg
import collections
import matplotlib.pyplot as plt
from math import sqrt, log
import tensorly as tl

Size = 300
Rank = 50

b0 = 1
eta_ada = 1

iterations_total = 100

lamb = 0.01

proprtions = np.linspace(0.01, 1, num=10)

eps = 1 / (len(proprtions))
eta_cpd = sqrt(2 * log(len(proprtions)))


X = createTensor(Size, Rank)

# init starting
A_init = initDecomposition(Size, Rank)

A, B, C = A_init[0], A_init[1], A_init[2]

numberOfFibers = Size ** 2
FibersSampled = (numberOfFibers * proprtions).astype(int)

n_mb = 100  # 200*200//5#FibersSampled[5]
norm_x = linalg.norm(X)

errors = {0: ("init", error(X, [A, B, C], norm_x))}
start_time = time.time()

configurationada = collections.namedtuple(
    "Configuration", "X b0 fibers A B C eta Gt norm_x iterations start_time errors"
)
configurationcpdmwu = collections.namedtuple(
    "Configuration",
    "X Rank sketching_rates lamb eps eta A B C norm_x X_unfold iterations start_time errors",
)

dim = len(X.shape)
X_unfold = [tl.unfold(X, m) for m in range(3)]

switches = 4

Gt = [b0 for _ in range(dim)]

iters = [(500, 0), (0, 200)]

for adaIters, cpdmwuIters in iters:
    # adaIters = int(iterations_total - i * iterations_total/switches)
    # cpdmwuIters = iterations_total - adaIters

    conf1 = configurationada(
        X, b0, n_mb, A, B, C, eta_ada, Gt, norm_x, adaIters, start_time, errors
    )
    print(conf1)
    c, estimate, errors = adaIteration(conf1)
    A, B, C = estimate[0], estimate[1], estimate[2]
    conf2 = configurationcpdmwu(
        X,
        Rank,
        proprtions,
        lamb,
        eps,
        eta_cpd,
        A,
        B,
        C,
        norm_x,
        X_unfold,
        cpdmwuIters,
        start_time,
        errors,
    )
    c, estimate, errors = cpdmwuiteration(conf2)
    A, B, C = estimate[0], estimate[1], estimate[2]

print(errors)

plt.plot(1, 1)
plt.yscale("log")
for t in errors:
    algo, error = errors[t]
    if algo == "init":
        plt.plot([t], [error], "+", color="orange")
    elif algo == "ada":
        plt.plot([t], [error], "+", color="blue")
    elif algo == "cpdmwu":
        plt.plot([t], [error], "+", color="green")
plt.show()
