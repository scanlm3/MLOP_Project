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
from CPCombined import *
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pickle


# Array containing tuples representing the size of a third order tensor to decompse and the time alloted
Sizes = [(200, 120), (300, 180), (400, 240)]

# rank of the tensors to generate and to decompose
Ranks = [50, 100, 200]

# Number of trials to run
trials = 3

testname = "eta_.5"
for trial in range(trials):
    for Size, max_time in Sizes:
        for Rank in Ranks:

            b0 = 1
            eta_ada = 1

            lamb = 0.0001

            proprtions = np.linspace(0.1, 1, num=5)

            eta_cpd = 0.001
            X = createTensor(Size, Rank)

            A_init = initDecomposition(Size, Rank)

            A, B, C = A_init[0], A_init[1], A_init[2]

            norm_x = linalg.norm(X)

            sketching_rates = [(p, True) for p in proprtions] + [
                (p, False) for p in proprtions
            ]
            eps = 1 / (2 * len(sketching_rates))

            A, B, C, NRE_A, weights = decompose(
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
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.suptitle("Error and Weight for Decomposition")
            viridis = cm.get_cmap("viridis", len(proprtions))
            ax1.set_title("Error")

            handles = [
                mpatches.Patch(color=viridis(p), label=f"{p} sketching rate")
                for p in proprtions
            ]
            handles = handles + [
                mlines.Line2D([], [], color="black", marker="o", label="Adagrad"),
                mlines.Line2D([], [], color="black", marker="x", label="Sketched ALS"),
            ]
            # handles.append()
            ax1.legend(handles=handles)
            ax1.set_yscale("log")
            for t in NRE_A:
                e, grad, s = NRE_A[t]
                m = "o" if grad else "x"
                ax1.scatter([t], [e], color=viridis(s), marker=m)

            weights_t = list(weights.keys())
            ax2.set_title("Probability Wieghts")
            for i in range(0, len(sketching_rates)):
                weight_y = []
                for t in weights_t:
                    weight_y.append(weights[t][i])
                m = "o" if sketching_rates[i][1] else "x"
                ax2.plot(
                    weights_t, weight_y, color=viridis(sketching_rates[i][0]), marker=m
                )
            ax2.legend(handles=handles)
            fig = mpl.pyplot.gcf()
            fig.set_size_inches(18.5, 10.5)
            plt.savefig(f"{testname}both{Size},{Rank},{trial}.svg")
            ax2.set_yscale("log")
            plt.savefig(f"{testname}both{Size},{Rank},{trial}log.svg")

            pickle.dump(
                (
                    A,
                    B,
                    C,
                    NRE_A,
                    weights,
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
                ),
                open(f"{testname}both{Size},{Rank},{trial}.dat", "wb"),
            )

            sketching_rates = [(p, True) for p in proprtions]
            sketching_rates = [(0.5, True)]
            eps = 1 / (2 * len(sketching_rates))
            A, B, C, NRE_A, weights = decompose(
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
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.suptitle("Error and Weights for Decomposition")
            viridis = cm.get_cmap("viridis", len(proprtions))
            ax1.set_title("Error")

            handles = [
                mpatches.Patch(color=viridis(p), label=f"{p} sketching rate")
                for p in proprtions
            ]
            handles = handles + [
                mlines.Line2D([], [], color="black", marker="o", label="Adagrad"),
                mlines.Line2D([], [], color="black", marker="x", label="Sketched ALS"),
            ]
            # handles.append()
            ax1.legend(handles=handles)
            ax1.set_yscale("log")
            for t in NRE_A:
                e, grad, s = NRE_A[t]
                m = "o" if grad else "x"
                ax1.scatter([t], [e], color=viridis(s), marker=m)

            weights_t = list(weights.keys())
            ax2.set_title("Probability Wieghts")
            for i in range(0, len(sketching_rates)):
                weight_y = []
                for t in weights_t:
                    weight_y.append(weights[t][i])
                m = "o" if sketching_rates[i][1] else "x"
                ax2.plot(
                    weights_t, weight_y, color=viridis(sketching_rates[i][0]), marker=m
                )
            ax2.legend(handles=handles)
            fig = mpl.pyplot.gcf()
            fig.set_size_inches(18.5, 10.5)
            fig.savefig("test2png.png", dpi=100)
            plt.savefig(f"{testname}grad{Size},{Rank},{trial}.svg")
            ax2.set_yscale("log")
            plt.savefig(f"{testname}adagrad{Size},{Rank},{trial}log.svg")
            pickle.dump(
                (
                    A,
                    B,
                    C,
                    NRE_A,
                    weights,
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
                ),
                open(f"{testname}adagrad{Size},{Rank},{trial}.dat", "wb"),
            )

            sketching_rates = [(p, False) for p in proprtions]
            # eps = 1/(2*len(sketching_rates))
            A, B, C, NRE_A, weights = decompose(
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
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.suptitle("Loss and Weights for Decomposition")
            viridis = cm.get_cmap("viridis", len(proprtions))
            ax1.set_title("Error")

            handles = [
                mpatches.Patch(color=viridis(p), label=f"{p} sketching rate")
                for p in proprtions
            ]
            handles = handles + [
                mlines.Line2D([], [], color="black", marker="o", label="Adagrad"),
                mlines.Line2D([], [], color="black", marker="x", label="Sketched ALS"),
            ]
            # handles.append()
            ax1.legend(handles=handles)
            ax1.set_yscale("log")
            for t in NRE_A:
                e, grad, s = NRE_A[t]
                m = "o" if sketching_rates[i][1] else "x"
                ax1.scatter([t], [e], color=viridis(s), marker=m)

            weights_t = list(weights.keys())
            ax2.set_title("Probability Weights")
            for i in range(0, len(sketching_rates)):
                weight_y = []
                for t in weights_t:
                    weight_y.append(weights[t][i])
                m = "o" if sketching_rates[i][1] else "x"
                ax2.plot(
                    weights_t, weight_y, color=viridis(sketching_rates[i][0]), marker=m
                )
            ax2.legend(handles=handles)
            fig = mpl.pyplot.gcf()
            fig.set_size_inches(18.5, 10.5)
            plt.savefig(f"{testname}als{Size},{Rank},{trial}.svg")
            ax2.set_yscale("log")
            plt.savefig(f"{testname}als{Size},{Rank},{trial}log.svg")
            pickle.dump(
                (
                    A,
                    B,
                    C,
                    NRE_A,
                    weights,
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
                ),
                open(f"{testname}sketch-als{Size},{Rank},{trial}.dat", "wb"),
            )
