import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

directory = "1597005828.2499151"

tests = []

time = 1200

for f in os.listdir(directory):
    with open(directory + "/" + f, "br") as file:
        tests.append(pickle.load(file))

plotsDirectory = directory + "plots"
if not os.path.exists(plotsDirectory):
    os.mkdir(plotsDirectory)
# create plot of cost over time
plotsDirectoryEach = plotsDirectory + "/each"
if not os.path.exists(plotsDirectoryEach):
    os.mkdir(plotsDirectoryEach)

plotsDirectoryAvg = plotsDirectory + "/avg"
if not os.path.exists(plotsDirectoryAvg):
    os.mkdir(plotsDirectoryAvg)


def interpolate(x0, y0, x1, y1, xt):
    return y0 + (xt - x0) * (y1 - y0) / (x1 - x0)


for test in tests:
    fig = plt.figure()
    ax1 = plt.gca()
    error = test["error"]

    x = [i for i in range(time + 1)]
    y = []
    for t in range(time + 1):
        if t in error:
            y.append(error[t])
        else:
            small = 0
            large = sorted(list(error.keys()))[-1]
            for sample in error:
                if sample < t:
                    small = sample
                elif sample < large:
                    large = sample
            interpolated = interpolate(small, error[small], large, error[large], t)
            y.append(interpolated)

    fiberPropotion = test["fiberPropotion"]
    Size = test["Size"]
    trial = test["trial"]
    Rank = test["Rank"]
    b0 = test["b0"]
    max_time = test["max_time"]

    ax1.scatter(x, y)
    plt.ylabel("Normalized Cost")
    plt.xlabel("Time")
    ax1.set_yscale("log")

    ax1.set_title(
        "AdaCPD on {}x{}x{} with {} propotion of fibers sampled\ntrial number {}".format(
            Size, Size, Size, fiberPropotion, trial
        )
    )

    plt.savefig(
        "{}/AdaCPD{}{}{}.png".format(plotsDirectoryEach, Size, fiberPropotion, trial)
    )
    plt.close()

fiberPropotions = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1]
sizes = [300]
for size in sizes:
    fig = plt.figure()
    ax1 = plt.gca()
    for propotion in fiberPropotions:
        x = [i for i in range(time + 1)]
        y_sum = [0 for y in range(time + 1)]
        trials = 0
        for test in tests:
            if test["Size"] == size and test["fiberPropotion"] == propotion:
                error = test["error"]
                trials += 1
                for t in range(time + 1):
                    if t in error:
                        y_sum[t] += error[t]
                    else:
                        small = 0
                        large = sorted(list(error.keys()))[-1]
                        for sample in error:
                            if sample < t:
                                small = sample
                            elif sample < large:
                                large = sample
                        interpolated = interpolate(
                            small, error[small], large, error[large], t
                        )
                        y_sum[t] += interpolated
        if trials == 0:
            continue
        y = [ys / trials for ys in y_sum]
        ax1.scatter(x, y, label=propotion, s=12, marker="+")

    plt.ylabel("Normalized Cost")
    plt.xlabel("Time")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.set_title("AdaCPD on {}x{}x{}".format(size, size, size))
    plt.savefig("{}/AdaCPD{}.png".format(plotsDirectoryAvg, size))
    plt.close()

for test in tests:
    error_array = np.array(list(test["error"].values()))
    time_array = np.array(list(test["error"].keys()))

    error_diff = np.diff(error_array)
    time_diff = np.diff(time_array)

    normalized_improvement = np.divide(error_diff, time_diff)
    normalized_improvement = np.multiply(-1, normalized_improvement)

    normalized_improvement_dic = {}
    t = 1
    for ni in normalized_improvement:
        normalized_improvement_dic[time_array[t]] = ni
        t += 1

    test["improvement"] = normalized_improvement_dic

for size in sizes:
    fig = plt.figure()
    ax1 = plt.gca()
    for propotion in fiberPropotions:
        x = [i for i in range(time + 1)]
        y_sum = [0 for y in range(time + 1)]
        trials = 0
        for test in tests:
            if test["Size"] == size and test["fiberPropotion"] == propotion:
                error = test["improvement"]
                trials += 1
                for t in range(1, time + 1):
                    if t in error:
                        y_sum[t] += error[t]
                    else:
                        small = sorted(list(error.keys()))[0]
                        large = sorted(list(error.keys()))[-1]
                        for sample in error:
                            if sample < t:
                                small = sample
                            elif sample < large:
                                large = sample
                        interpolated = interpolate(
                            small, error[small], large, error[large], t
                        )
                        y_sum[t] += interpolated
        if trials == 0:
            continue
        y = [ys / trials for ys in y_sum]
        print(len(x), len(y))
        ax1.scatter(x[10:], y[10:], label=propotion, s=20, marker="+")

    plt.ylabel("Normalized Cost improvement")
    plt.xlabel("Time")
    ax1.legend()
    ax1.set_yscale("symlog")
    ax1.set_title("AdaCPD Improvement on {}x{}x{}".format(size, size, size))
    plt.savefig("{}/AdaCPDImprovement{}.png".format(plotsDirectoryAvg, size))
    plt.close()
