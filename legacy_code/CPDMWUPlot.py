import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

directory = "1599505711.6086807"

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


print(len(tests))

for test in tests:
    print(test["fiberPropotion"])
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
    max_time = test["max_time"]

    ax1.scatter(x, y)
    plt.ylabel("Normalized Cost")
    plt.xlabel("Time")
    ax1.set_yscale("log")

    ax1.set_title(
        "Rank {} CPDMWU on {}x{}x{} with sketching rate(s)\n{}\ntrial number {}".format(
            Rank, Size, Size, Size, fiberPropotion, trial
        )
    )
    print("{}/CPDMWU{}{}{}.png".format(plotsDirectoryEach, Size, fiberPropotion, trial))
    plt.savefig(
        "{}/CPDMWU{}{}{}{}.png".format(
            plotsDirectoryEach, Size, Rank, fiberPropotion, trial
        )
    )
    plt.close()

for test in tests:
    fig = plt.figure()
    ax1 = plt.gca()
    rates = test["rates"]

    fiberPropotion = test["fiberPropotion"]

    x = rates.keys()
    y = [rates[i] for i in x]

    Size = test["Size"]
    trial = test["trial"]
    Rank = test["Rank"]
    max_time = test["max_time"]

    ax1.scatter(x, y)
    plt.ylabel("Fibers Sampled")
    plt.xlabel("Time")
    ax1.set_yscale("log")

    ax1.set_title(
        "Rank {} CPDMWU on {}x{}x{} with sketching rate(s)\n{}\ntrial number {}".format(
            Rank, Size, Size, Size, fiberPropotion, trial
        )
    )

    plt.savefig(
        "{}/CPDMWUFibersSelected{}{}{}{}.png".format(
            plotsDirectoryEach, Size, Rank, fiberPropotion, trial
        )
    )
    plt.close()

prop = [0.01, 0.05, 0.1, 0.25, 0.5, 1]
fiberPropotions = [[p] for p in prop]
fiberPropotions.append(prop)

sizes = [50, 100, 200, 300, 400]
for size in sizes:
    fig = plt.figure()
    ax1 = plt.gca()
    for propotion in fiberPropotions:
        print(propotion)
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
        if len(propotion) == 1:
            ax1.scatter(x, y, label=str(propotion[0]), s=12, marker="+")
        else:
            ax1.scatter(x, y, label="Full", s=12, marker="+")
    plt.ylabel("Normalized Cost")
    plt.xlabel("Time")
    ax1.legend()
    ax1.set_yscale("log")
    ax1.set_title("CPDMWU on {}x{}x{}".format(size, size, size))
    plt.savefig("{}/CPDMWU{}.png".format(plotsDirectoryAvg, size))
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
    ax1.set_title("CPDMWU Improvement on {}x{}x{}".format(size, size, size))
    plt.savefig("{}/CPDMWUImprovement{}.png".format(plotsDirectoryAvg, size))
    plt.close()
