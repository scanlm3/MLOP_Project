import numpy as np

c_etas = np.linspace(1, 10, 5)

N = 10

fiberPropotions = np.linspace(0.001, 1, N)

Sizes = [(50, 20), (100, 60), (200, 180), (300, 300), (400, 900)]

c_eps = np.linspace(2 / len(fiberPropotions), 100, 5)

totalTime = 0

for s in Sizes:
    totalTime += s[1] * 5
print(totalTime / 60 + 10)

for c_ep in c_eps:
    for c_et in c_etas:

        eps = 1 / (c_ep * len(fiberPropotions))

        if eps > 1:
            print(eps)
            continue

        print("making job for {},{}, {}".format(eps, c_ep, c_et))

        jobname = "eps{}cet{}.sh".format(eps, c_et)

        with open(jobname, "w") as f:
            f.write("python3 CPDMWUTimeUpdated.py {} {}".format(c_ep, c_et))
            os.system(
                "/usr/bin/sbatch -p cluster -N 1 -n 1 --mail-type=ALL --mail-user=jerrec@rpi.edu -t {} -o out{}{}.out ./{}".format(
                    int(totalTime / 60 + 10), eps, c_et, jobname
                )
            )
