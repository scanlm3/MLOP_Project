import matplotlib.pyplot as plt

x = [5, 10, 20, 30, 40, 50, 75, 150, 200]

y0 = [
    4.95615217827958e-06,
    7.04116158378997e-05,
    0.001649626864837,
    0.000311470440329,
    0.001221801819949,
    0.002979836024774,
    0.025344376042616,
    0.043221392700960,
    0.054390169674924,
]

y1 = [
    854.977129435539,
    562.457083315320,
    336.168218739828,
    277.474167733722,
    237.551734937562,
    209.814468701681,
    183.584001183510,
    139.415717000432,
    120.803398238288,
]


fig = plt.figure()
ax1 = plt.gca()
ax1.scatter(x, y0)
plt.ylabel("Objective")
plt.xlabel("Fibers")
ax1.set_yscale("log")

ax1.set_title("AdaCPD Number of Fibers Performance\n b0=.25, Size=200, Rank=50")


ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("MTTKR Time", color=color)
ax2.scatter(x, y1, color=color)
ax2.set_yscale("log")

plt.show()
plt.savefig("numfibersobjective.png")
