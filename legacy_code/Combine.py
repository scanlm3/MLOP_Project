import numpy as np
import matplotlib.pyplot as plt
from AdaCPD import AdaCPD
from Utils import save_trial_data, videoToTensor, synteticTensor, tensorToVideo
import cv2
from CPDMWU import CPD_MWU

import tensorly as tl
import tensorly.tenalg as tl_alg

F = 100

# X, _ = synteticTensor([100,100,100], F)

X = videoToTensor("testvideo")

X = X / 256

Hinit = []
for d in range(3):
    Hinit.append(np.random.random((X.shape[d], F)))

iters = 30


sketching_rates = list(np.linspace(10 ** (-3), 10 ** (-1), 4)) + [1]

lamb = 0.001
shape = (300, 300, 300)
nu = 2

A, B, C, error, res_time = CPD_MWU(
    X, F, sketching_rates, lamb, 0.0001, nu, Hinit, mttkrps=3
)

Hinit = [A, B, C]

time_A, NRE_A, MSE_A, A = AdaCPD(X, 1, 100, 10, Hinit, None)

reconstructed = PP = tl.kruskal_to_tensor((np.ones(F), A))

tensorToVideo(reconstructed, "reconstruced")
