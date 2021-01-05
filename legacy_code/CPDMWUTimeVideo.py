from CPDMWUTime import CPDMWUTime
from Utils import save_trial_data, createTensor, initDecomposition, videoToTensor
import time
import pickle
import os
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from math import sqrt, log

randon.seed(0)

fiberPropotions = [.001, .005, .01, .05, .1, .25, .5, 1]

Ranks = [100,150,200]

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
                arrangements.append(([fiberPropotion], Size, trial, Rank, maxtime, dir))
for Rank in Ranks:
    for Size in Sizes:
        for trial in range(Trials):
            arrangements.append((fiberPropotions, Size, trial, Rank, maxtime, dir))

def saveCPDTimeTrial(X, A,B,C,fiberPropotion,lamb, eps, nu, Size, trial, Rank, max_time, error,rates,dir):
    filename = '{}/CPDResults_{}_{}_{}_{}_{}.dat'.format(dir,fiberPropotion, Size, trial, Rank, max_time)
    results = {
        'A':A, 'B':B, 'C':C, 'fiberPropotion':fiberPropotion, 'Size':Size, 'trial':trial, 'Rank':Rank, 'lamb':lamb, 'eps':eps, 'nu':nu, 'max_time':max_time, 'error':error, 'rates':rates
    }
    with open(filename, 'wb') as f:
        pickle.dump(results, f)


def runTest(conf):
    fiberPropotion, Size, trial, Rank, max_time,dir = conf
    print("Running trial with the following:\n\tProporion of Fibers = {}\n\tSize = {}\n\tRank = {}\n\tTrialNumber = {}\n\tMax Time = {}".format(fiberPropotion, Size, Rank, trial, maxtime))

    #Create tensor
    X = videoToTensor('600Test.mp4')

    #init starting
    A_init = initDecomposition(Size,Rank)

    #X, F, sketching_rates, lamb, eps, nu, Hinit, max_time ,sample_interval=.5
    eps = 1/(1.2len(fiberPropotion))
    nu = sqrt(2 * log(len(fiberPropotion))/50)
    lamb = .001

    A,B,C,error,rates = CPDMWUTime(X, Rank, fiberPropotion, lamb, eps, nu, A_init, max_time,sample_interval=.5)
    saveCPDTimeTrial(X,A,B,C, fiberPropotion, lamb, eps, nu, Size, trial, Rank, max_time, error,rates,dir)

for i in arrangements:
    runTest(i)