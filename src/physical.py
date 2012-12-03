from numpy import *
from euler import gask

def physical(W, name):
    FUNC_MAP = { \
        'density' : computeRho, \
        'X-velocity' : computeVX, \
        'Y-velocity' : computeVY, \
        'static pressure' : computeP, \
        'stagnation pressure' : computePt, \
        'mach number' : computeMach, \
        'entropy' : computeEntropy \
    }
    assert name in FUNC_MAP.keys()
    return FUNC_MAP[name](W)


def computeRho(W):
    return W[:,0]

def computeVX(W):
    return W[:,1] / W[:,0]

def computeVY(W):
    return W[:,2] / W[:,0]

def computeP(W):
    q, p, u, c = gask(W)
    return p

def computePt(W):
    q, p, u, c = gask(W)
    M = computeMach(W)
    return p * (1 + 0.2 * M*M)**(7./2)

def computeMach(W):
    q, p, u, c = gask(W)
    return sqrt((u**2).sum(1)) / c

def computeEntropy(W):
    q, p, u, c = gask(W)
    nR = 8.314 / 29E-3
    return (5./2 * log(p) - 7./2 * log(W[:,0])) * nR
