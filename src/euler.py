import os
import threading
from pylab import *
from numpy import *
from scipy.integrate import ode
from scipy.interpolate import griddata

from mesh import *

def gask(W, gamma=1.4):
    '''
    Returns kinetic energy q, pressure p, velocity u and sound speed c
    '''
    u = W[:,1:3] / W[:,:1]           # velocity

    #q = 0.5 * W[:,0] * sum(u**2,1)   # kinetic energy
    q = dot_each(u, u); q *= W[:,0]; q*= 0.5; #almost twice as fast

    p = (W[:,3] - q); p *= (gamma - 1)   # pressure

    #c = sqrt(gamma * p / W[:,0] )     # speed of sound
    c = p/W[:,0]; c*=gamma; sqrt(c, out=c)

    return q, p, u, c

def jacConsSym(W, n=None, gamma=1.4):
    '''
    Returns S, T, (L)
    S: Jacobian transforming conservative vars [d_rho, d_rhoU, d_rhoV, d_rhoE]
       to symmetrizable vars [d_p/(rho c), d_u, d_v, d_p - c^2 d_rho]
    T: Jacobian transforming symmetrizable vars to conservative vars
    L: symmetric Jacobian (only if n is not None)
    '''
    q, p, u, c = gask(W, gamma)
    rhoC = W[:,0] * c
    rhoOverC = W[:,0] / c
    rhoH = W[:,3] + p
    c2 = c**2

    S = zeros(W.shape[:1] + (4, 4))
    S[:,0,0] = (gamma - 1) * q / W[:,0] / rhoC
    S[:,0,1:3] = -(gamma - 1) * u / rhoC[:,newaxis]
    S[:,0,3] = (gamma - 1) / rhoC
    S[:,1:3,0] = -u / W[:,:1]
    S[:,[1,2],[1,2]] = 1. / W[:,:1]
    S[:,3,0] = (gamma - 1) * q / W[:,0] - c**2
    S[:,3,1:3] = -(gamma - 1) * u
    S[:,3,3] = (gamma - 1)

    T = zeros(W.shape[:1] + (4, 4))
    T[:,0,0] = rhoOverC
    T[:,0,3] = -1 / c2
    T[:,1:3,0] = rhoOverC[:,newaxis] * u
    T[:,[1,2],[1,2]] = W[:,:1]
    T[:,1:3,3] = -u / c2[:,newaxis]
    T[:,3,0] = rhoH / c
    T[:,3,1:3] = W[:,1:3]
    T[:,3,3] = -q / W[:,0] / c2

    if n is not None:
        uDotN = (u * n).sum(1)
        cN = c[:,newaxis] * n
        # symmetric Jacobian
        L = zeros([W.shape[0], 4, 4])
        L[:,r_[:4], r_[:4]] = uDotN[:,newaxis]
        L[:,1:3, 0] = cN
        L[:,0, 1:3] = cN
        return S, T, L
    else:
        return S, T

def fluxE(WL, WR, n):
    assert WL.shape == WR.shape == n.shape[:1] + (4,)
    WE = 0.5 * (WL + WR)
    q, p, u, c = gask(WE)
    uDotN = dot_each(u, n) #(u * n).sum(1)
    flux = zeros(WL.shape)
    flux[:,0] = WE[:,0] * uDotN
    flux[:,1:3] = WE[:,1:3] * uDotN[:,newaxis] + p[:,newaxis] * n
    flux[:,3] = (WE[:,3] + p) * uDotN
    return flux

def jacE(WE, n):
    assert WE.shape == n.shape[:1] + (4,)
    S, T, L = jacConsSym(WE, n)
    JE = matrixMult(T, L, S)
    return JE

def specRad(WE, n, gamma=1.4):
    q, p, u, c = gask(WE, gamma)
    uDotN = dot_each(u, n) #(u * n).sum(1)
    cNrmN = c * sqrt( dot_each(n,n) )#sqrt((n**2).sum(1))
    return (absolute(uDotN) + cNrmN)

def fluxD(WL, WR, dWL, dWR, n, HiRes=0):
    assert WL.shape == WR.shape == n.shape[:1] + (4,)
    A = specRad(0.5 * (WL + WR), n)[:,newaxis]
    dW = WR - WL
    flux = -0.5 * dW * A
    # minmod type limiter
    if HiRes > 0:
        # remove contribution of dW from cell gradients dWL and dWR,
        # which are computed as averages of edge gradients
        dWL, dWR = 1.5 * dWL - 0.5 * dW, 1.5 * dWR - 0.5 * dW
        limSign = (dWL * dW > 0) * (dWR * dW > 0) * sign(dW)
        limiter = limSign * minimum(absolute(dW), minimum(absolute(dWR), absolute(dWL)))
        flux += 0.5 * limiter * A * HiRes
    return flux

def wallBc(W, n):
    assert W.shape == n.shape[:1] + (4,)
    q, p, u, c = gask(W)
    flux = zeros(W.shape)
    flux[:,1:3] = p[:,newaxis] * n
    return flux

def farBc(W, W0, n):
    assert W.shape == n.shape[:1] + (4,)
    return fluxE(W0, W, n) + fluxD(W0, W, 0, 0, n)


class Euler:
    def __init__(self, v, t, b, Mach, HiRes=.9):
        self.mesh = Mesh(v, t, b)
        self.Mach = float(Mach)
        self.HiRes = float(HiRes)
        assert 0 <= self.HiRes <= 1
        self.ode = ode(lambda t,W: self.ddt(W))
        self.ode.set_integrator('dopri5', nsteps=10000, rtol=1e-2, atol=1e-5)

        self.solnLock = threading.Lock()

    def integrate(self, t, W0=None, t0=None):
        if W0 is not None:
            if t0 is None: t0 = 0
            self.ode.set_initial_value(ravel(W0), t0)
        self.ode.integrate(t)
        self.setSolnCache(self.time, self.soln)
        return self.soln

    @property
    def nt(self):
        return self.mesh.nt

    @property
    def time(self):
        return self.ode.t

    @property
    def soln(self):
        return reshape(self.ode.y, [-1,4])

    @property
    def Wref(self):
        Wr = self.freeStream(1)[0]
        Wr[2] = Wr[1]
        return Wr

    def freeStream(self, nT=None):
        if nT is None: nT = self.nt
        R, gamma, rho0, T0 = 8.314 / 29E-3, 1.4, 1.225, 288.75
        u0 = self.Mach * sqrt(gamma * R * T0)
        E0 = T0 * R / (gamma - 1) + 0.5 * u0**2
        return array([rho0, rho0 * u0, 0, rho0 * E0]) + zeros([nT, 1])

    def ddt(self, W):
        assert W.size == self.nt * 4
        shp = W.shape
        W = W.reshape([-1, 4])
        m = self.mesh
        gradW = m.gradTri(W)

        xt = m.xt()
        dxt = m.dxt #xt[m.e[:,3],:] - xt[m.e[:,2],:]
        # interior flux

        #WL, WR = W[m.e[:,2],:], W[m.e[:,3],:]
        WL, WR = m.leftRightTri(W) #abount twice as fast

        #dWL = (gradW[m.e[:,2],:] * dxt[:,:,newaxis]).sum(1)
        #dWR = (gradW[m.e[:,3],:] * dxt[:,:,newaxis]).sum(1)
        gWL, gWR = m.leftRightTri(gradW) #about 3 times as fast
        dWL = einsum( 'nij, ni -> nj', gWL, dxt )
        dWR = einsum( 'nij, ni -> nj', gWR, dxt )

        flux = fluxE(WL, WR, m.n) + fluxD(WL, WR, dWL, dWR, m.n, self.HiRes)
        # boundary condition categories
        xeBnd = m.v[m.e[m.ieBnd,:2],:].mean(1)
        isWall = ~m.isFar(xeBnd)
        # boundary flux
        WBnd, nBnd = WL[m.ieBnd,:], m.n[m.ieBnd,:]
        flux[m.ieBnd[isWall]] = wallBc(WBnd[isWall,:], nBnd[isWall,:])
        Wfar = self.freeStream((~isWall).sum())
        flux[m.ieBnd[~isWall]] = farBc(WBnd[~isWall,:], Wfar, nBnd[~isWall,:])
        # accumunate flux to cells
        return (m.distributeFlux(flux) / m.a[:,newaxis]).reshape(shp)

    def setSolnCache(self, t, update):
        'Thread safe method, update the solution copy'
        self.solnLock.acquire()
        self._soln_t = t
        self._soln_copy = update.copy()
        self.solnLock.release()

    def getSolnCache(self):
        'Thread safe method, return the previous copy of the solution'
        self.solnLock.acquire()
        _soln_t_copy = self._soln_t
        _soln_copy_copy = self._soln_copy.copy()
        self.solnLock.release()
        return _soln_t_copy, _soln_copy_copy

    def metric(self, W=None):
        if W is None: W = self.soln
        gradV = self.mesh.gradTriVrt(W / self.Wref)
        gradT = self.mesh.gradTri(W / self.Wref)
        hessian = self.mesh.gradTriVrt(gradT)
        hessian += hessian.transpose([0,2,1,3])
        # return (hessian + gradV[:,newaxis,:,:] * gradV[:,:,newaxis,:]).sum(-1)
        return hessian.sum(-1)

