import os
from pylab import *
from numpy import *
from scipy.integrate import ode
from scipy.interpolate import griddata

from mesh import *
from euler import *
from trajectory import *

def gaskAdj(W, Qq=None, Qp=None, Qu=None, Qc=None, gamma=1.4):
    Q = zeros(W.shape)
    if Qq is None: Qq = zeros(W.shape[0])

    if Qp is not None:
        Q[:,3] += (gamma - 1) * Qp
        Qq -= (gamma - 1) * Qp

    if Qq is not None:
        u = W[:,1:3] / W[:,:1]           # velocity
        Q[:,1:3] += u * Qq[:,newaxis]
        Q[:,0] -= 0.5 * (u**2).sum(1) * Qq

    assert Qc is None and Qu is None     # not implemented yet
    return Q

def cellAvg(mesh, value, isAdjoint=False):
    if not mesh.__dict__.has_key('matCellAvg'):
        mat2 = accumarray(mesh.e[:,2], mesh.t.shape[0]).mat
        mat3 = accumarray(mesh.e[:,3], mesh.t.shape[0]).mat
        mesh.matCellAvg = 0.5 * (mat2 + mat3)
    if isAdjoint:
        return mesh.matCellAvg * value
    else:
        return mesh.matCellAvg.T * value

def wallBcAdj(W, Qflux, n):
    assert W.shape == Qflux.shape == n.shape[:1] + (4,)
    Qp = (Qflux[:,1:3] * n).sum(1)
    return gaskAdj(W, Qp=Qp)

def farBcAdj(W, W0, Qflux, n):
    assert W.shape == Qflux.shape == n.shape[:1] + (4,)
    WE = 0.5 * (W + W0)
    QE = (jacE(WE, n) * Qflux[:,:,newaxis]).sum(1)
    return 0.5 * QE

def fluxDsemiAdj(WE, Qflux, n):
    assert WE.shape == n.shape[:1] + (4,)
    A = specRad(WE, n)[:,newaxis]
    return 0.5 * A * Qflux


class EulerAdj:
    def __init__(self, euler, traj):
        self.euler = euler
        self.traj = traj
        self.mesh = euler.mesh
        self.Mach = euler.Mach
        self.HiRes = euler.HiRes
        assert 0 <= self.HiRes <= 1
        self.ode = ode(lambda t,Q: -self.ddt(self.traj(-t), Q))
        self.ode.set_integrator('dopri5', nsteps=10000, rtol=1e-2, atol=1e-5)

    def integrate(self, t, Q0=None, t0=None):
        if Q0 is not None:
            if t0 is None: t0 = self.traj.tlim[1]
            self.ode.set_initial_value(ravel(Q0), -t0)
        self.ode.integrate(-t)
        return self.soln

    @property
    def nt(self):
        return self.euler.nt

    @property
    def time(self):
        return -self.ode.t

    @property
    def soln(self):
        return reshape(self.ode.y, [-1,4])

    def ddt(self, W, Q):
        assert Q.size == W.size == self.nt * 4
        shp = Q.shape
        Q, W = Q.reshape([-1, 4]), W.reshape([-1, 4])
        m = self.mesh
        # flow quantities that needs to be computed for adjoint
        WL, WR = W[m.e[:,2],:], W[m.e[:,3],:]
        WE = 0.5 * (WL + WR)
        # cell adjoint to flux adjoint
        Qflux = m.distributeFlux(Q, isAdjoint=True)
        # boundary condition categories
        xeBnd = m.v[m.e[m.ieBnd,:2],:].mean(1)
        isWall = ~m.isFar(xeBnd)
        # boundary flux adjoint
        QfluxBnd, WLBnd, nBnd = Qflux[m.ieBnd], WL[m.ieBnd,:], m.n[m.ieBnd]
        dQdtBnd = zeros(QfluxBnd.shape)
        dQdtBnd[isWall] = wallBcAdj(WLBnd[isWall], QfluxBnd[isWall],
                                    nBnd[isWall])
        Wfar = self.euler.freeStream((~isWall).sum())
        dQdtBnd[~isWall] = farBcAdj(WLBnd[~isWall], Wfar, QfluxBnd[~isWall],
                                    nBnd[~isWall])
        # interior flux adjoint
        Qflux[m.ieBnd] = 0
        QE = (jacE(WE, m.n) * Qflux[:,:,newaxis]).sum(1)
        dQdt = m.interpTri2Edg(QE, isAdjoint=True)
        # numerical dissipation flux adjoint
        dQdt -= m.distributeFlux(fluxDsemiAdj(WE, Qflux, m.n))
        dQdt[m.e[m.ieBnd,3]] += dQdtBnd
        return (dQdt / m.a[:,newaxis]).reshape(shp)

    def checkAdj(self):
        '''
        Generate a flow with small linear variation (supress numerical diss.),
        check adjoint ddt vs Euler ddt
        '''
        xt = self.mesh.xt()
        R = 0.1 * self.mesh.diameter()
        decay = exp(-(xt**2).sum(1) / R**2)[:,newaxis]
        # random flow field
        randCoef = 0.3 * (random.rand(1,2,4) - .5) / self.mesh.diameter()
        variation = (xt[:,:,newaxis] * randCoef).sum(1) * decay
        W0 = self.euler.Wref * (1 + variation)
        # random perturbation
        EPS = 0.00001
        randCoef = EPS * (random.rand(1,2,4) - .5) / self.mesh.diameter()
        variation = (xt[:,:,newaxis] * randCoef).sum(1) * decay
        W1 = W0 + self.euler.Wref * variation
        # random adjoint field
        variation = (random.rand(*W0.shape) - 0.5) * decay
        variation = (random.rand(*W0.shape) - 0.5)
        Q = 1/EPS / self.euler.Wref * variation
        # compute ddt
        dW0dt = self.euler.ddt(W0)
        dW1dt = self.euler.ddt(W1)
        dQdt = self.ddt(0.5 * (W0 + W1), Q)
        dW, ddWdt = W1 - W0, dW1dt - dW0dt
        # compare integrals
        integ1 = (ddWdt * Q * self.mesh.a[:,newaxis]).sum()
        integ2 = (dQdt * dW * self.mesh.a[:,newaxis]).sum()
        print 'checkAdj: ', integ1-integ2, integ1, integ2
        return Q, dW, dQdt, ddWdt


if __name__ == '__main__':
    geom, nE = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]], 2500
    Mach = 0.3
    HiRes = 0.
    v, t, b = initMesh(geom, nE)
    solver = Euler(v, t, b, Mach, HiRes)
    adj = EulerAdj(solver, None)
    for i in range(8):
        Q, dW, dQdt, ddWdt = adj.checkAdj()
