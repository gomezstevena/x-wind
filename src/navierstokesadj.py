import os
from pylab import *
from numpy import *
from scipy.integrate import ode
from scipy.interpolate import griddata

from mesh import *
from navierstokes import *
from euleradj import *

def gradTriEdgAdj(mesh, QgradPhi):
    '''
    Implements the adjoint of gradTriEdg in meshGradients.py
    Input should be of dimension (nE, 2, m) or (nE, 2)
    '''
    shape = QgradPhi.shape[2:]   # (m,) or (), for shaping the output
    QgradPhi = QgradPhi.reshape([mesh.e.shape[0] * 2, -1])
    Qphi = mesh.matGradTriEdg.T * QgradPhi
    QphiBc = mesh.bcmatGradTriEdg.T * QgradPhi
    return Qphi.reshape((-1,) + shape), QphiBc.reshape((-1,) + shape)

def fluxVAdj(UE, gradUE, Qflux, n):
    '''
    Viscous flux in NS equation
    '''
    assert UE.shape == n.shape[:1] + (2,)
    assert Qflux.shape == n.shape[:1] + (4,)
    assert gradUE.shape == n.shape[:1] + (2,2)
    # compute flux[:,1:3]
    flux = zeros([gradUE.shape[0], 4])
    f1 = (gradUE * n[:,:,newaxis]).sum(1) \
       + (gradUE * n[:,newaxis,:]).sum(2)
    f2 = gradUE[:,[0,1],[0,1]].sum(1)[:,newaxis] * n
    # momentum viscous flux
    flux[:,1:3] = -f1 + 2/3. * f2
    # energy viscous flux, ignore conduction
    QUE = Qflux[:,3:] * flux[:,1:3]
    Qf1 = -(Qflux[:,1:3] + Qflux[:,3:] * UE)
    # momentum viscous flux
    QgradUE = Qf1[:,newaxis,:] * n[:,:,newaxis] \
            + Qf1[:,:,newaxis] * n[:,newaxis,:]
    QgradUE[:,[0,1],[0,1]] -= 2/3. * (Qf1 * n).sum(1)[:,newaxis]
    return QUE, QgradUE


class NavierStokesAdj(EulerAdj):
    def __init__(self, navierstokes, traj):
        EulerAdj.__init__(self, navierstokes, traj)
        self.navierstokes = navierstokes   # alias to "euler"
        self.mu = navierstokes.mu

    def ddt(self, W, Q):
        ddtEulerAdj = EulerAdj.ddt(self, W, Q)
        # Viscous flux adjoint calculation
        assert Q.size == W.size == self.nt * 4
        shp = Q.shape
        Q, W = Q.reshape([-1, 4]), W.reshape([-1, 4])
        m = self.mesh
        # flow quantities that needs to be computed for adjoint
        u = W[:,1:3] / W[:,:1]
        uE = 0.5 * (u[m.e[:,2],:] + u[m.e[:,3],:])
        # velocity is 0 at wall, copy cell velocity at far field
        uBc = zeros([m.ieBnd.size, 2])
        xeBnd = m.v[m.e[m.ieBnd,:2],:].mean(1)
        isFar = m.isFar(xeBnd)
        ieFar = m.ieBnd[isFar]
        uBc[isFar,:] = u[m.e[ieFar,3],:]
        graduE = m.gradTriEdg(u, uBc)
        # cell adjoint to flux adjoint
        Qflux = m.distributeFlux(Q, isAdjoint=True)
        QuE, QgraduE = fluxVAdj(uE, graduE, self.mu * Qflux, m.n)
        # adjoint the edge average and edge gradient
        dQudt, dQudtBc = gradTriEdgAdj(m, QgraduE)
        ieFar = m.ieBnd[isFar]
        dQudt[m.e[ieFar,3]] += dQudtBc[isFar]
        dQudt += cellAvg(m, QuE, isAdjoint=True)
        dQdt = zeros(Q.shape)
        dQdt[:,1:3] = dQudt / W[:,:1]
        dQdt[:,0] = -(dQudt * u).sum(1) / W[:,0]
        return (dQdt / m.a[:,newaxis]).reshape(shp) + ddtEulerAdj


if __name__ == '__main__':
    geom, nE = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]], 1000
    Mach = 0.3
    Re = 50.
    HiRes = 0.
    v, t, b = initMesh(geom, nE)
    solver = NavierStokes(v, t, b, Mach, Re, HiRes)
    adj = NavierStokesAdj(solver, None)
    for i in range(8):
        Q, W, dWdt, dQdt = adj.checkAdj()
