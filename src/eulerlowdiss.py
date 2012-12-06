import os
from pylab import *
from numpy import *
from scipy.sparse import kron as spkron

from mesh import *
from integrator import Ode
from euler import gask, jacGask, wallBc, jacWall, Euler

def fluxLowDiss(WL, WR, n):
    assert WL.shape == WR.shape == n.shape[:1] + (4,)
    qL, pL, uL, cL = gask(WL)
    qR, pR, uR, cR = gask(WR)
    uE, pE, WE = 0.5 * (uL + uR), 0.5 * (pL + pR), 0.5 * (WL + WR)
    uDotN = (uE * n).sum(1)
    flux = WE * uDotN[:,newaxis]
    flux[:,1:3] += pE[:,newaxis] * n
    flux[:,3] += pE * uDotN
    return flux

def jacLowDiss(WL, WR, n):
    assert WL.shape == WR.shape == n.shape[:1] + (4,)
    qL, pL, uL, cL = gask(WL)
    qR, pR, uR, cR = gask(WR)
    qL_WL, pL_WL, uL_WL, cL_WL = jacGask(WL)
    qR_WR, pR_WR, uR_WR, cR_WR = jacGask(WR)
    uE, pE, WE = 0.5 * (uL + uR), 0.5 * (pL + pR), 0.5 * (WL + WR)
    uDotN = (uE * n).sum(1)
    uDotN_WL = 0.5 * (uL_WL * n[:,:,newaxis]).sum(1)
    uDotN_WR = 0.5 * (uR_WR * n[:,:,newaxis]).sum(1)
    # flux = WE * uDotN[:,newaxis]
    flux_WL = 0.5 * uDotN[:,newaxis,newaxis] * eye(4) \
            + WE[:,:,newaxis] * uDotN_WL[:,newaxis,:]
    flux_WR = 0.5 * uDotN[:,newaxis,newaxis] * eye(4) \
            + WE[:,:,newaxis] * uDotN_WR[:,newaxis,:]
    # flux[:,1:3] += pE[:,newaxis] * n
    flux_WL[:,1:3,:] += 0.5* pL_WL[:,newaxis,:] * n[:,:,newaxis]
    flux_WR[:,1:3,:] += 0.5* pR_WR[:,newaxis,:] * n[:,:,newaxis]
    # flux[:,3] += pE * uDotN
    flux_WL[:,3,:] += 0.5* pL_WL * uDotN[:,newaxis] + pE[:,newaxis] * uDotN_WL
    flux_WR[:,3,:] += 0.5* pR_WR * uDotN[:,newaxis] + pE[:,newaxis] * uDotN_WR
    return flux_WL, flux_WR

class EulerLowDiss(Euler):
    def __init__(self, v, t, b, Mach):
        self.mesh = Mesh(v, t, b)
        self.Mach = float(Mach)
        self.ode = Ode(self.ddt, self.J)

    def ddt(self, W):
        assert W.size == self.nt * 4
        shp = W.shape
        W = W.reshape([-1, 4])
        m = self.mesh
        gradW = m.gradTri(W)
        xt = m.xt(); dxt = xt[m.e[:,3],:] - xt[m.e[:,2],:]
        # boundary condition categories
        xeBnd = m.v[m.e[m.ieBnd,:2],:].mean(1)
        isWall = ~m.isFar(xeBnd)
        # interior and far field flux
        WL, WR = W[m.e[:,2],:], W[m.e[:,3],:]
        WL[m.ieBnd[~isWall]] = self.freeStream(1)
        flux = fluxLowDiss(WL, WR, m.n)
        # boundary flux
        WBnd, nBnd = WL[m.ieBnd,:], m.n[m.ieBnd,:]
        flux[m.ieBnd[isWall]] = wallBc(WBnd[isWall,:], nBnd[isWall,:])
        # accumunate flux to cells
        return (m.distributeFlux(flux) / m.a[:,newaxis]).reshape(shp)

    def J(self, W):
        if not self.__dict__.has_key('matJacDistFlux'):
            self.prepareJacMatrices()
        assert W.size == self.nt * 4
        shp = W.shape
        W = W.reshape([-1, 4])
        m = self.mesh
        # prepare values at edge
        WL, WR = W[m.e[:,2],:], W[m.e[:,3],:]
        isFar = m.isFar(m.v[m.e[m.ieBnd,:2],:].mean(1))
        WL[m.ieBnd[isFar]] = self.freeStream(1)
        # jacobian computation
        flux_WL, flux_WR = jacLowDiss(WL, WR, m.n)
        # modify for wall BC
        ieWall = m.ieBnd[~isFar]
        flux_WL[ieWall,:] = jacWall(WL[ieWall], m.n[ieWall])
        flux_WR[ieWall,:] = 0
        J_flux = block_diags(flux_WL) * self.matJacWL \
               + block_diags(flux_WR) * self.matJacWR
        return self.matJacDistFlux * J_flux


if __name__ == '__main__':
    geom = loadtxt('../data/n0012c.dat')
    geom = rotate(geom, 10*pi/180)
    nE = 1000
    dt = 0.0001
    Mach = 0.4
    
    v, t, b = initMesh(geom, nE)
    solver = EulerLowDiss(v, t, b, Mach)
    for i in range(8):
        solver.checkJacobian()
