import os
from pylab import *
from numpy import *
from scipy.integrate import ode
from scipy.interpolate import griddata

from mesh import *
from euler import *

def fluxV(UE, gradUE, n):
    '''
    Viscous flux in NS equation
    '''
    assert UE.shape == n.shape[:1] + (2,)
    assert gradUE.shape == n.shape[:1] + (2,2)
    flux = zeros([gradUE.shape[0], 4])
    f1 = (gradUE * n[:,:,newaxis]).sum(1) \
       + (gradUE * n[:,newaxis,:]).sum(2)
    f2 = gradUE[:,[0,1],[0,1]].sum(1)[:,newaxis] * n
    # momentum viscous flux
    flux[:,1:3] = -f1 + 2/3. * f2
    # energy viscous flux, ignore conduction
    flux[:,3] = (flux[:,1:3] * UE).sum(1)
    return flux

class NavierStokes(Euler):
    def __init__(self, v, t, b, Mach, Re, HiRes=.9):
        Euler.__init__(self, v, t, b, Mach, HiRes)
        # compute viscosity from Reynolds number
        self.mu = self.Wref[1] / Re

    def ddt(self, W):
        ddtEuler = Euler.ddt(self, W)
        # Navier Stokes terms
        assert W.size == self.nt * 4
        shp = W.shape
        W = W.reshape([-1, 4])
        m = self.mesh
        q, p, u, c = gask(W)
        # velocity is 0 at wall, copy cell velocity at far field
        uBc = zeros([m.ieBnd.size, 2])
        xeBnd = m.v[m.e[m.ieBnd,:2],:].mean(1)
        isFar = m.isFar(xeBnd)
        ieFar = m.ieBnd[isFar]
        uBc[isFar,:] = u[m.e[ieFar,3],:]
        # compute and accumulate viscous flux
        uE = 0.5 * (u[m.e[:,2],:] + u[m.e[:,3],:])
        graduE = m.gradTriEdg(u, uBc)
        flux = self.mu * fluxV(uE, graduE, m.n)
        ddtVisc = (m.distributeFlux(flux) / m.a[:,newaxis]).reshape(shp)
        return ddtEuler + ddtVisc

    def metric(self, W=None):
        '''
        Hessian of NS flow is different from Euler flow at wall
        '''
        if W is None: W = self.soln
        m = self.mesh
        # set wall values
        Wbnd = W[m.e[m.ieBnd,3],:]
        xeBnd = m.v[m.e[m.ieBnd,:2],:].mean(1)
        isWall = ~m.isFar(xeBnd)
        Wbnd[isWall,1:3] = 0
        # freestream reference
        grad = self.mesh.gradTriVrt(W / self.Wref, Wbnd / self.Wref)
        return (grad[:,newaxis,:,:] * grad[:,:,newaxis,:]).sum(-1)

