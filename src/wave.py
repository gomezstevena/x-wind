import os
from pylab import *
from numpy import *
from scipy.integrate import odeint

from mesh import *

class Wave:
    def __init__(self, v, t, b):
        self.mesh = Mesh(v, t, b)

        m = self.mesh
        xeBnd = m.v[m.e[m.ieBnd,:2],:].mean(1)
        self.phiBnd = array((xeBnd**2).sum(1) < 10, float)

    @property
    def nt(self):
        return self.mesh.t.shape[0]

    def fluxE(self, phiL, phiR, n):
        assert phiL.shape == phiR.shape == n.shape[:1] + (3,)
        phiE = 0.5 * (phiL + phiR)
        flux = zeros(phiL.shape)
        flux[:,0] = (phiE[:,1:] * n).sum(1)
        flux[:,1:] = phiE[:,:1] * n
        return flux

    def fluxD(self, phiL, phiR, n):
        assert phiL.shape == phiR.shape == n.shape[:1] + (3,)
        flux = 0.5 * (phiL - phiR) * sqrt((n**2).sum(1))[:,newaxis]
        return flux

    def ddt(self, phi, t=0):
        assert phi.size == self.nt * 3
        shp = phi.shape
        phi = phi.reshape([-1, 3])
        m = self.mesh
        phiL, phiR = phi[m.e[:,2],:], phi[m.e[:,3],:]
        flux = self.fluxE(phiL, phiR, m.n) + self.fluxD(phiL, phiR, m.n)
        # boundary condition
        phiBnd = phiL[m.ieBnd,:].copy()
        phiBnd[:,0] = self.phiBnd
        flux[m.ieBnd] = self.fluxE(phiBnd, phiR[m.ieBnd,:], m.n[m.ieBnd]) \
                      + self.fluxD(phiBnd, phiR[m.ieBnd,:], m.n[m.ieBnd])
        # accumunate to cells
        flux = vstack([-flux, flux, flux[m.ieBnd]])
        indx = hstack([m.e[:,2], m.e[:,3], m.e[m.ieBnd,2]])
        return (accumarray(indx)(flux) / m.a[:,newaxis]).reshape(shp)


if __name__ == '__main__':
    geom = [[1, 0], [0.3, 0], [0, 0.1], [-0.3, 0], [-1, 0], [-0.4, -0.5],
            [-0.7, -1.2], [0, -0.1], [0.7, -1.2], [0.4, -0.5], [1, 0]]
    nE = 5000
    v, t, b = initMesh(geom, nE)
    solver = Wave(v, t, b)

    T = linspace(0, 1, 1001)
    phi = odeint(solver.ddt, zeros(solver.nt*3), T)

    if not os.path.exists('fig'):
        os.mkdir('fig')
    figure()
    for i in arange(0,1001,100):
        clf()
        solver.mesh.plotTriScalar(phi[i].reshape([-1,3])[:,0])
        axis([-2,2,-2,1])
        savefig('fig/wave_init_{0:06d}.png'.format(i))
    solver.mesh.plotMesh()

    for iAdapt in range(3):
        print 'Adapt cycle {0}'.format(iAdapt)

        # compute metric
        metric = zeros([v.shape[0], 2, 2])
        for iStep in range(phi.shape[0]/2, phi.shape[0]):
            phiStep = phi[iStep].reshape([-1, 3])
            for i in range(3):
                phiGrad = solver.mesh.gradTriVrt(phiStep[:,i])
                metric += phiGrad[:,newaxis,:] * phiGrad[:,:,newaxis]
    
        # adapt and solve
        v, t, b = adaptMesh(geom, v, t, b, nE, metric)
        solver = Wave(v, t, b)
    
        phi = odeint(solver.ddt, zeros(solver.nt*3), T)

        figure()
        for i in arange(0,1001,100):
            clf()
            solver.mesh.plotTriScalar(phi[i].reshape([-1,3])[:,0])
            axis([-2,2,-2,1])
            savefig('fig/wave_adapt{1:1d}_{0:06d}.png'.format(i, iAdapt))
        solver.mesh.plotMesh()
