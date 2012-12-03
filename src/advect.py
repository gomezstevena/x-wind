import time
from pylab import *
from numpy import *
from scipy.integrate import odeint

from mesh import *

class Advect:
    def __init__(self, v, t, b, U):
        self.mesh = Mesh(v, t, b)

        self.U = array(U).copy()
        assert self.U.ndim == 1 and self.U.size == 2

    @property
    def nt(self):
        return self.mesh.t.shape[0]

    def ddt(self, phi, t=0):
        assert phi.ndim == 1 and phi.size == self.nt
        m = self.mesh
        phiL, phiR = phi[m.e[:,2]], phi[m.e[:,3]]
        fluxE = 0.5 * (phiL + phiR) * (m.n * self.U).sum(1)
        fluxD = 0.5 * (phiL - phiR) * abs((m.n * self.U).sum(1))
        fluxE += fluxD
        # boundary condition
        iBnd = (m.e[:,2] == m.e[:,3]).nonzero()[0]
        iBndIn = iBnd[(m.n[iBnd,:] * self.U).sum(1) > 0]
        xe = m.v[m.e[:,:2],:].mean(1)
        phiBnd = array((xe[iBndIn]**2).sum(1) < 10, float)
        fluxE[iBndIn] = phiBnd * (m.n[iBndIn,:] * self.U).sum(1)
        # accumunate to cells
        flux = hstack([-fluxE, fluxE, fluxE[iBnd]])
        indx = hstack([m.e[:,2], m.e[:,3], m.e[iBnd,2]])
        #return accumarray(indx)(flux) / m.a
        return bincount(indx, flux) / m.a


if __name__ == '__main__':
    geom = [[1, 0], [1, 1], [0, 1], [-1, 0], [-1, -1], [0, -1], [1, 0]]
    nE = 1000
    v, t, b = initMesh(geom, nE)
    solver = Advect(v, t, b, [.8,.6])

    T = linspace(0, 5, 201)
    phi = odeint(solver.ddt, zeros(solver.nt), T)

    figure()
    solver.mesh.plotTriScalar(phi[-1])
    solver.mesh.plotMesh(alpha=0.2)
    axis([-2,5,-2,5])

    matplotlib.interactive(True)
    draw()
    time.sleep(0.1)

    for iAdapt in range(3):
        print 'Adapt cycle {0}'.format(iAdapt)

        # compute metric on vertices
        metric = zeros([v.shape[0], 2, 2])
        for iStep in range(phi.shape[0]):
            phiGrad = solver.mesh.gradTriVrt(phi[iStep])
            metric += phiGrad[:,newaxis,:] * phiGrad[:,:,newaxis]
    
        # adapt and solve
        v, t, b = adaptMesh(geom, v, t, b, nE, metric)
        solver = Advect(v, t, b, [.8,.6])
    
        phi = odeint(solver.ddt, zeros(solver.nt), T)

        figure()
        solver.mesh.plotTriScalar(phi[-1])
        solver.mesh.plotMesh(alpha=0.1)
        axis([-2,5,-2,5])
