from pylab import *
from numpy import *
from scipy.integrate import odeint

from mesh import *


class AdvectDiffuse:
    def __init__(self, v, t, b, U, nu):
        self.mesh = Mesh(v, t, b)

        self.U = array(U).copy()
        self.nu = float(nu)
        assert self.U.ndim == 1 and self.U.size == 2

        m = self.mesh
        xeBnd = m.v[m.e[m.ieBnd,:2],:].mean(1)
        self.phiBnd = array((xeBnd**2).sum(1) < 10, float)

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
        # advective boundary condition
        isIn = ((m.n[m.ieBnd,:] * self.U).sum(1) > 0)
        ieBndIn = m.ieBnd[isIn]
        fluxE[ieBndIn] = self.phiBnd[isIn] * (m.n[ieBndIn,:] * self.U).sum(1)
        # viscous part
        gradPhi = m.gradTriEdg(phi, self.phiBnd)
        fluxD = -self.nu * (gradPhi * m.n).sum(1)
        fluxE += fluxD
        # accumunate to cells
        flux = hstack([-fluxE, fluxE, fluxE[m.ieBnd]])
        indx = hstack([m.e[:,2], m.e[:,3], m.e[m.ieBnd,2]])
        
        #return accumarray(indx)(flux) / m.a
        return bincount(indx, flux) / m.a


if __name__ == '__main__':
    geom, nE = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]], 500
    v, t, b = initMesh(geom, nE)
    solver = AdvectDiffuse(v, t, b, [.8,.6], .05)

    T = linspace(0, 5, 2001)
    phi = odeint(solver.ddt, zeros(solver.nt), T)

    figure()
    solver.mesh.plotTriScalar(phi[-1])
    solver.mesh.plotMesh(alpha=0.1)
    axis([-2,5,-2,5])

    for iAdapt in range(5):
        print 'Adapt cycle {0}'.format(iAdapt)

        # compute metric
        metric = zeros([v.shape[0], 2, 2])
        for iStep in range(phi.shape[0] / 2, phi.shape[0]):
            phiGrad = solver.mesh.gradTriVrt(phi[iStep], solver.phiBnd)
            metric += phiGrad[:,newaxis,:] * phiGrad[:,:,newaxis]
    
        # adapt and solve
        v, t, b = adaptMesh(geom, v, t, b, nE, metric)
        solver = AdvectDiffuse(v, t, b, [.8,.6], .01)
    
        phi = odeint(solver.ddt, zeros(solver.nt), T)

        figure()
        solver.mesh.plotTriScalar(phi[-1])
        solver.mesh.plotMesh(alpha=0.1)
        axis([-2,5,-2,5])
