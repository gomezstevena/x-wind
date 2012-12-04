from pylab import *
from numpy import *
from scipy.integrate import odeint
from scipy.sparse import spdiags, bsr_matrix

from mesh import *
import integrator

class AdvectDiffuse:
    def __init__(self, v, t, b, U, nu):
        self.mesh = Mesh(v, t, b)

        self.U = array(U).copy()
        self.nu = float(nu)
        assert self.U.ndim == 1 and self.U.size == 2

        m = self.mesh
        xeBnd = m.v[m.e[m.ieBnd,:2],:].mean(1)
        self.phiBnd = array((xeBnd**2).sum(1) < 10, float)

        self.ode = integrator.Ode(self.ddt, self.J)

    @property
    def nt(self):
        return self.mesh.t.shape[0]

    def J(self, phi):
        if not self.__dict__.has_key('matJ'):
            m = self.mesh
            nT, nE = m.t.shape[0], m.e.shape[0]
            matL = accumarray(m.e[:,2], nT).mat.T
            matR = accumarray(m.e[:,3], nT).mat.T
            D = block_diags((m.n * self.U).sum(1)[:,newaxis,newaxis])
            fluxE = 0.5 * D * (matL + matR)
            fluxD = 0.5 * abs(D) * (matL - matR)
            flux = fluxE + fluxD
            # boundary condition
            isIn = ((m.n[m.ieBnd,:] * self.U).sum(1) > 0)
            ieBndIn = m.ieBnd[isIn]
            flux = flux.tolil()
            flux[ieBndIn,:] = 0
            # viscous part
            flux -= self.nu * block_diags(m.n[:,newaxis,:]) * m.matGradTriEdg
            # accumunate to cells
            D = block_diags(1./ m.a[:,newaxis,newaxis])
            self.matJ = (D * m.matDistFlux * flux).tocsr()
        return self.matJ

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
        return m.distributeFlux(fluxE) / m.a


if __name__ == '__main__':
    geom, nE = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]], 500
    v, t, b = initMesh(geom, nE)
    solver = AdvectDiffuse(v, t, b, [.8,.6], .05)

    nsteps = 100
    T, phi = linspace(0, 5, nsteps+1), zeros([nsteps, solver.nt])
    solver.ode.integrate(T[0]+1E-8, zeros(solver.nt), T[0])
    for i in range(T.size - 1):
        phi[i] = solver.ode.integrate(T[i])

    figure()
    solver.mesh.plotTriScalar(phi[-1])
    solver.mesh.plotMesh(alpha=0.1)
    axis([-2,5,-2,5])

    for iAdapt in range(3):
        print 'Adapt cycle {0}'.format(iAdapt)

        # compute metric
        metric = zeros([v.shape[0], 2, 2])
        for iStep in range(phi.shape[0] / 2, phi.shape[0]):
            phiGrad = solver.mesh.gradTriVrt(phi[iStep], solver.phiBnd)
            metric += phiGrad[:,newaxis,:] * phiGrad[:,:,newaxis]
    
        # adapt and solve
        v, t, b = adaptMesh(geom, v, t, b, nE, metric)
        solver = AdvectDiffuse(v, t, b, [.8,.6], .01)
    
        T, phi = linspace(0, 5, nsteps+1), zeros([nsteps, solver.nt])
        solver.ode.integrate(T[0]+1E-8, zeros(solver.nt), T[0])
        for i in range(T.size - 1):
            phi[i] = solver.ode.integrate(T[i])

        figure()
        solver.mesh.plotTriScalar(phi[-1])
        solver.mesh.plotMesh(alpha=0.1)
        axis([-2,5,-2,5])
