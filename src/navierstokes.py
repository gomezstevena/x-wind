import os
from pylab import *
from numpy import *
from scipy.sparse import kron as spkron
from scipy.sparse import linalg as splinalg

from mesh import *
from euler import *
#from eulerlowdiss import *


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

def jacV(UE, gradUE, n):
    '''
    Viscous flux in NS equation
    '''
    assert UE.shape == n.shape[:1] + (2,)
    assert gradUE.shape == n.shape[:1] + (2,2)
    # original flux
    flux = zeros([gradUE.shape[0], 4])
    f1 = einsum('nij,ni->nj', gradUE, n) + einsum('nij,nj->ni', gradUE, n)
    f2 = n * einsum('nii', gradUE)[:,newaxis]

    # momentum viscous flux
    flux[:,1:3] = -f1 + 2/3. * f2
    # start Jacobian
    fluxJ_UE = zeros(n.shape[:1] + (4,2))
    fluxJ_gradUE = zeros(n.shape[:1] + (4,2,2))

    I2 = eye(2)
    f1Jac = einsum('ni,jk->njik', n, I2) + einsum('ni,jk->njki', n, I2)
    f2Jac = einsum('ni,jk->nijk', n, I2)

    # momentum viscous flux
    fluxJ_gradUE[:,1:3] = -f1Jac + 2/3. * f2Jac
    # energy viscous flux, ignore conduction
    fluxJ_gradUE[:,3] = einsum('nijk,ni->njk', fluxJ_gradUE[:,1:3], UE)
    fluxJ_UE[:,3] = flux[:,1:3]
    return fluxJ_UE, fluxJ_gradUE


class NavierStokes(Euler):
    def __init__(self, v, t, b, Mach, Re):
        Euler.__init__(self, v, t, b, Mach)
        # compute viscosity from Reynolds number
        self.mu = self.Wref[1] / Re

    def ddt(self, W, t = 0):
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
        uE = m.triToEdge(u)#0.5 * (u[m.e[:,2],:] + u[m.e[:,3],:])
        graduE = m.gradTriEdg(u, uBc)
        flux = self.mu * fluxV(uE, graduE, m.n)
        ddtVisc = (m.distributeFlux(flux) / m.a[:,newaxis]).reshape(shp)
        return ddtEuler + ddtVisc

    def J(self, W):
        if not self.__dict__.has_key('matGradU'):
            self.prepareJacMatricesVisc()
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
        # compute uE and graduE
        uE = m.triToEdge(u)
        graduE = m.gradTriEdg(u, uBc)
        # Before are same as in ddt. now Jacobian computation
        nE, nT = m.ne, m.nt
        jacW2U = zeros([nT, 2, 4])
        jacW2U[:,[0,1],[1,2]] = 1./ W[:,:1]
        jacW2U[:,:,0] = -u / W[:,:1]
        jacW2U = block_diags(jacW2U)
        fluxJ_UE, fluxJ_gradUE = jacV(uE, graduE, m.n)
        fluxJ_UE = block_diags(self.mu * fluxJ_UE.reshape((-1,4,2)))
        fluxJ_gradUE = block_diags(self.mu * fluxJ_gradUE.reshape((-1,4,4)))

        J_flux = (fluxJ_UE*self.matJacUE + fluxJ_gradUE*self.matGradU)*jacW2U
        J_out = Euler.J(self, W) + self.matJacDistFlux*J_flux

        return J_out#.tobsr(blocksize = (4,4) )

    def J_col_dumb(self, W, t, j, ia, ja):
        if "J_store" not in self.__dict__:
            self.J_store = self.J(W)

        JC = self.J_store.getrow(j).data

        if j == W.size-1:
            del self.J_store
        return JC

    def prepareJacMatricesVisc(self):
        m = self.mesh
        # velocity gradient operator matrix
        xeBnd = m.v[m.e[m.ieBnd,:2],:].mean(1)
        isFar = m.isFar(xeBnd)
        ieFar = m.ieBnd[isFar]
        indi, indj = isFar.nonzero()[0], m.e[ieFar,3]
        shape = (isFar.size, m.t.shape[0])
        matJacUBc = csr_matrix((ones(isFar.sum()), (indi, indj)), shape)
        matGradU = m.matGradTriEdg + m.bcmatGradTriEdg * matJacUBc
        self.matGradU = spkron(matGradU, eye(2)).tocsr()
        # velocity avarage matrix
        #matL = accumarray(m.e[:,2], m.t.shape[0]).mat.T
        #matR = accumarray(m.e[:,3], m.t.shape[0]).mat.T
        self.matJacUE = spkron( m.e_map, eye(2), format='csr')

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
        gradV = self.mesh.gradTriVrt(W / self.Wref, Wbnd / self.Wref)
        #gradV /= ((gradV**2).sum(1)[:,newaxis,:])**.05
        wG = (gradV[:,newaxis,:,:] * gradV[:,:,newaxis,:]).sum(-1)
        # refine for entropy
        q, p, u, c = gask(W)
        qRef, pRef, uRef, cRef = gask(self.Wref[newaxis,:])
        nR = 8.314 / 29E-3
        S = (5./2 * log(p / pRef) - 7./2 * log(W[:,0] / self.Wref[0])) * nR
        S = self.mesh.interpTri2Vrt(S)
        wS = (1 - exp(-S / 50))[:,newaxis,newaxis] * eye(2)
        return wS + wG * (wS[:,0,0] + wS[:,1,1]).mean()


if __name__ == '__main__':
    geom = loadtxt('../data/n0012c.dat')
    geom = rotate(geom, 10*pi/180)
    nE = 1000
    dt = 0.0001
    Mach = 0.4
    Re = 1
    
    v, t, b = initMesh(geom, nE)
    solver = NavierStokes(v, t, b, Mach, Re)
    for i in range(8):
        solver.checkJacobian()
