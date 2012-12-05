import os
from pylab import *
from numpy import *
from scipy.sparse import kron as spkron
from scipy.sparse import linalg as splinalg

from mesh import *
from euler import *

#from IPython import embed

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
    #f1 = (gradUE * n[:,:,newaxis]).sum(1) + (gradUE * n[:,newaxis,:]).sum(2)
    f1 = einsum('nij,ni->nj', gradUE, n) + einsum('nij,nj->ni', gradUE, n)
    f2 = n * einsum('nii', gradUE)[:,newaxis]

    # momentum viscous flux
    flux[:,1:3] = -f1 + 2/3. * f2
    # start Jacobian
    fluxJ_UE = zeros(n.shape[:1] + (4,2))
    fluxJ_gradUE = zeros(n.shape[:1] + (4,2,2))

    I2 = eye(2)
    #f1Jac = n[:,newaxis,:,newaxis] * I2[newaxis,:,newaxis,:] \
    #        + n[:,newaxis,newaxis,:] * I2[newaxis,:,:,newaxis]
    f1Jac = einsum('ni,jk->njik', n, I2) + einsum('ni,jk->njki', n, I2)
    #f2Jac = n[:,:,newaxis,newaxis] * I2[newaxis,newaxis,:,:]
    f2Jac = einsum('ni,jk->nijk', n, I2)

    # momentum viscous flux
    fluxJ_gradUE[:,1:3] = -f1Jac + 2/3. * f2Jac
    # energy viscous flux, ignore conduction
    #fluxJ_gradUE[:,3] = (fluxJ_gradUE[:,1:3] * UE[:,:,newaxis,newaxis]).sum(1)
    fluxJ_gradUE[:,3] = einsum('nijk,ni->njk', fluxJ_gradUE[:,1:3], UE)
    fluxJ_UE[:,3] = flux[:,1:3]
    return fluxJ_UE, fluxJ_gradUE


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
        uE = m.triToEdge(u)#0.5 * (u[m.e[:,2],:] + u[m.e[:,3],:])
        graduE = m.gradTriEdg(u, uBc)
        flux = self.mu * fluxV(uE, graduE, m.n)
        ddtVisc = ( m.distributeFlux(flux) / m.a[:,newaxis] ).reshape(shp)
        return ddtEuler + ddtVisc

    #@profile
    def J(self, W):
        if not self.__dict__.has_key('matGradU'):
            self.prepareJacMatricesVisc()
        #ddtEuler = Euler.ddt(self, W)
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
        uE = m.triToEdge(u)#0.5 * (u[m.e[:,2],:] + u[m.e[:,3],:])
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
        #embed()
        J_flux = (fluxJ_UE*self.matJacUE + fluxJ_gradUE*self.matGradU)*jacW2U
        return Euler.J(self, W) + self.matJacDistFlux*J_flux

    def J_Oper(self, W ):
        if not self.__dict__.has_key('matGradU'):
            self.prepareJacMatricesVisc()
        #ddtEuler = Euler.ddt(self, W)
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
        uE = 0.5 * (u[m.e[:,2],:] + u[m.e[:,3],:])
        graduE = m.gradTriEdg(u, uBc)
        # Before are same as in ddt. now Jacobian computation
        nE, nT = m.e.shape[0], m.t.shape[0]
        jacW2U = zeros([nT, 2, 4])
        jacW2U[:,[0,1],[1,2]] = 1./ W[:,:1]
        jacW2U[:,:,0] = -u / W[:,:1]
        jacW2U = block_diags(jacW2U)
        fluxJ_UE, fluxJ_gradUE = jacV(uE, graduE, m.n)
        fluxJ_UE = block_diags(self.mu * fluxJ_UE.reshape((-1,4,2)))
        fluxJ_gradUE = block_diags(self.mu * fluxJ_gradUE.reshape((-1,4,4)))

        EJ = Euler.J(self, W)


        def matvec(X):

            out = EJ*X
            JW2_X = jacW2U * X
            JFlux_X = fluxJ_UE * (self.matJacUE * JW2_X)
            JFlux_X += fluxJ_gradUE * (self.matGradU * JW2_X)
            out += self.matJacDistFlux * JFlux_X
            return out


        return matvec


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
        matL = accumarray(m.e[:,2], m.t.shape[0]).mat.T
        matR = accumarray(m.e[:,3], m.t.shape[0]).mat.T
        self.matJacUE = spkron(0.5 * (matL + matR), eye(2)).tocsr()

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
        # weight by entropy adjoint
        # q, p, u, c = gask(W)
        # V = hstack([q[:,newaxis], W[:,1:3], W[:,:1]]) * self.Wref
        # gradV *= m.interpTri2Vrt(V)
        return (gradV[:,newaxis,:,:] * gradV[:,:,newaxis,:]).sum(-1)


if __name__ == '__main__':
    geom = loadtxt('../data/n0012c.dat')
    geom = rotate(geom, 10*pi/180)
    nE = 1000
    dt = 0.0001
    Mach = 4.0
    HiRes = 0.0
    Re = 1
    
    v, t, b = initMesh(geom, nE)
    solver = NavierStokes(v, t, b, Mach, Re, HiRes)
    for i in range(8):
        solver.checkJacobian()
