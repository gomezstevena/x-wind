import os
from pylab import *
from numpy import *
from scipy.sparse import kron as spkron

from mesh import *
from integrator import Ode, CrankNicolson

def gask(W, gamma=1.4):
    '''
    Returns kinetic energy q, pressure p, velocity u and sound speed c
    '''
    u = W[:,1:3] / W[:,:1]           # velocity

    #q = 0.5 * W[:,0] * sum(u**2,1)   # kinetic energy
    q = dot_each(u, u); q *= W[:,0]; q*= 0.5;
    p = (W[:,3] - q); p *= (gamma - 1)   # pressure
    #c = sqrt(gamma * p / W[:,0] )     # speed of sound
    c = p/W[:,0]; c*=gamma; sqrt(c, out=c)

    return q, p, u, c

def jacGask(W, gamma=1.4):
    q, p, u, c = gask(W, gamma)
    jacU = zeros(W.shape[:1] + (2,4))
    jacU[:,[0,1],[1,2]] = 1./ W[:,:1]
    jacU[:,:,0] = -u / W[:,:1]
    jacQ = zeros(W.shape)
    jacQ[:,1:3] = u
    jacQ[:,0] = -q / W[:,0]
    jacP = -jacQ * (gamma - 1)
    jacP[:,3] = gamma - 1
    jacC = jacP / p[:,newaxis]
    jacC[:,0] -= 1 / W[:,0]
    jacC *= 0.5 * c[:,newaxis]
    return jacQ, jacP, jacU, jacC

def jacConsSym(W, n=None, gamma=1.4):
    '''
    Returns S, T, (L)
    S: Jacobian transforming conservative vars [d_rho, d_rhoU, d_rhoV, d_rhoE]
       to symmetrizable vars [d_p/(rho c), d_u, d_v, d_p - c^2 d_rho]
    T: Jacobian transforming symmetrizable vars to conservative vars
    L: symmetric Jacobian (only if n is not None)
    '''
    q, p, u, c = gask(W, gamma)
    rhoC = W[:,0] * c
    rhoOverC = W[:,0] / c
    rhoH = W[:,3] + p
    c2 = c**2

    S = zeros(W.shape[:1] + (4, 4))
    S[:,0,0] = (gamma - 1) * q / W[:,0] / rhoC
    S[:,0,1:3] = -(gamma - 1) * u / rhoC[:,newaxis]
    S[:,0,3] = (gamma - 1) / rhoC
    S[:,1:3,0] = -u / W[:,:1]
    S[:,[1,2],[1,2]] = 1. / W[:,:1]
    S[:,3,0] = (gamma - 1) * q / W[:,0] - c**2
    S[:,3,1:3] = -(gamma - 1) * u
    S[:,3,3] = (gamma - 1)

    T = zeros(W.shape[:1] + (4, 4))
    T[:,0,0] = rhoOverC
    T[:,0,3] = -1 / c2
    T[:,1:3,0] = rhoOverC[:,newaxis] * u
    T[:,[1,2],[1,2]] = W[:,:1]
    T[:,1:3,3] = -u / c2[:,newaxis]
    T[:,3,0] = rhoH / c
    T[:,3,1:3] = W[:,1:3]
    T[:,3,3] = -q / W[:,0] / c2

    if n is not None:
        uDotN = (u * n).sum(1)
        cN = c[:,newaxis] * n
        # symmetric Jacobian
        L = zeros([W.shape[0], 4, 4])
        L[:,r_[:4], r_[:4]] = uDotN[:,newaxis]
        L[:,1:3, 0] = cN
        L[:,0, 1:3] = cN
        return S, T, L
    else:
        return S, T

def fluxE(WE, n):
    assert WE.shape == n.shape[:1] + (4,)
    q, p, u, c = gask(WE)
    uDotN = dot_each(u, n)
    flux = zeros(WE.shape)
    flux[:,0] = WE[:,0] * uDotN
    flux[:,1:3] = WE[:,1:3] * uDotN[:,newaxis] + p[:,newaxis] * n
    flux[:,3] = (WE[:,3] + p) * uDotN
    return flux

def jacE(WE, n):
    assert WE.shape == n.shape[:1] + (4,)
    S, T, L = jacConsSym(WE, n)
    JE = matrixMult(T, L, S)
    return JE

def specRad(WE, n):
    q, p, u, c = gask(WE)
    uDotN = dot_each(u, n) #(u * n).sum(1)
    cNrmN = c * sqrt( dot_each(n,n) )
    return (abs(uDotN) + cNrmN)

def jacSpecRad(WE, n):
    q, p, u, c = gask(WE)
    uDotN = (u * n).sum(1)
    jacQ, jacP, jacU, jacC = jacGask(WE)
    J_uDotN = (jacU * n[:,:,newaxis]).sum(1)
    J_cNrmN = jacC * sqrt((n**2).sum(1))[:,newaxis]
    return sign(uDotN)[:,newaxis] * J_uDotN + J_cNrmN

def fluxD(WE, dW, dWL, dWR, n):
    assert WE.shape == dW.shape == n.shape[:1] + (4,)
    A = specRad(WE, n)[:,newaxis]
    flux = -0.5 * A * dW
    flux += 0.25 * A * (dWR + dWL)
    return flux

def jacD(WE, dW, dWL, dWR, n):
    assert WE.shape == n.shape[:1] + (4,)
    A = specRad(WE, n)
    jacA = jacSpecRad(WE, n)
    J_WE = -0.5 * dW[:,:,newaxis] * jacA[:,newaxis,:]
    J_WE += 0.25 * (dWL + dWR)[:,:,newaxis] * jacA[:,newaxis,:]
    J_dW = -0.5 * A[:,newaxis,newaxis] * eye(4)
    J_dWL = 0.25 * A[:,newaxis,newaxis] * eye(4)
    J_dWR = 0.25 * A[:,newaxis,newaxis] * eye(4)
    return J_WE, J_dW, J_dWL, J_dWR

def wallBc(W, n):
    assert W.shape == n.shape[:1] + (4,)
    q, p, u, c = gask(W)
    flux = zeros(W.shape)
    flux[:,1:3] = p[:,newaxis] * n
    return flux

def jacWall(W, n):
    jacQ, jacP, jacU, jacC = jacGask(W)
    J = zeros(W.shape[:1] + (4,4))
    J[:,1:3,:] = jacP[:,newaxis,:] * n[:,:,newaxis]
    return J


class Euler:
    def __init__(self, v, t, b, Mach):
        self.mesh = Mesh(v, t, b)
        self.Mach = float(Mach)
        self.ode = CrankNicolson(self.ddt, self.J)

    def integrate(self, *args, **argv):
        return self.ode.integrate(*args, **argv)

    @property
    def nt(self):
        return self.mesh.nt

    @property
    def time(self):
        return self.ode.t

    @property
    def soln(self):
        return reshape(self.ode.y, [-1,4])

    @property
    def Wref(self):
        Wr = self.freeStream(1)[0]
        Wr[2] = Wr[1]
        return Wr

    def freeStream(self, nT=None):
        if nT is None: nT = self.nt
        R, gamma, rho0, T0 = 8.314 / 29E-3, 1.4, 1.225, 288.75
        u0 = self.Mach * sqrt(gamma * R * T0)
        E0 = T0 * R / (gamma - 1) + 0.5 * u0**2
        return array([rho0, rho0 * u0, 0, rho0 * E0]) + zeros([nT, 1])

    def ddt(self, W):
        assert W.size == self.nt * 4
        shp = W.shape
        W = W.reshape([-1, 4])
        m = self.mesh
        gradW = m.gradTri(W)

        xt = m.xt()
        dxt = m.dxt

        # boundary condition categories
        xeBnd = m.v[m.e[m.ieBnd,:2],:].mean(1)
        isWall = ~m.isFar(xeBnd)

        # interior flux

        #WL, WR = W[m.e[:,2],:], W[m.e[:,3],:]
        WL, WR = m.leftRightTri(W)
        WL[m.ieBnd[~isWall]] = self.freeStream(1)

        WE = 0.5*(WL+WR)
        dW = WR - WL

        #dWL = (gradW[m.e[:,2],:] * dxt[:,:,newaxis]).sum(1)
        #dWR = (gradW[m.e[:,3],:] * dxt[:,:,newaxis]).sum(1)
        gWL, gWR = m.leftRightTri(gradW)
        dWL = einsum( 'nij, ni -> nj', gWL, dxt )
        dWR = einsum( 'nij, ni -> nj', gWR, dxt )

        flux = fluxE(WE, m.n) + fluxD(WE, dW, dWL, dWR, m.n)
        
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
        gradW = m.gradTri(W)
        xt = m.xt(); 
        dxt = m.dxt
        WL, WR = m.leftRightTri(W)
        isFar = m.isFar(m.v[m.e[m.ieBnd,:2],:].mean(1))
        WL[m.ieBnd[isFar]] = self.freeStream(1)
        WE, dW = 0.5 * (WL + WR), WR - WL
        JE_WE = jacE(WE, m.n)

        gWL, gWR = m.leftRightTri(gradW)
        dWL = einsum( 'nij, ni -> nj', gWL, dxt )
        dWR = einsum( 'nij, ni -> nj', gWR, dxt )

        JD_WE, JD_dW, JD_dWL, JD_dWR  = jacD(WE, dW, dWL, dWR, m.n)
        # modify for wall BC
        J_WE = JE_WE + JD_WE
        ieWall = m.ieBnd[~isFar]
        J_WE[ieWall,:] = jacWall(WE[ieWall], m.n[ieWall])
        J_flux = block_diags(J_WE) * self.matJacWE \
               + block_diags(JD_dW) * self.matJacdW \
               + block_diags(JD_dWL) * self.matJacdWL \
               + block_diags(JD_dWR) * self.matJacdWR
        return self.matJacDistFlux * J_flux

    def prepareJacMatrices(self):
        m = self.mesh
        matL = accumarray(m.e[:,2], m.t.shape[0]).mat.T.tolil()
        matR = accumarray(m.e[:,3], m.t.shape[0]).mat.T.tolil()
        # WL on far bc is freestream
        isFar = m.isFar(m.v[m.e[m.ieBnd,:2],:].mean(1))
        matL[m.ieBnd[isFar],:] = 0
        # average and difference
        self.matJacWL = spkron(matL, eye(4)).tocsr()
        self.matJacWR = spkron(matR, eye(4)).tocsr()
        self.matJacWE = spkron(0.5 * (matL + matR), eye(4)).tocsr()
        self.matJacdW = spkron(matR - matL, eye(4)).tocsr()
        # distribute flux matrix
        D = block_diags(1./ m.a[:,newaxis,newaxis])
        self.matJacDistFlux = spkron(D * m.matDistFlux, eye(4)).tocsr()
        # Tri to edg gradient with boundary copy
        matTriToBnd = accumarray(m.e[m.ieBnd,2], m.t.shape[0]).mat.T
        matGradTriEdg = m.matGradTriEdg + m.bcmatGradTriEdg * matTriToBnd
        # left and right gradient in cells
        edgOfTri = invertMap(m.e[:,2:])[1].reshape([-1, 3])
        avgEdgToTri = (accumarray(edgOfTri[:,0], m.e.shape[0]).mat.T + \
                       accumarray(edgOfTri[:,1], m.e.shape[0]).mat.T + \
                       accumarray(edgOfTri[:,2], m.e.shape[0]).mat.T) / 3.
        matGradTri = spkron(avgEdgToTri, eye(2)) * matGradTriEdg
        xt = m.xt(); dxt = xt[m.e[:,3],:] - xt[m.e[:,2],:]
        matL = spkron(accumarray(m.e[:,2], m.t.shape[0]).mat.T, eye(2))
        matR = spkron(accumarray(m.e[:,3], m.t.shape[0]).mat.T, eye(2))
        matJacdWL = block_diags(dxt[:,newaxis,:]) * matL * matGradTri
        matJacdWR = block_diags(dxt[:,newaxis,:]) * matR * matGradTri
        self.matJacdWL = spkron(matJacdWL, eye(4)).tocsr()
        self.matJacdWR = spkron(matJacdWR, eye(4)).tocsr()

    def metric(self, W=None):
        if W is None: W = self.soln
        gradV = self.mesh.gradTriVrt(W / self.Wref)
        gradV /= ((gradV**2).sum(1)[:,newaxis,:])**.25
        return (gradV[:,newaxis,:,:] * gradV[:,:,newaxis,:]).sum(-1)
        # return hessian.sum(-1)

    def checkJacobian(self):
        '''
        Generate a flow with small linear variation (supress numerical diss.),
        check adjoint ddt vs Euler ddt
        '''
        xt = self.mesh.xt()
        R = 0.2 * self.mesh.diameter()
        decay = exp(-(xt**2).sum(1) / R**2)[:,newaxis]
        nT = self.mesh.t.shape[0]
        # random flow field
        variation = 0.1 * (random.rand(nT, 4) - 0.5)
        W0 = self.freeStream() + self.Wref * variation
        # random perturbation
        EPS = 0.000001
        variation = EPS * (random.rand(nT, 4) - 0.5)
        W1 = W0 + self.Wref * variation
        # compare ddt with Jacobian
        deltaFD = self.ddt(ravel(W1)) - self.ddt(ravel(W0))
        deltaJac = self.J(0.5 * (W0 + W1)) * ravel(W1 - W0)
        # difference
        print sqrt(((deltaFD - deltaJac)**2).sum()), \
              sqrt((deltaJac**2).sum()), sqrt((deltaFD**2).sum())


if __name__ == '__main__':
    geom = loadtxt('../data/n0012c.dat')
    geom = rotate(geom, 10*pi/180)
    nE = 1000
    dt = 0.0001
    Mach = 4.0
    HiRes = 0.0
    
    v, t, b = initMesh(geom, nE)
    solver = Euler(v, t, b, Mach)
    for i in range(8):
        solver.checkJacobian()
