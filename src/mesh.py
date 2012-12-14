from numpy import *
from scipy import sparse
from scipy.sparse import csc_matrix, csr_matrix, bsr_matrix
import scipy.sparse.linalg as splinalg

from meshUtils import *
import meshAni2D as _ani
import meshGradients as _grad
import meshVisualize as _vis

def extractEdges(v, t):
    '''
    EXTRACTEDGES extract all edges in a triangular mesh
    Inputs v, t have the same format as ani2D
        v[:,0] are the x coordinates
        v[:,1] are the y coordinates
        t[:,0], t[:,1) and t[:,2] are the p-indices of the triangle vertices.
    Returns e:
        e[:,0] and e[:,1] are the p-indices of the edge end points
        e[:,2] is the t-index of the triangle on the left
        e[:,3] is the t-index of the triangle on the right,
               it is the same as e[2,:] for boundary edges.
    '''
    # pick out edges from triangle sides
    allEdges = vstack([t[:,:2], t[:,1:3], t[:,[0,2]]])
    allEdges.sort(axis=1)
    int64Edges = allEdges[:,0] * (2**32) + allEdges[:,1]
    isort = int64Edges.argsort(kind='mergesort')      # mergesort is stable
    iunsort = isort.argsort(kind='mergesort')
    # unique edges
    sortedEdges = int64Edges[isort]
    imax = isort.size - 1
    isel0 = hstack([0, (sortedEdges[1:] > sortedEdges[:-1]).nonzero()[0] + 1])
    isel1 = hstack([(sortedEdges[1:] > sortedEdges[:-1]).nonzero()[0], imax])
    e0 = sortedEdges[isel0]
    e1 = sortedEdges[isel1]
    assert (e0 == e1).all()
    assert (isel1 - isel0).min() == 0 and (isel1 - isel0).max() == 1
    # get back to int32
    e = transpose([array(e0 / (2**32), int32), array(e0 % (2**32), int32)])
    # find corresponding triangles
    it0 = isort[isel0] % (t.shape[0])
    it1 = isort[isel1] % (t.shape[0])
    # find which side the triangles are
    tcenter = (v[t[:,0],:] + v[t[:,1],:] + v[t[:,2],:]) / 3.
    dpEdg = v[e[:,1],:] - v[e[:,0],:]
    dpCtr1 = tcenter[it1,:] - v[e[:,0],:]
    crossProd = dpEdg[:,0] * dpCtr1[:,1] - dpEdg[:,1] * dpCtr1[:,0]
    # swap to make sure it1 is on the left
    isel = (crossProd < 0)
    tmp = e[isel,0]
    e[isel,0] = e[isel,1]
    e[isel,1] = tmp
    return hstack([e, it0[:,newaxis], it1[:,newaxis]])

def adaptMesh(geom, v, t, b, nE, metric, diameter=None):
    '''
    Adapt mesh with a metric defined on vertices
    '''
    assert metric.shape == (v.shape[0], 2, 2)
    # compute initial metric
    c, d = centerDiameter(geom)
    if diameter is not None: d = diameter
    m = (((v-c)**2).sum(1) + 5*d**2)**-2
    # adjust diagonal for positive definiteness and limit anisotropy
    mxx, myy, mxy = metric[:,0,0], metric[:,1,1], metric[:,0,1]
    mMin = (mxx + myy) / 2. - sqrt((mxx - myy)**2 / 4. + mxy**2)
    mMax = (mxx + myy) / 2. + sqrt((mxx - myy)**2 / 4. + mxy**2)
    mAdjust = maximum(mMax.max() * 1E-12, mMax*.001 - mMin)
    mxx += mAdjust
    myy += mAdjust
    mMin += mAdjust
    mMax += mAdjust
    # adjust length ratio 1,000,000,000:1
    maxOverMin = mMax.max() / mMax.min()
    expFactor = minimum(1, log10(1E9) / log10(maxOverMin))
    multFactor = mMax**(expFactor-1)
    m *= multFactor
    # combine
    m = transpose([m * mxx, m * myy, m * mxy])
    return _ani.mbaMesh(v, t, b, nE, m)


def initMesh(geom, nE, diameter=None):
    '''
    Generate mesh with a simple metric
    '''
    c, d = centerDiameter(geom)
    if diameter is not None: d = diameter
    v, t, b = _ani.aftMesh(geom, d)
    # compute initial metric
    metric = (((v-c)**2).sum(1) + 5*d**2)**-2
    metric = transpose([metric, metric, metric * 0])
    return _ani.mbaMesh(v, t, b, nE, metric)


class Mesh:
    '''
    Constructed from v, t, b (can come from initMesh or adaptmesh)
    Provide axiliary geometry information, differential operators,
    visualization etc
    '''
    def __init__(self, v, t, b):
        self.v, self.t, self.b = v, t, b
        self.e = extractEdges(v, t)
        self.a = triArea(v, t)
        self.n = edgNormal(v, self.e)

        self.ieBnd = (self.e[:,3] == self.e[:,2]).nonzero()[0]

        self.edgOfTri = invertMap(self.e[:,2:])[1].reshape([-1, 3])
        # EdgeMean when multiplied by values at edges gives values 
        # at triangles by averaging across edges
        self.edgeMean = indexMap( self.edgOfTri[:,0], self.ne, 1./3. )
        for i in xrange(1, 3):
            self.edgeMean = self.edgeMean + indexMap( self.edgOfTri[:,i], self.ne, 1/3.0)

        # Clean up edge Mean matrix
        self.edgeMean.sum_duplicates()
        self.edgeMean.prune()


        self.l_map = indexMap(self.e[:,2], self.nt)
        self.r_map = indexMap(self.e[:,3], self.nt)
        self.lr_map = sparse.vstack([self.l_map, self.r_map], format='csr')
        self.e_map = 0.5*(self.l_map + self.r_map)

        ## X-location of each triangle center
        self._xt = self.v[self.t,:].mean(1)

        xl, xr = self.leftRightTri(self._xt)
        self._dxt = xr-xl #self._xt[self.e[:,3],:] - self._xt[self.e[:,2],:]



    @property
    def nt(self):
        return self.t.shape[0]

    @property
    def ne(self):
        return self.e.shape[0]

    def center(self):
        '0.5 * (v.min(0) + v.max(0))'
        return 0.5 * (self.v.min(0) + self.v.max(0))

    def diameter(self):
        'max(v.max(0) - v.min(0))'
        return max(self.v.max(0) - self.v.min(0))

    def isFar(self, xy):
        return ((xy - self.center())**2).sum(1) >= 0.4 * self.diameter()

    def audit(self):
        v, t, b, e, a, n = self.v, self.t, self.b, self.e, self.a, self.n
        xt = self.xt()
        xe = v[e[:,:2],:].mean(1)
        dxt2e = xt[e[:,2],:] - xe
        dxt3e = xt[e[:,3],:] - xe
        assert ((dxt3e * n).sum(1) > 0).all()
        ieInterior = (e[:,2] != e[:,3])
        assert ((dxt2e * n).sum(1)[ieInterior] < 0).all()

    def xt(self):
        return self._xt
    
    @property
    def dxt(self):
        return self._dxt

    
    leftRightTri = _grad.leftRightTri
    leftTri = _grad.leftTri
    rightTri = _grad.rightTri
    triToEdge = _grad.triToEdge
    edgeToTri = _grad.edgeToTri

    # -------------- plotting ---------------- #
    def plotMesh(self, detail=0, alpha=1):
        '''
        draw primal and dual mesh and edge normals
        detail = 0 (least) - 2 (most)
        '''
        _vis.plotMesh(self, detail, alpha)

    def plotTriScalar(self, phi):
        '''
        Contour plot of scalar field phi defined on mesh triangles
        shading is flat, i.e. piecewise constant
        '''
        _vis.plotTriScalar(self, phi)

    def plotTriVector(self, vec, *argc, **argv):
        '''
        Plot of vector field vec of shape (nt, 2) defined on mesh triangles
        '''
        _vis.plotTriVector(self, vec)

    def plotEdgScalar(self, phi):
        '''
        Contour plot of scalar field phi defined on mesh edges
        shading is flat, i.e. piecewise constant
        '''
        _vis.plotEdgScalar(self, phi)

    def plotVrtScalar(self, phi):
        '''
        Contour plot of scalar field phi defined on mesh vertices
        Represented by circles of proportional size as dual volume area
        '''
        return _vis.plotVrtScalar(self, phi)


    # -------------- gradients --------------- #
    def gradTri(self, phi, phiBc=None):
        '''
        Input: scalar field phi of shape (nt,m) defined on triangles
        Output: vector field gradPhi of shape (nt,2,m) defined on triangles
        '''
        return _grad.gradTri(self, phi, phiBc)

    def gradTriVrt(self, phi, phiBc=None):
        '''
        Input: scalar field phi of shape (nt,m) defined on triangles
        Output: vector field gradPhi of shape (nv,2,m) defined on vertices
        '''
        return _grad.gradTriVrt(self, phi, phiBc)

    def gradTriEdg(self, phi, phiBc=None):
        '''
        Input: scalar field phi of shape (nt,m) defined on triangles
               phiBc defines phi at boundaries,
               in same order as boundary edges in mesh.e
        Output: vector field gradPhi of shape (ne,2,m) defined on EDGES
        '''
        return _grad.gradTriEdg(self, phi, phiBc)

    def distributeFlux(self, flux, isAdjoint=False):
        '''
        When isAdjoint is False (default):
        Input: scalar flux (already dotted with n) on edges
        Output: flux accumulated in cells

        when isAdjoint is True, invoke the adjoint operator (cell to edges)
        '''
        return _grad.distributeFlux(self, flux, isAdjoint)

    def interpTri2Vrt(self, phi):
        return _grad.interpTri2Vrt(self, phi)

    '''
    def interpTri2Edg(self, phi, isAdjoint=False):
        return _grad.interpTri2Edg(self, phi, isAdjoint)'''

if __name__ == '__main__':
    v, t, b = initMesh( \
        [[1, 0], [0.3, 0], [0, 0.1], [-0.3, 0], [-1, 0], [-0.4, -0.5],
         [-0.7, -1.2], [0, -0.1], [0.7, -1.2], [0.4, -0.5], [1, 0]], 2000)
    from pylab import *
    mesh = Mesh(v, t, b)
    mesh.audit()
    mesh.plotMesh()
    # test gradient
    p = mesh.v[mesh.t,1].mean(1)**2
    pe = mesh.v[mesh.e[:,:2],1].mean(1)**2
    pgrad = mesh.gradTri(p)
    figure()
    mesh.plotTri(pgrad[:,1])

