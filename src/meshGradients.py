'''
Module for computing gradients and related operators on a mesh,
intended to be used by mesh.py
'''

from numpy import *
from scipy.sparse import csc_matrix, csr_matrix, bsr_matrix
import scipy.sparse.linalg as splinalg

from meshUtils import *

def interpTri2Vrt(mesh, phi):
    a0 = accumarray(ravel(mesh.t[:,0]), mesh.v.shape[0])
    a1 = accumarray(ravel(mesh.t[:,1]), mesh.v.shape[0])
    a2 = accumarray(ravel(mesh.t[:,2]), mesh.v.shape[0])
    one = ones(mesh.t.shape[0])
    n = a0(one) + a1(one) + a2(one)
    n = n.reshape((-1,) + (1,) * (phi.ndim - 1))
    return (a0(phi) + a1(phi) + a2(phi)) / n

def gradTri(mesh, phi, phiBc):
    '''
    Input: scalar field phi of shape (nt,m) defined on triangles
    Output: vector field gradPhi of shape (nt,2,m) defined on triangles
    '''
    assert phi.shape[0] == mesh.nt
    # edge gradient for each triangle
    gradEdg = gradTriEdg(mesh, phi, phiBc)
    # average of edge gradients
    
    return edgeToTri(mesh, gradEdg)

def edgeToTri(mesh, edgVal):
    '''
    Input: values on cell edges shape:(ne, val_shape )
    Output: values on triangles from the mean of value on each edge shape:(nt, val_shape)
    '''
    triVal = mesh.edgeMean * edgVal.reshape( (mesh.ne, -1) )
    return triVal.reshape( (-1,) + edgVal.shape[1:] )

def leftRightTri(mesh, W):
    '''
    Input: W values on triangles, shape:(nt, vshape)
    Output: W value on (left, right) of each edge, shapes: (ne, vshape), (ne, vshape)
    '''
    Wf = W.reshape( (mesh.nt, -1) )
    Wlr = mesh.lr_map * Wf
    Wlr.shape = (2*mesh.ne,) + W.shape[1:]
    return Wlr[:mesh.ne], Wlr[mesh.ne:]

def leftTri(mesh, W):
    '''
    Input: W values on triangles, shape:(nt, vshape)
    Output: W value on left of each edge, shape:(ne, vshape)
    '''
    Wf = W.reshape( (mesh.nt, -1) )
    WL = mesh.l_map * Wf
    WL.shape = (mesh.ne,) + W.shape[1:]
    return WL

def rightTri(mesh, W):
    '''
    Input: W values on triangles, shape:(nt, vshape)
    Output: W value on right of each edge, shape:(ne, vshape)
    '''
    Wf = W.reshape( (mesh.nt, -1) )
    WR = mesh.r_map * Wf
    WR.shape = (mesh.ne,) + W.shape[1:]
    return WR

def triToEdge(mesh, W):
    '''
    Input:  W values on triangles, shape:(nt, vshape)
    Output: WE value on each edge, shape:(ne, vshape)
    '''
    Wf = W.reshape( (mesh.nt, -1) )

    Wlr = mesh.lr_map * Wf
    Wlr.shape = (2, mesh.ne,) + W.shape[1:]
    return Wlr.mean(0)

def gradTriVrt(mesh, phi, phiBc):
    '''
    Input: scalar field phi of shape (nt,m) defined on triangles
    Output: vector field gradPhi of shape (nv,2,m) defined on vertices
    '''
    assert phi.shape[0] == mesh.t.shape[0]
    # edge gradient for each triangle
    gradEdg = gradTriEdg(mesh, phi, phiBc)
    # average of edge gradients

    a2 = accumarray(ravel(mesh.e[:,0]), mesh.v.shape[0])
    a3 = accumarray(ravel(mesh.e[:,1]), mesh.v.shape[0])
    n = a2(ones(mesh.e.shape[0])) + a3(ones(mesh.e.shape[0]))
    
    # this works here but can't handle more complicated inputs
    #n = bincount( mesh.e[:,0].ravel() ) + bincount( mesh.e[:,1].ravel() )

    n = n.reshape((-1,) + (1,) * phi.ndim)
    return (a2(gradEdg) + a3(gradEdg)) / n

def gradTriEdg(mesh, phi, phiBc):
    '''
    Input: scalar field phi of shape (nt,m) defined on triangles
    Output: vector field gradPhi of shape (ne,2,m) defined on EDGES
    '''
    if phiBc is None:
        phiBc = phi[mesh.e[mesh.ieBnd, 3]]
    elif type(phiBc) is not ndarray:
        phiBc = ones(mesh.b.shape[0]) * phiBc
    assert phi.shape[0] == mesh.t.shape[0]
    assert phiBc.shape[0] == mesh.ieBnd.size
    shape = phi.shape[1:]
    phi = phi.reshape(phi.shape[:1] + (-1,))
    phiBc = phiBc.reshape(phiBc.shape[:1] + (-1,))
    if not mesh.__dict__.has_key('matGradTriEdg'):
        v, t, b, e, a, n = mesh.v, mesh.t, mesh.b, mesh.e, mesh.a, mesh.n
        xt = v[t,:].mean(1)
        xe = v[e[:,:2],:].mean(1)
        # vector between neighboring triangles
        dxTri = xt[e[:,3],:] - xt[e[:,2],:]
        dxTri[mesh.ieBnd] = xt[e[mesh.ieBnd,3],:] - xe[mesh.ieBnd,:]
        # this is going to be slow, needs to be done once for each mesh
        # probably need to be replaced by something more vectorized
        indptr, indices = invertMap(t)
        coef, indi, indj, coefBc, indiBc = [], [], [], [], []
        for ie in range(e.shape[0]):
            iv0, iv1 = e[ie,:2]
            tri0 = indices[indptr[iv0]:indptr[iv0+1]]
            tri1 = indices[indptr[iv1]:indptr[iv1+1]]
            allTris = set(hstack([tri0, tri1]))  # all tris of vrts of edg
            farTris = array(list(allTris.difference(e[ie,2:])), int)
            assert farTris.size > 0
            penaltyFar = ((xt[farTris,:] - xe[ie,:])**2).sum(1)
            dxtriFar = xt[e[ie,3],:] - xt[farTris,:]
            # build KKT
            D = diag(hstack([0, penaltyFar]))
            C = vstack([dxTri[ie][newaxis,:], dxtriFar])
            Z = zeros((2,2))
            KKT = vstack([hstack([D, C]), hstack([C.T, Z])])
            B = vstack([zeros([D.shape[0], 2]), eye(2)])
            # solve KKT and assign coefficients
            x = linalg.solve(KKT, B)[:D.shape[0]]
            if e[ie,2] == e[ie,3]:
                coef.append(vstack([x.sum(0)[newaxis,:], -x[1:,:]]))
                indj.append(hstack([e[ie,3], farTris]))
                indi.append(ie + zeros(D.shape[0], int))
                coefBc.append(-x[0,:])
                indiBc.append(ie)
            else:
                coef.append(vstack([-x[:1,:], x.sum(0)[newaxis,:], -x[1:,:]]))
                indj.append(hstack([e[ie,2], e[ie,3], farTris]))
                indi.append(ie + zeros(D.shape[0] + 1, int))
        # assemble matrices
        coef = ravel(vstack(coef).T)
        indi = ravel([hstack(indi) * 2, hstack(indi) * 2 + 1])
        indj = hstack(indj * 2)
        mesh.matGradTriEdg = csc_matrix((coef, (indi, indj)))
        shp = (e.shape[0] * 2, mesh.ieBnd.size)
        coefBc = ravel(vstack(coefBc).T)
        indiBc = ravel([hstack(indiBc) * 2, hstack(indiBc) * 2 + 1])
        indjBc = hstack([arange(mesh.ieBnd.size), arange(mesh.ieBnd.size)])
        mesh.bcmatGradTriEdg = csc_matrix((coefBc, (indiBc, indjBc)), shp)
    # gradient computation using sparse matrix
    gradPhi = (mesh.matGradTriEdg * phi + mesh.bcmatGradTriEdg * phiBc)
    return gradPhi.reshape((mesh.e.shape[0], 2) + shape)

def distributeFlux(mesh, flux, isAdjoint=False):
    '''
    When isAdjoint is False (default):
    Input: scalar flux (already dotted with n) on edges
    Output: flux accumulated in cells

    when isAdjoint is True, invoke the adjoint operator (cell to edges)
    '''
    if not mesh.__dict__.has_key('matDistFlux'):
        v, t, b, e, a, n = mesh.v, mesh.t, mesh.b, mesh.e, mesh.a, mesh.n
        ne = e.shape[0]
        ieBnd = mesh.ieBnd
        data = hstack([-ones(ne), ones(ne + ieBnd.size)])
        indi = hstack([e[:,2], e[:,3], e[ieBnd,3]])
        indj = hstack([r_[:ne], r_[:ne], ieBnd])
        mesh.matDistFlux = csr_matrix((data, (indi, indj)))
    if not isAdjoint:
        return mesh.matDistFlux * flux
    else:
        return mesh.matDistFlux.T * flux

def interpTri2Edg(mesh, value):
    if not mesh.__dict__.has_key('matInterpTri2Edg'):
        mat2 = accumarray(mesh.e[:,2], mesh.t.shape[0]).mat
        mat3 = accumarray(mesh.e[:,3], mesh.t.shape[0]).mat
        mesh.matInterpTri2Edg = 0.5 * (mat2 + mat3)
    if isAdjoint:
        return mesh.matInterpTri2Edg * value
    else:
        return mesh.matInterpTri2Edg.T * value

