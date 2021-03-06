from numpy import *
from scipy.sparse import csc_matrix, csr_matrix, bsr_matrix

def block_diags(blocks):
    assert blocks.ndim == 3
    n = blocks.shape[0]
    return bsr_matrix((blocks, r_[:n], r_[:n+1]))

def matrixMult(*mats):
    assert len(mats) > 1
    assert len(set(m.shape[0] for m in mats)) == 1
    prod = mats[0]
    for m in mats[1:]:
        prod = (prod[:,:,:,newaxis] * m[:,newaxis,:,:]).sum(2)
    return prod

class accumarray:
    '''
    Matlab accumarray(indx, value), in sparse matrix form for reuse:
    Method 1: accumarray(indx)(value)
    Method 2: a = accumarray(indx)
              a(value1)
              a(value2)
              ...
    '''
    def __init__(self, indx, n=None):
        if n is None: n = indx.max() + 1
        self.mat = csc_matrix((ones(indx.size), (indx, r_[:indx.size])), \
                              shape = (n, indx.size))

    def __call__(self, value):
        shape = value.shape
        value = value.reshape([shape[0], -1])
        return (self.mat * value).reshape((-1,) + shape[1:])

def indexMap(indx, n = None, weight = 1):
    if n is None: n = indx.max() + 1
    shape = (indx.size, n)

    vals = repeat(weight, indx.size)
    return csr_matrix( (vals, (r_[:indx.size], indx) ), shape=shape  )

def dot_each(a, b):
    return einsum('ni, ni -> n', a, b)

def dot_all(a, b):
    return dot( a.ravel(), b.ravel() )

def invertMap(m):
    '''
    Input m is an (ni, nj) array that maps
       each of the ni elements [0,1,...,ni-1] to nj elements, whose
       indices are in the corresponding row of m.
    Returns a tuple (indptr, indices).  Each element j is mapped from
       indptr[j+1] - indptr[j] elements, whose indices are
       indices[indptr[j]:indptr[j+1]]
    '''
    i = r_[:m.shape[0]][:,newaxis] + zeros([1, m.shape[1]])
    mat = csc_matrix((ones(m.size), (ravel(i), ravel(m))))
    return mat.indptr, mat.indices

def centerDiameter(xy):
    xy = asarray(xy, float)
    diameter = sqrt(sum((xy.max(0) - xy.min(0))**2))
    center = 0.5 * (xy.max(0) + xy.min(0))
    return center, diameter

def rotate(xy, angle, origin=[0.,0.]):
    mat = array([[cos(angle), sin(angle)], [-sin(angle), cos(angle)]])
    xy = ((xy - array(origin))[:,newaxis,:] * mat).sum(-1)
    return xy + origin

def triArea(v, t):
    '''
    Inputs v, t have the same format as ani2D
        v[:,0] are the x coordinates
        v[:,1] are the y coordinates
        t[:,0], t[:,1) and t[:,2] are the p-indices of the triangle vertices.
    '''
    dp1 = v[t[:,1],:] - v[t[:,0],:]
    dp2 = v[t[:,2],:] - v[t[:,0],:]
    crossProd = dp1[:,0] * dp2[:,1] - dp1[:,1] * dp2[:,0]
    return .5 * abs(crossProd)


def edgNormal(v, e):
    '''
    Inputs v, e have the same format as ani2D
        v[:,0] are the x coordinates
        v[:,1] are the y coordinates
        e[:,0], e[:,1] are the p-indices of the edge end points.
    '''
    dp = v[e[:,1],:] - v[e[:,0],:]
    dp[:,1] *= -1
    return dp[:,[1,0]]


def triCenters(v, t):
    return v[t,:].mean(1)

if __name__ == '__main__':
    n = 25
    indx = randint(n, size=100)

    A = accumarray(indx, n = n).mat
    B = indexMap(indx, n = n)

