from numpy import *
from subprocess import Popen, PIPE

from meshUtils import *

def aftMesh(xy, d=None):
    '''
    xy should be an n by 2 array describing the boundary:
    1. The boundary is a union of loops.
    2. In each loop, the first node and the last node must be identical.
       This fact is used to distinguish different loops.
    3. When moving to the next node within one loop,
       the fluid domain must be located on the right.
    4. Loops can overlap in any way, in this case the shared node(s)
       must be presented in each loop.

    Returns v, t, b
    v: nv by 2 float array, vertices
    t: nt by 3 int array, triangles
    b: nt by 2 int array, boundaries
    '''
    xy = array(xy, float)
    assert xy.shape[1] == 2
    assert (xy[-1] == xy[0]).all()  # that's right, comparing floating poinst
    center, diameter = centerDiameter(xy)
    if d is not None: diameter = d
    # far field
    s = linspace(0, 2*pi, 17)
    s[-1] = 0
    far = transpose([sin(s), cos(s)]) * diameter * 10 + center
    xy = vstack([xy, far])
    #
    proc = Popen('./aftMesh', stdin=PIPE, stdout=PIPE)
    proc.stdin.write('{0}\n'.format(xy.shape[0]))
    for x, y in xy:
        proc.stdin.write('{0} {1}\n'.format(x, y))
    return readOutput(proc.stdout)


def mbaMesh(v, t, b, nE, metric):
    '''
    Both input and return are in the form of v, t, b
    v: nv by 2 float array, vertices
    t: nt by 3 int array, triangles
    b: nt by 2 int array, boundaries
    Additional input:
    nE: desired number of elements
    metric: nv by 3 float array controlling density
            [metric[0] metric[2]]
            [metric[2] metric[1]]
    '''
    t, b = t + 1, b + 1
    #
    proc = Popen('./mbaMesh', stdin=PIPE, stdout=PIPE)
    proc.stdin.write('{0} {1} {2} {3}\n'.format(nE, \
                     v.shape[0], t.shape[0], b.shape[0]))
    for x, y in v:
        proc.stdin.write('{0} {1}\n'.format(x, y))
    for i1, i2, i3 in t:
        proc.stdin.write('{0} {1} {2}\n'.format(i1, i2, i3))
    for i1, i2 in b:
        proc.stdin.write('{0} {1}\n'.format(i1, i2))
    for mxx, myy, mxy in metric:
        proc.stdin.write('{0} {1} {2}\n'.format(mxx, myy, mxy))
    PASSCODE = '=== Output Data Starts Here 09887654321 ==='
    lines = [proc.stdout.readline()]
    while lines[-1] != '' and PASSCODE not in lines[-1]:
        lines.append(proc.stdout.readline())
    if lines[-1] != '':
        return readOutput(proc.stdout)
    else:
        print ''.join(lines)


def readOutput(f):
    '''
    Used in aftMesh and mbaMesh, reads FORTRAN output of mesh
    '''
    nv, nt, nb = array(f.readline().strip().split(), int)
    lines = f.readlines()
    v = array([l.strip().split() for l in lines[:nv]], float)
    t = array([l.strip().split() for l in lines[nv:nv+nt]], int)
    b = array([l.strip().split() for l in lines[nv+nt:nv+nt+nb]], int)
    assert v.shape[1] == 2 and t.shape[1] == 3 and b.shape[1] == 2
    # shift for python indexing
    return v, t-1, b-1



