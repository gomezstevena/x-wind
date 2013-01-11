'''
Module for plotting stuff, intended to be used by mesh.py
'''

from numpy import *

from pylab import *
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection, LineCollection

from meshUtils import *

def plotMesh(mesh, detail, alpha):
    v, t, b, e, a, n = mesh.v, mesh.t, mesh.b, mesh.e, mesh.a, mesh.n
    plot(v[e[:,:2], 0].T, v[e[:,:2], 1].T, '-k', alpha=alpha)
    plot(v[b, 0].T, v[b, 1].T, '-k', lw=2, alpha=alpha)
    if detail > 0:
        xt = v[t,:].mean(1)
        xe = v[e[:,:2],:].mean(1)
        plot([xe[:,0], xt[e[:,2],0]], [xe[:,1], xt[e[:,2],1]], '-r')
        plot([xe[:,0], xt[e[:,3],0]], [xe[:,1], xt[e[:,3],1]], '-b')
    if detail > 1:
        plot([xe[:,0], xe[:,0]+.2*n[:,0]], [xe[:,1], xe[:,1]+.2*n[:,1]], '-g')
    axis('scaled')

def plotTriScalar(mesh, phi, norm = None):
    '''
    Contour plot of scalar field phi defined on mesh triangles
    shading is flat, i.e. piecewise constant
    '''
    v, t, b, e, a, n = mesh.v, mesh.t, mesh.b, mesh.e, mesh.a, mesh.n
    assert phi.ndim == 1 and phi.size == t.shape[0]
    norm = norm or Normalize(phi.min(), phi.max())
    p = [Polygon(v[ti]) for ti in t]
    p = PatchCollection(p, norm=norm, edgecolors='none')
    l = LineCollection(v[e[:,:2]], norm=norm)
    p.set_array(phi)
    l.set_array(phi[e[:,2:]].mean(1))
    gca().add_collection(p)
    gca().add_collection(l)
    axis('scaled')
    colorbar(p)

def plotTriVector(mesh, vec, *argc, **argv):
    '''
    Plot of vector field vec of shape (nt, 2) defined on mesh triangles
    '''
    v, t, b, e, a, n = mesh.v, mesh.t, mesh.b, mesh.e, mesh.a, mesh.n
    assert vec.ndim == 2 and vec.shape == (t.shape[0], 2)
    xt = v[t,:].mean(1)
    plot([xt[:,0], xt[:,0] + vec[:,0]], [xt[:,1], xt[:,1] + vec[:,1]], \
         *argc, **argv)
    axis('scaled')

def plotEdgScalar(mesh, phi, norm = None):
    '''
    Contour plot of scalar field phi defined on mesh edges
    shading is flat, i.e. piecewise constant
    '''
    v, t, b, e, a, n = mesh.v, mesh.t, mesh.b, mesh.e, mesh.a, mesh.n
    assert phi.ndim == 1 and phi.size == e.shape[0]
    norm = norm or Normalize(phi.min(), phi.max())
    l = LineCollection(v[e[:,:2]], norm=norm)
    l.set_array(phi)
    gca().add_collection(l)
    axis('scaled')
    colorbar(l)

def plotVrtScalar(mesh, phi, norm=None):
    '''
    Contour plot of scalar field phi defined on mesh vertices
    Represented by circles of proportional size as dual volume area
    '''
    v, t, b, e, a, n = mesh.v, mesh.t, mesh.b, mesh.e, mesh.a, mesh.n
    assert phi.ndim == 1 and phi.size == v.shape[0]
    norm = norm or Normalize(phi.min(), phi.max())
    vrtArea = accumarray(ravel(t.T))(ravel([a, a, a])) / 3.
    vrtR = sqrt(vrtArea / pi) * 0.6
    p = [Circle(vi, vrtRi) for vi, vrtRi in zip(v, vrtR)]
    p = PatchCollection(p, norm=norm, edgecolors='none')
    p.set_array(phi)
    gca().add_collection(p)
    axis('scaled')
    colorbar(p)


class Visualize:
    def __init__(self, v, t, e, fig, win, axesLimit=[-3,3.5,-2,2]):
        self.e = e.copy()
        self.p = [Polygon(v[ti]) for ti in t]
        self.p = PatchCollection(self.p, edgecolors='none')
        self.l = LineCollection(v[e[:,:2]])

        win = win or fig.canvas.manager.window
        if fig is None: fig = gcf()
        fig.clf()
        ax = fig.add_axes([0.02,0.02,.98,.98])
        ax.axis('scaled')
        ax.axis(axesLimit)
        ax.set_autoscale_on(False)
        self.axis, self.fig, self.win = ax, fig, win

        ax.add_collection(self.p)
        ax.add_collection(self.l)
        # ax.add_collection(self.l1)
        # ax.add_collection(self.l2)

    def update(self, title, phi):
        norm = Normalize(phi.min(), phi.max())
        self.p.set_norm(norm)
        self.l.set_norm(norm)
        self.p.set_array(phi)
        self.l.set_array(phi[self.e[:,2:]].mean(1))
        if not self.__dict__.has_key('colorbar'):
            self.colorbar = self.fig.colorbar(self.p)
        self.win.set_title(title)
        #self.fig.canvas.set_window_title(title)
        self.fig.canvas.draw()

