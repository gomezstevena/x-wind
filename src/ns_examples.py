import matplotlib
import os
from pylab import *
from numpy import *
from scipy.integrate import ode
from scipy.interpolate import griddata

from matplotlib import tri as Tri

from mesh import *
from navierstokes import *

geom = rotate(loadtxt('../data/n0012c.dat'), 30*pi/180)
# geom = transpose([cos(linspace(0,2*pi,33)), sin(linspace(0,2*pi,33))])
# geom[-1] = geom[0]
nE = 5000
dt = 0.005
nsteps = 0
Mach = 0.3
Re = 10000
diameter = 3

if not os.path.exists('fig'): os.mkdir('fig')
if not os.path.exists('data'): os.mkdir('data')


v, t, b = initMesh(geom, nE, diameter)
solver = NavierStokes(v, t, b, Mach, Re )
# solver = Euler(v, t, b, Mach, HiRes)
solver.integrate(1E-8, solver.freeStream())

    

solution = zeros([nsteps, solver.nt, 4])
metric = zeros([v.shape[0], 2, 2]) # metric for next adaptation

T = Tri.Triangulation( v[:,0], v[:,1], t )
W = solver.soln

tripcolor( v[:,0], v[:,1], triangles=t, facecolors=W[:,0])


for istep, T in enumerate(arange(1,nsteps+1) * dt):
        print 'istep: {0}'.format(istep)
        solver.integrate(T)
        solution[istep] = solver.soln.copy()
        metric += solver.metric()

