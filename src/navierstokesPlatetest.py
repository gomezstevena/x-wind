import matplotlib
matplotlib.use('Agg')
import os
from pylab import *
from numpy import *
from scipy.integrate import ode
from scipy.interpolate import griddata

from mesh import *
from navierstokes import NavierStokes

geom = [[0, -1], [8, -1], [4, -1.01], [0, -1],
        [0,  1], [8,  1], [4,  1.01], [0,  1], [0,-1]]
nE = 5000
dt = 0.001
nsteps = 50
Mach = 0.3
Re = 10
HiRes = 1.

if not os.path.exists('fig'): os.mkdir('fig')
if not os.path.exists('data'): os.mkdir('data')

for iAdapt in range(10):
    print 'Adapt cycle {0}'.format(iAdapt)
    
    if iAdapt == 0:
        v, t, b = initMesh(geom, nE)
        solver = NavierStokes(v, t, b, Mach, Re, HiRes)
        solver.integrate(1E-8, solver.freeStream())
    else:
        xt0, W0 = solver.mesh.xt(), solver.soln
        v, t, b = adaptMesh(geom, v, t, b, nE, hessian)
        solver = NavierStokes(v, t, b, Mach, Re, HiRes)
        W0 = griddata(xt0, W0, solver.mesh.xt(), method='nearest')
        solver.integrate(1E-8, W0)

    solution = zeros([nsteps, solver.nt, 4])
    hessian = zeros([solver.nt, 2, 2]) # metric for next adaptation

    for istep, T in enumerate(arange(1,nsteps+1) * dt):
        solver.integrate(T)
        solution[istep] = solver.soln.copy()
        hessian += solver.hessian()

    fname = 'data/navierstokesPlate_adapt{0:1d}.npz'.format(iAdapt)
    savez(fname, geom=array(geom), v=v, t=t, b=b, soln=solution)
    
