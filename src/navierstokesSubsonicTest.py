import matplotlib
#matplotlib.use('Agg')
import os
from pylab import *
from numpy import *
from scipy.integrate import ode
from scipy.interpolate import griddata

from mesh import *
from navierstokes import *

geom = rotate(loadtxt('../data/n0012c.dat'), 5*pi/180)
# geom = transpose([cos(linspace(0,2*pi,33)), sin(linspace(0,2*pi,33))])
# geom[-1] = geom[0]
nE = 5000
dt = 0.005
nsteps = 25
Mach = 0.3
Re = 10000
HiRes = 1.
diameter = 3

from IPython import embed

ion()

if not os.path.exists('fig'): os.mkdir('fig')
if not os.path.exists('data'): os.mkdir('data')

for iAdapt in range(1):
    print 'Adapt cycle {0}'.format(iAdapt)
    
    if iAdapt == 0:
        v, t, b = initMesh(geom, nE, diameter)
        solver = NavierStokes(v, t, b, Mach, Re)
        # solver = Euler(v, t, b, Mach, HiRes)
        solver.integrate(1E-8, solver.freeStream())
    else:
        xt0, W0 = solver.mesh.xt(), solver.soln
        v, t, b = adaptMesh(geom, v, t, b, nE, metric, diameter)
        solver = NavierStokes(v, t, b, Mach)
        # solver = Euler(v, t, b, Mach, HiRes)
        W0 = griddata(xt0, W0, solver.mesh.xt(), method='nearest')
        solver.integrate(1E-8, W0)

    solution = zeros([nsteps, solver.nt, 4])
    metric = zeros([v.shape[0], 2, 2]) # metric for next adaptation

    
    #fig = figure()
    #solver.mesh.plotMesh(0, 0.2)
    #draw()
    #pause(0.01)

    for istep, T in enumerate(arange(1,nsteps+1) * dt):
        print 'istep: {0}'.format(istep)
        solver.integrate(T)
        solution[istep] = solver.soln.copy()
        metric += solver.metric()

        #WPlot = solution[istep][:,0]
        #solver.mesh.plotTriScalar(WPlot)
        #pause(0.001)



    """
    fname = 'data/navierstokesSubsonic_adapt{0:1d}.npz'.format(iAdapt)
    savez(fname, geom=array(geom), v=v, t=t, b=b, soln=solution)
    
    figure(figsize=(16,9))
    solver.mesh.plotTriScalar(solver.soln[:,1])
    solver.mesh.plotMesh(alpha=0.2)
    axis([-1,2.5,-1.5,1.5])
    savefig('fig/navierstokesSubsonic_adapt{0:1d}.png'.format(iAdapt))
    axis([-.1,.3,-.2,.2])
    savefig('fig/navierstokesSubsZoom_adapt{0:1d}.png'.format(iAdapt))  
    """
