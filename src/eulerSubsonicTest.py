import matplotlib
matplotlib.use('Agg')
import os
from pylab import *
from numpy import *
from scipy.integrate import ode
from scipy.interpolate import griddata

from mesh import *
from euler import Euler

geom = rotate(loadtxt('../data/n0012c.dat'), 30*pi/180)
nE = 1000
dt = 0.001
Mach = 0.3
HiRes = 1.

if not os.path.exists('fig'): os.mkdir('fig')

for iAdapt in range(4):
    print 'Adapt cycle {0}'.format(iAdapt)
    
    if iAdapt == 0:
        v, t, b = initMesh(geom, nE)
        solver = Euler(v, t, b, Mach, HiRes)
        solver.integrate(1E-8, solver.freeStream())
    else:
        xt0, W0 = solver.mesh.xt(), solver.soln
        v, t, b = adaptMesh(geom, v, t, b, nE, metric)
        solver = Euler(v, t, b, Mach, HiRes)
        W0 = griddata(xt0, W0, solver.mesh.xt(), method='nearest')
        solver.integrate(1E-8, W0)

    metric = zeros([v.shape[0], 2, 2]) # metric for next adaptation

    for T in arange(1,21) * dt:
        solver.integrate(T)
        metric += solver.metric()
        clf()
        solver.mesh.plotTriScalar(solver.soln[:,0])
        solver.mesh.plotMesh(alpha=0.2)
        axis([-2,4,-2.5,2.5])
        savefig('fig/eulerSubsonic_adapt{1:1d}_{0:0.6f}.png'.format(T, iAdapt))
    
