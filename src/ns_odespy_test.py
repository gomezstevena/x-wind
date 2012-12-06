import matplotlib
matplotlib.use('Agg')
import os
from pylab import *
from numpy import *
from scipy.integrate import ode
from scipy.interpolate import griddata

from mesh import *
from navierstokes import *

geom = rotate(loadtxt('../data/n0012c.dat'), 30*pi/180)
# geom = transpose([cos(linspace(0,2*pi,33)), sin(linspace(0,2*pi,33))])
# geom[-1] = geom[0]
nE = 1500
dt = 0.002
nsteps = 5
Mach = 0.3
Re = 10000
HiRes = 1.
diameter = 3

if not os.path.exists('fig'): os.mkdir('fig')
if not os.path.exists('data'): os.mkdir('data')


v, t, b = initMesh(geom, nE, diameter)
solver = NavierStokes(v, t, b, Mach, Re, HiRes)
# solver = Euler(v, t, b, Mach, HiRes)
solver.integrate(1E-8, solver.freeStream())
    

solution = zeros([nsteps, solver.nt, 4])
metric = zeros([v.shape[0], 2, 2]) # metric for next adaptation

"""
for istep, T in enumerate(arange(1,nsteps+1) * dt):
    print 'istep: {0}'.format(istep)
    solver.integrate(T)
    solution[istep] = solver.soln.copy()
    metric += solver.metric()
"""

"""
U0 = solver.ode.y
J0 = solver.J(U0).tocsr(True)
J0.sort_indices()
ia = [int(i) for i in (J0.indptr + 1)]
ja = [int(j) for j in (J0.indices + 1)]

rtol, atol = 1e-4, 1e-4

import sys
sys.path.append('../odespy/build/lib.linux-x86_64-2.7/')
from odespy import *

method = Lsodes

ode = method( solver.ddt, rtol=rtol, atol=atol, jac_column=solver.J_col_dumb, ia=ia, ja=ja, adams_or_bdf='bdf' )
ode.set_initial_condition( solver.ode.y )

time_steps = [0, dt]

U, T = ode.solve(time_steps)
"""
U0 = solver.ode.y
from integrator import CrankNicolson
CN = CrankNicolson(solver.ddt, solver.J)
CN.integrate(1e-8, U0, 0)
CN.integrate( dt )
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
