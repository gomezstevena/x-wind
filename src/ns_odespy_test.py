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
nE = 750
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

U0 = solver.ode.y
J0 = solver.J(U0)
ia = J0.indptr + 1
ja = J0.indices

import odespy
ode = odespy.Lsodes( solver.ddt, jac_column=solver.J_col_dumb )
ode.set_initial_condition( solver.ode.y )

time_steps = r_[:nsteps]*dt

U, T = ode.solve(time_steps)

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
