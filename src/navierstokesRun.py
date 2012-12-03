import os
import sys
from numpy import *
from scipy.integrate import ode
from scipy.interpolate import griddata

from mesh import *
from navierstokes import NavierStokes

nE = 5000
dt = 0.0005
nsteps = 2000
Mach = 0.3
Re = 10000
HiRes = 1.

z = load('data/navierstokesInit.npz')
geom, v, t, b, soln = z['geom'], z['v'], z['t'], z['b'], z['soln']

solver = NavierStokes(v, t, b, Mach, Re, HiRes)
solver.integrate(1E-8, soln[-1])

for istep, T in enumerate(arange(1,nsteps+1) * dt):
    solver.integrate(T)
    sys.stdout.write('t = {0}\n'.format(solver.time)); sys.stdout.flush()
    fname = 'data/navierstokesStep{0:06d}.npz'.format(istep)
    savez(fname, geom=array(geom), v=v, t=t, b=b, soln=solver.soln)

