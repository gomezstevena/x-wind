import multiprocessing as mp


import matplotlib
matplotlib.use('GTKAgg')

from matplotlib.pyplot import *
from numpy import *

from navierstokes import *
from meshVisualize import Visualize

from mesh import initMesh
from physical import physical




def chooseSolver(v, t, b, Mach, Re):
	if isfinite(Re):
		return NavierStokes(v, t, b, Mach, Re)
	else:
		return Euler(v, t, b, Mach)


class SolutionData(object):
	def __init__(self, soln, time ):
		self.soln = soln
		self.time = time


class SolverProcess(mp.Process):
	def __init__(self, v, t, b, Mach, Re, out_queue ):
		mp.Process.__init__(self)
		self.data = ( v, t, b, Mach, Re)

		self.out = out_queue

		self.dt = 1.0E-3


	def run(self):
		v, t, b, Mach, Re = self.data
		self.solver = chooseSolver(v,t, b, Mach, Re)
		self.solver.integrate( 1e-9, self.solver.freeStream() )

		print 'Starting loop'

		while True:

			self.solver.integrate( self.solver.time + self.dt )

			data_out = SolutionData( self.solver.soln, self.solver.time )
			self.out.put( data_out )


class SimpleGui(object):
	def __init__(self, qlen = 2):
		self.fig = figure()
		self.fig.show()

		self.Mach = 0.3
		self.Re = 12500
		self.ne = 2500
		self.geom = rotate(loadtxt('../data/n0012c.dat'), 2./180*pi)

		self.data_queue = mp.Queue(qlen)
		v, t, b = initMesh(self.geom, self.ne )
		self.vis = Visualize(v, t, extractEdges(v, t), self.fig, None)

		self.solver_proc = SolverProcess(v, t, b, self.Mach, self.Re, self.data_queue)

	def run(self):

		self.solver_proc.start()

		while True:
			if not self.data_queue.empty():
				try:
					new_data = self.data_queue.get(False)

					t = new_data.time
					soln = new_data.soln
					field = physical(soln, 'X-velocity')
					self.vis.update( 'MIT XWind, t = {0:.5f}'.format(t), field)

				except e:
					print 'no data available?'



if __name__ == '__main__':
	
	gui = SimpleGui()

	gui.run()