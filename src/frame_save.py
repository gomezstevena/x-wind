import numpy as np

import os

def ensure_dir(path):
	if not os.path.isdir(path):
		os.mkdir(path)

class FrameSaver(object):
	def __init__(self, v, t, b, path = './frames/', prepend_name=''):
		self.step = 0

		self.verts, self.tri, self.bnds = v, t, b

		self.path = path
		ensure_dir(path)


		self.prepend_name = prepend_name + '_' if prepend_name!='' else ''

		fname = '{0}mesh_data.npz'.format(self.path + self.prepend_name)
		np.savez( fname, verts=self.verts, tri=self.tri, bnds=self.bnds )

	def update_data(self, data):
		if hasattr(self, 'data_shape'):
			assert data.shape == self.data_shape
		else:
			self.data_shape = data.shape

		fname = '{0}{1:06d}.npy'.format( self.path + self.prepend_name, self.step )
		np.save( fname, data )

		self.step += 1

import matplotlib
matplotlib.use('Agg')
from mesh import Mesh
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import meshVisualize as mviz

class FrameToMovie(object):
	def __init__(self, path = './frames/', prepend_name = '', fps=10, filetype='mp4'):
		self.path = path
		self.prepend_name = prepend_name + '_' if prepend_name!='' else ''
		self.fps = fps
		self.filetype = filetype if filetype.startswith('.') else ('.' + filetype )

		self.mesh_data_fname = '{0}mesh_data.npz'.format(self.path + self.prepend_name)
		f = np.load( self.mesh_data_fname )
		t = f['tri']
		v = f['verts']
		b = f['bnds']
		f.close()
		self.mesh = Mesh(v,t,b)

		self._get_data_files()

	def _get_data_files(self):
		files = [ f for f in os.listdir(self.path) if f.startswith(self.prepend_name) ]
		print files
		files.remove( 'mesh_data.npz' )
		n = len(self.prepend_name)

		files.sort( key = lambda x: int(x[n:-4]) )
		self.files = files
		self.nsteps = len(self.files)

	def _make_plot(self, indx, plotter, extractor, norm, axes = None, **kwargs ):
		assert indx <= self.nsteps
		fname = self.files[indx]
		data = np.load(self.path + fname)
		plt_data = extractor(data)

		plotter(self.mesh, plt_data, norm = norm)

		if axes:
			plt.axes( axes )

		plt.savefig( self.path + fname[:-4] + '.png', **kwargs )

	def make_plots(self, plotter = None, extractor = 0, norm = None, axes = None, **kwargs):
		if type(extractor) is int:
			col = extractor
			extractor = lambda d: d[..., col]

		assert callable(extractor)

		fname0 = self.files[0]
		data0 = np.load(self.path + fname0)
		print data0, data0.shape
		plt_data0 = extractor(data0)
		(n,)  = plt_data0.shape

		if plotter is None:
			if n == self.mesh.nt:
				print 'plotting on triangles'
				plotter = mviz.plotTriScalar
			elif n == self.mesh.ne:
				print 'plotting on edges'
				plotter = mviz.plotEdgScalar
			elif n == len(self.mesh.v):
				print 'plotting on vertices'
				plotter = mviz.plotVrtScalar
			else:
				raise ValueError('Cannot infer plotter')


		norm = norm or Normalize( plt_data0.min(), plt_data0.max() )

		ensure_dir(self.path)

		for i in xrange(self.nsteps):
			print 'Making plot for step', i,
			self._make_plot( i, plotter, extractor, norm=norm, axes=axes, **kwargs )
			print '... done'

		f_reg = self.path + "%06d.png"
		outname = self.path + self.prepend_name + "video.mp4"
		os.system( 'ffmpeg -i {f_reg} -r {FPS} -o {outname}'.format( f_reg=f_reg, FPS=self.fps, outname=outname) )



