import numpy as np
from util import *

from IPython import embed

class Path(object):

	def __init__(self, xwind_gui, nav):
		self.path_active = False
		self.path = []
		self.xwind_gui = xwind_gui
		self.nav = nav

		event_dict = { \
		'button_press_event': self.onClick,
		'button_release_event': self.onRelease,
		'motion_notify_event': self.onMouseMove,
		'figure_enter_event': self.clearPath,
		'figure_leave_event': self.clearPath,
		'draw_event': self.onDraw
		}

		for event_name, event_func in event_dict.iteritems():
			xwind_gui.fig.canvas.mpl_connect(event_name, event_func)

		self.path_plot = ( ( xwind_gui.fig.get_axes()[0] ).plot(-1000,-1000, '-k', linewidth=2) )[0]


	def clearPath(self, event = None):
		self.path_active = False
		self.path = []

	def onClick(self, event):
		#print 'A Click!'
		if not self.path_active and not self.nav._active:
			self.path_active = True
			self.path = [ (event.xdata, event.ydata) ]

	def onRelease(self, event):
		#print 'A release!'
		if self.path_active:
			self.path.append( (event.xdata, event.ydata) )
			path = self.path
			self.finishPath( path )

			self.clearPath()
			
	def onMouseMove(self, event):
		#print 'A movement!'
		if self.path_active:
			#print 'A drag'
			self.path.append( (event.xdata, event.ydata) )
			#self.onDraw(None)

	def finishPath(self, path):
		path.append( path[0] )
		path = np.array( path )

		#np.save('raw_path.npy', path)
		path = fixPath(path)
		print path
		print 'path.shape =', path.shape
		#np.save('example_path.npy', path)
		self.xwind_gui.geom = np.vstack([self.xwind_gui.geom, path])
		self.xwind_gui.changeGeom()

	def onDraw(self, event):
		if self.path:
			self.path_plot.set_data( np.array(self.path).T )
			self.xwind_gui.fig.canvas.draw()
		
 

#from IPython import embed
from matplotlib.pyplot import *
from scipy import signal
def pathSmoother( path ):
	x, y = path.T
	dsv = np.diff(path, axis = 0)
	dsm = dot_each( dsv, dsv )

	sm = np.r_[ 0.0, np.cumsum( dsm ) ]

	n = path.shape[0]
	p , ss = signal.resample(path, 50, t = sm, axis=0, window = ('gaussian', 10) )

	p = np.vstack([p, p[0]])

	return p

def selfIntersect( path ):
	n = len(path)

	result = False
	intersects = []

	for i in xrange(n-1):
		pi = path[i:i+2]
		for j in xrange(i+2, n-1):
		
			pj = path[j:j+2]

			inter, point = segmentIntersect(pi, pj)
			if inter:
				result = True
				intersects.append( (i, j, point) )


	return result, intersects

def splitPath(path, points):
	n_inter = len(points)

	all_paths = []

	for i,j, p in points:
		first = path[:i]
		rest = path[j+1:]

		new_path = np.vstack( [first, p, rest ] )
		all_paths.append( new_path )

		other_path = path[i+1:j]
		other_path = np.vstack( [p, other_path[::-1], p] )
		all_paths.append( other_path )

	return all_paths

def joinPath(path, ij):
	n_inter = len(ij)

	if n_inter > 0:
		i, j = ij[0]
		beg, mid, end = path[:i+1], path[i+1:j+1], path[j+1:]

		ij_new = (j-1) - ij[1:,::-1]
		new_mid = joinPath( mid[::-1], ij_new )

		new_path = np.vstack( [beg, new_mid, end] )

		return new_path

	else:
		return path


def segmentIntersect( l1, l2 ):
	p = l1[0]
	r = l1[1]-l1[0]
	
	q = l2[0]
	s = l2[1]-l2[0]

	rxs = np.cross(r, s)
	if abs(rxs) < 1e-5:
		return False, None

	t = np.cross(q-p, s)/ rxs
	u = np.cross(q-p, r)/ rxs

	#embed()

	in_segment = ( 0<t<1 and 0<u<1 )
	if in_segment:
		loc = p + t*r
	else:
		loc = None

	return in_segment, loc

def handleIntersect(path):
	intersect, points = selfIntersect(path)
	if intersect:
		ij = np.array( [ (i,j) for (i,j,p) in points ] )
		new_path = joinPath(path, ij )
		return new_path
	else:
		return path


def fixPath( path ):
	path_i = handleIntersect(path)
	path_is = pathSmoother(path_i)

	return path_is

if __name__ == '__main__':
	
	path_raw = np.load('raw_path.npy')
	path = pathSmoother( path_raw )

	"""
	l1 = np.array( [ [0,0], [1,1.] ] )
	l2 = np.array( [ [0,.5], [.5, 0] ] )

	plot( l1[:,0], l1[:,1],'-o' ,l2[:,0], l2[:,1], '-o' )

	b, p = segmentIntersect(l1, l2)"""

	figure()
	#plot( path_raw[:,0], path_raw[:,1], '-', path[:,0], path[:,1], '-x' )
	#plot( *path.T )
	#savefig('smoothing.pdf')

	path_si = handleIntersect(path)


	path_is = handleIntersect(path_raw)
	path_is = pathSmoother(path_is)

	path_sis = pathSmoother(path_si)
	#plot( c_[ path[i,0], path[i+1,0] ], c_[ path[i,1], path[i+1,1] ], 's')

	plot( path_raw[:,0], path_raw[:,1], '--b')
	plot( path_is[:,0], path_is[:,1], '-g' )
	plot( path_si[:,0], path_si[:,1], '-r')
	plot( path_sis[:,0], path_sis[:,1], '-k')

	legend( ['raw path', 'intersect -> smooth', 'smooth -> intersect', 'smooth -> intersect -> smooth'] )

	'''new_paths = splitPath(path, points)
	for npa in new_paths:
		plot( npa[:,0], npa[:,1], '--o' )

	savefig('splitting.pdf')'''