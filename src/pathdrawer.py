import numpy as np
from util import *
class Path(object):



	def __init__(self, xwind_gui):
		self.path_active = False
		self.path = []
		self.xwind_gui = xwind_gui

		event_dict = { \
		'button_press_event': self.onClick,
		'button_release_event': self.onRelease,
		'motion_notify_event': self.onMouseMove,
		'figure_enter_event': self.clearPath,
		'figure_leave_event': self.clearPath
		}

		for event_name, event_func in event_dict.iteritems():
			xwind_gui.fig.canvas.mpl_connect(event_name, event_func)


	def clearPath(self, event = None):
		self.path_active = False
		self.path = []

	def onClick(self, event):
		print 'A Click!'
		if not self.path_active:
			self.path_active = True
			self.path = [ (event.xdata, event.ydata) ]

	def onRelease(self, event):
		print 'A release!'
		if self.path_active:
			self.path.append( (event.xdata, event.ydata) )
			path = self.path
			self.clearPath()
			self.finishPath( path )
			

	def onMouseMove(self, event):
		#print 'A movement!'
		if self.path_active:
			print 'A drag'
			self.path.append( (event.xdata, event.ydata) )


	def finishPath(self, path):
		path.append( path[0] )
		path = np.array( path )

		path = pathSmoother(path)

		print path, path.shape

		np.save('example_path.npy', path)
		
		self.xwind_gui.geom = np.vstack([self.xwind_gui.geom, path])
		self.xwind_gui.changeGeom()
		
 

from IPython import embed
from matplotlib.pyplot import *
from scipy import signal
def pathSmoother( path ):
	x, y = path.T
	dsv = np.diff(path, axis = 0)
	dsm = dot_each( dsv, dsv )

	sm = np.r_[ 0.0, np.cumsum( dsm ) ]

	n = path.shape[0]
	p , ss = signal.resample(path, 100, t = sm, axis=0, window = ('gaussian', 7) )

	p = np.vstack([p, p[0]])

	return p

if __name__ == '__main__':
	
	path = np.load('example_path.npy')
	path_smooth = pathSmoother( path )