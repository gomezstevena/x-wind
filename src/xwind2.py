import sys
import gtk
import time
import multiprocessing as mp

import gobject
gobject.threads_init()

from matplotlib.figure import Figure
from pylab import *
from numpy import *

from navierstokes import *
from meshVisualize import Visualize
from physical import physical

from matplotlib.backends.backend_gtkagg \
     import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.backends.backend_gtkagg \
     import NavigationToolbar2GTKAgg as NavigationToolbar

from scipy.interpolate import griddata
from util import linNearGrid, freeStream

import meshUtils

import collections


def chooseSolver(v, t, b, Mach, Re):
    if isfinite(Re):
        return NavierStokes(v, t, b, Mach, Re)
    else:
        return Euler(v, t, b, Mach)

SolutionData = collections.namedtuple( 'SolutionData', ['soln', 'time', 'metric'] )

class CommandData(object):
    def __init__(self, name, *args):
        self.name = name
        self.data = args

class InitCommand(CommandData):
    def __init__(self, W0):
        CommandData.__init__(self, 'Init', W0)


class SolverProcess(mp.Process):
    def __init__(self, v, t, b, Mach, Re, out, W0=None ):
        mp.Process.__init__(self)
        self.data = ( v, t, b, Mach, Re)

        self.out = out

        self.dt = 1.0E-3

        self.isRunning = False

        self.W0 = W0


    def run(self):
        v, t, b, Mach, Re = self.data
        self.solver = chooseSolver(v,t, b, Mach, Re)
        
        #self.initField()
        #print 'Starting loop'
        self.initField(self.W0)

        while True:
            if self.isRunning:
                self.solver.integrate( self.solver.time + self.dt )
                data_out = SolutionData( soln=self.solver.soln, time=self.solver.time, metric=self.solver.metric() )
                self.out.put( data_out )

            #if self.out.poll():
            #    self.handleCommand( self.out.recv() )


    def handleCommand(self, command_data):
        print 'got data:', command_data

    def initField(self, W0 = None):
        if W0 is None:
            W0 = self.solver.freeStream()

        self.solver.integrate( 1e-9, W0)
        self.isRunning = True


class SolverGui(object):
    def __init__(self, win, fig, buttons, combobox, queue_length = 2):
        self.win = win
        self.fig = fig

        self.buttons = buttons
        self.combobox = combobox

        self.Mach = 0.3
        self.M0 = self.Mach

        self.Re = 12500
        self.nE = 500
        self.geom = rotate(loadtxt('../data/n0012c.dat'), 2./180*pi)
        # self.geom = array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
        self.buttons['MachDisp'].set_label('Mach = {0}'.format(self.Mach))
        self.buttons['ReDisp'].set_label('Re = {0}'.format(self.Re))
        self.buttons['NEDisp'].set_label('{0} cells'.format(self.nE))


        self.data_queue = mp.Queue(queue_length)

        self.changeGeom()
        gobject.timeout_add(100, self.update)


    def terminateSolver(self):
        'Terminate the solver thread running "self.advance()"'
        if hasattr(self, 'solver') and self.solver.is_alive():
            self.solver.terminate()
            self.data_queue.close()
            self.data_queue = mp.Queue(2)
            #self.data_queue.close()
            self.isKeepRunning = False
        
    def startSolver(self):
        'Start the solver thread running "self.advance()"'
        if self.__dict__.has_key('isKeepRunning'):
            assert self.isKeepRunning == False
        #self.solverThread = threading.Thread(target=self.advance)
        #self.solverThread.daemon = True
        self.isKeepRunning = True
        self.solver.start()
        
    def update(self):
        'Called automatically every second to update screen'
        #print 'trying to update'
        if not self.data_queue.empty():

            try:
                #self.solnLock.acquire()
                #t, soln = self._soln_t, self._soln_copy.copy()
                data = self.data_queue.get(False)
                #print 'got data!!!'

                t = data.time
                self.last_soln = data.soln
                new_metric = data.metric

                field = physical(self.last_soln, self.combobox.get_active_text())
                self.vis.update('MIT xWind  t ={0:10.6f}'.format(t), field)

                self.metric = 0.99*self.metric + 0.01*new_metric
            except e:
                print 'no data available?', e
                #return False

        else:
            pass#print 'data queue is empty'

        return True

    def changeGeom(self):
        'Change in geometry, new solver, new mesh, starting from freestream'
        self.terminateSolver()
        self.v, self.t, self.b = initMesh(self.geom, self.nE)
        print "v:{0}\nt:{1}\nb:{2}\ngeom:{3}".format(self.v.shape, self.t.shape, self.b.shape, self.geom.shape)
        self.vis = Visualize(self.v, self.t, extractEdges(self.v, self.t), self.fig, self.win)
        self.solver = SolverProcess(self.v, self.t, self.b, self.Mach, self.Re, self.data_queue)
        self.xt = triCenters(self.v, self.t)
        nv = len(self.v)
        self.metric = zeros([nv, 2, 2])
        self.startSolver()

    def changeWind(self, *args):
        '''Change in Mach, Re or nE (# element) by changing the solver
           and interpolating old solution'''
        for b in self.buttons.values():
            b.set_sensitive(False)
        self.terminateSolver()

        xt0, W0 = self.xt, self.last_soln
        freeStream0 = freeStream(self.M0)
        axesLimit = ravel(array(fig.axes[0].viewLim).T)

        self.v, self.t, self.b = adaptMesh(self.geom, self.v, self.t, self.b, self.nE, self.metric)
        e = extractEdges(self.v, self.t)
        self.vis = Visualize(self.v, self.t, e, self.fig, self.win, axesLimit)
        self.xt = triCenters(self.v, self.t)
        
        nv = len(self.v)
        self.metric = zeros([nv, 2, 2])


        W0 = linNearGrid(xt0, W0, self.xt )
        W0 += freeStream(self.Mach) - freeStream0

        self.solver = SolverProcess(self.v,self.t,self.b,self.Mach, self.Re, self.data_queue, W0)


        self.startSolver()
        self.buttons['MachDisp'].set_label('Mach = {0}'.format(self.Mach))
        self.buttons['ReDisp'].set_label('Re = {0}'.format(self.Re))
        self.buttons['NEDisp'].set_label('{0} cells'.format(self.nE))
        for b in self.buttons.values():
            b.set_sensitive(True)

    # ------- callback for change in Mach, Re and nE ------- #
    def MachUp(self, *args):
        if self.Mach < 8:
            #self.Mach += 0.1
            #self.changeWind()
            self.setMach(self.Mach + 0.1)

    def MachDown(self, *args):
        if self.Mach > 0.2:
            #self.Mach -= 0.1
            #self.changeWind()
            self.setMach(self.Mach - 0.1)

    def setMach(self, newMach):
        self.M0, self.Mach  = self.Mach, newMach
        self.changeWind()

    def ReUp(self, *args):
        if not isfinite(self.Re):
            self.Re = 25
        else:
            self.Re *= 2
        if self.Re >= 100000:
            self.Re = inf
        self.changeWind()

    def ReDown(self, *args):
        if not isfinite(self.Re):
            self.Re = 51200
        else:
            self.Re /= 2
        if self.Re <= 20:
            self.Re = inf
        self.changeWind()

    def nEUp(self, *args):
        self.nE = int(ceil(self.nE * 1.5))
        self.changeWind()

    def nEDown(self, *args):
        self.nE = int(ceil(self.nE / 1.5))
        self.changeWind()

    # ------- callback for new, open and save ------- #
    def newWind(self, *args):
        pass
    
    def openWind(self, *args):
        chooser = gtk.FileChooserDialog(action=gtk.FILE_CHOOSER_ACTION_OPEN,
                  buttons=(gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL,
                           gtk.STOCK_OPEN,gtk.RESPONSE_OK))
        if chooser.run() == gtk.RESPONSE_OK:
            print chooser.get_filename()
        chooser.destroy()
    
    def saveWind(self, *args):
        chooser = gtk.FileChooserDialog(action=gtk.FILE_CHOOSER_ACTION_SAVE,
                  buttons=(gtk.STOCK_CANCEL,gtk.RESPONSE_CANCEL,
                           gtk.STOCK_SAVE,gtk.RESPONSE_OK))
        if chooser.run() == gtk.RESPONSE_OK:
            self.solnLock.acquire()
            soln = array(self.solutionList)
            self.solnLock.release()
            try:
                m = self.solver.mesh
                fname = chooser.get_filename()
                savez(fname, geom=self.geom, v=m.v, t=m.t, b=m.b, soln=soln)
            except e:
                sys.stderr.write('Error in saveWind: {0}\n'.format(e))
        chooser.destroy()

if __name__ == '__main__':
        
    # =========== create all gtk stuff ============== #
    win = gtk.Window()
    win.set_default_size(1280, 750)
    win.set_title("MIT xWind")

    combobox = gtk.combo_box_new_text()
    combobox.append_text('density')
    combobox.append_text('X-velocity')
    combobox.append_text('Y-velocity')
    combobox.append_text('static pressure')
    combobox.append_text('stagnation pressure')
    combobox.append_text('mach number')
    combobox.append_text('entropy')
    combobox.set_active(0)

    buttons = {}
    buttons['New'] = gtk.ToolButton(gtk.STOCK_NEW)
    buttons['Open'] = gtk.ToolButton(gtk.STOCK_OPEN)
    buttons['Save'] = gtk.ToolButton(gtk.STOCK_SAVE_AS)
    buttons['ReMesh'] = gtk.Button("ReMesh")
    buttons['MachDown'] = gtk.ToolButton(gtk.STOCK_GO_BACK)
    buttons['MachDisp'] = gtk.Button("")
    buttons['MachUp'] = gtk.ToolButton(gtk.STOCK_GO_FORWARD)
    buttons['ReDown'] = gtk.ToolButton(gtk.STOCK_GO_BACK)
    buttons['ReDisp'] = gtk.Button("")
    buttons['ReUp'] = gtk.ToolButton(gtk.STOCK_GO_FORWARD)
    buttons['NEDown'] = gtk.ToolButton(gtk.STOCK_GO_BACK)
    buttons['NEDisp'] = gtk.Button("")
    buttons['NEUp'] = gtk.ToolButton(gtk.STOCK_GO_FORWARD)

    fig = Figure()
    canvas = FigureCanvas(fig)  # a gtk.DrawingArea
    nav = NavigationToolbar(canvas, win)
    nav.set_size_request(250, 35);

    sep = [gtk.SeparatorToolItem() for i in range(10)]

    toolbar = gtk.HBox(False, 2)
    toolbar.pack_start(buttons['New'], False, False, 0)
    toolbar.pack_start(buttons['Open'], False, False, 0)
    toolbar.pack_start(buttons['Save'], False, False, 0)
    toolbar.pack_start(sep[0], False, False, 0)
    toolbar.pack_start(nav, False, False, 0)
    toolbar.pack_start(sep[1], False, False, 0)
    toolbar.pack_start(combobox, False, False, 0)
    toolbar.pack_start(sep[2], False, False, 0)
    toolbar.pack_start(buttons['ReMesh'], False, False, 0)
    toolbar.pack_start(sep[3], False, False, 0)
    toolbar.pack_start(buttons['MachDown'], False, False, 0)
    toolbar.pack_start(buttons['MachDisp'], False, False, 0)
    toolbar.pack_start(buttons['MachUp'], False, False, 0)
    toolbar.pack_start(sep[4], False, False, 0)
    toolbar.pack_start(buttons['ReDown'], False, False, 0)
    toolbar.pack_start(buttons['ReDisp'], False, False, 0)
    toolbar.pack_start(buttons['ReUp'], False, False, 0)
    toolbar.pack_start(sep[5], False, False, 0)
    toolbar.pack_start(buttons['NEDown'], False, False, 0)
    toolbar.pack_start(buttons['NEDisp'], False, False, 0)
    toolbar.pack_start(buttons['NEUp'], False, False, 0)

    vbox = gtk.VBox(False, 3)
    vbox.pack_start(toolbar, False, False, 0)
    vbox.add(canvas)

    # =========== create solver ============== #
    xwind = SolverGui(win, fig, buttons, combobox)
    buttons['New'].connect("clicked", xwind.newWind)
    buttons['Open'].connect("clicked", xwind.openWind)
    buttons['Save'].connect("clicked", xwind.saveWind)
    buttons['ReMesh'].connect("clicked", xwind.changeWind)
    buttons['MachUp'].connect("clicked", xwind.MachUp)
    buttons['MachDown'].connect("clicked", xwind.MachDown)
    buttons['ReUp'].connect("clicked", xwind.ReUp)
    buttons['ReDown'].connect("clicked", xwind.ReDown)
    buttons['NEUp'].connect("clicked", xwind.nEUp)
    buttons['NEDown'].connect("clicked", xwind.nEDown)

    win.add(vbox)

    def destroyAll(*x):
        xwind.terminateSolver()
        gtk.main_quit()

    win.connect("destroy", destroyAll)

    from pathdrawer import Path
    pather = Path(xwind, nav)


    # ================ Run program ================= #
    win.show_all()
    gtk.main()

