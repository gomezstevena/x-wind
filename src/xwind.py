import sys
import gtk
import time
import threading

import gobject
gobject.threads_init()

from matplotlib.figure import Figure
from pylab import *
from numpy import *
from scipy.interpolate import griddata

from navierstokes import *
from meshVisualize import Visualize
from physical import physical

from matplotlib.backends.backend_gtkagg \
     import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.backends.backend_gtkagg \
     import NavigationToolbar2GTKAgg as NavigationToolbar


def chooseSolver(v, t, b, Mach, Re):
    if isfinite(Re):
        return NavierStokes(v, t, b, Mach, Re, HiRes=1)
    else:
        return Euler(v, t, b, Mach, HiRes=0)


class SolverGuiCoupler:
    def __init__(self, win, fig, buttons, combobox):
        self.win = win
        self.fig = fig

        self.buttons = buttons
        self.combobox = combobox

        self.Mach = 0.5
        self.Re = inf #12800
        self.nE = 250
        self.geom = rotate(loadtxt('../data/n0012c.dat'), .5)
        # self.geom = rotate(loadtxt('../data/default.dat'), 0)
        # self.geom = array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]])
        self.buttons['MachDisp'].set_label('Mach = {0}'.format(self.Mach))
        self.buttons['ReDisp'].set_label('Re = {0}'.format(self.Re))
        self.buttons['NEDisp'].set_label('{0} cells'.format(self.nE))

        self.changeGeom()
        gobject.timeout_add(1000, self.update)

    def advance(self):
        'Workhorse subroutine run in separate solver thread'
        self.solnLock = threading.Lock()
        c, d = centerDiameter(self.geom)
        self.dt = 0.02 * d / 300.
        nV = self.solver.mesh.v.shape[0]
        self.metric = zeros([nV, 2, 2]) # metric for adaptation
        self.solutionList = []
        while self.isKeepRunning:
            self.solver.integrate(self.solver.time + self.dt)
            # store a copy
            self.solnLock.acquire()
            self._soln_t = self.solver.time
            self._soln_copy = self.solver.soln.copy()
            self.solutionList.append(self._soln_copy)
            self.solnLock.release()
            # exponential moving average of metric
            self.metric = 0.99 * self.metric + 0.01 * self.solver.metric()

    def terminateSolver(self):
        'Terminate the solver thread running "self.advance()"'
        if self.__dict__.has_key('solverThread'):
            self.isKeepRunning = False
            self.solverThread.join()
        
    def startSolver(self):
        'Start the solver thread running "self.advance()"'
        if self.__dict__.has_key('isKeepRunning'):
            assert self.isKeepRunning == False
        self.solverThread = threading.Thread(target=self.advance)
        self.solverThread.daemon = True
        self.isKeepRunning = True
        self.solverThread.start()
        
    def update(self):
        'Called automatically every second to update screen'
        self.solnLock.acquire()
        t, soln = self._soln_t, self._soln_copy.copy()
        self.solnLock.release()
        field = physical(soln, self.combobox.get_active_text())
        self.vis.update('MIT xWind  t ={0:10.6f}'.format(t), field)
        return True

    def changeGeom(self):
        'Change in geometry, new solver, new mesh, starting from freestream'
        self.terminateSolver()
        v, t, b = initMesh(self.geom, self.nE)
        self.vis = Visualize(v, t, extractEdges(v, t), self.fig, self.win)
        self.solver = chooseSolver(v, t, b, self.Mach, self.Re)
        self.solver.integrate(1E-9, self.solver.freeStream())
        self._soln_copy = self.solver.soln.copy()
        self.startSolver()

    def changeWind(self, *args):
        '''Change in Mach, Re or nE (# element) by changing the solver
           and interpolating old solution'''
        for b in self.buttons.values():
            b.set_sensitive(False)
        self.terminateSolver()

        m = self.solver.mesh
        xt0, W0 = m.xt(), self.solver.soln
        freeStream0 = self.solver.freeStream(1)
        axesLimit = ravel(array(fig.axes[0].viewLim).T)

        v, t, b = adaptMesh(self.geom, m.v, m.t, m.b, self.nE, self.metric)
        e = extractEdges(v, t)
        self.vis = Visualize(v, t, e, self.fig, self.win, axesLimit)
        self.solver = chooseSolver(v, t, b, self.Mach, self.Re)
        W0 = griddata(xt0, W0, self.solver.mesh.xt(), method='nearest')
        W0 += self.solver.freeStream(1) - freeStream0
        self.solver.integrate(1E-9, W0)
        self._soln_copy = self.solver.soln.copy()

        self.startSolver()
        self.buttons['MachDisp'].set_label('Mach = {0}'.format(self.Mach))
        self.buttons['ReDisp'].set_label('Re = {0}'.format(self.Re))
        self.buttons['NEDisp'].set_label('{0} cells'.format(self.nE))
        for b in self.buttons.values():
            b.set_sensitive(True)

    # ------- callback for change in Mach, Re and nE ------- #
    def MachUp(self, *args):
        if self.Mach < 8:
            self.Mach += 0.1
            self.changeWind()

    def MachDown(self, *args):
        if self.Mach > 0.2:
            self.Mach -= 0.1
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
nav.set_size_request(200, 35);

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
xwind = SolverGuiCoupler(win, fig, buttons, combobox)
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

# ================ Run program ================= #
win.show_all()
gtk.main()

