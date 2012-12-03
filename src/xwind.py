import sys
import gtk
import time
import gobject
gobject.threads_init()

from matplotlib.figure import Figure
from pylab import *
from numpy import *

from navierstokes import *
from meshVisualize import Visualize
from physical import physical

def chooseSolver(v, t, b, Mach, Re):
    if isfinite(Re):
        return NavierStokes(v, t, b, Mach, Re, HiRes=1)
    else:
        return Euler(v, t, b, Mach, HiRes=1)


class XWind:
    def __init__(self, win, fig, buttons, combobox):
        self.win = win
        self.fig = fig

        self.buttons = buttons
        self.combobox = combobox

        self.Mach = 0.5
        self.Re = 12800
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
        c, d = centerDiameter(self.geom)
        self.dt = 0.02 * d / 300.
        nV = self.solver.mesh.v.shape[0]
        self.metric = zeros([nV, 2, 2]) # metric for adaptation
        while self.isKeepRunning:
            self.solver.integrate(self.solver.time + self.dt)
            # exponential moving average of metric
            self.metric = 0.99 * self.metric + 0.01 * self.solver.metric()

    def stop(self):
        if self.__dict__.has_key('solverThread'):
            self.isKeepRunning = False
            self.solverThread.join()
        
    def start(self):
        if self.__dict__.has_key('isKeepRunning'):
            assert self.isKeepRunning == False
        self.solverThread = threading.Thread(target=self.advance)
        self.solverThread.daemon = True
        self.isKeepRunning = True
        self.solverThread.start()
        
    def changeGeom(self):
        self.stop()
        v, t, b = initMesh(self.geom, self.nE)
        self.vis = Visualize(v, t, extractEdges(v, t), self.fig, self.win)
        self.solver = chooseSolver(v, t, b, self.Mach, self.Re)
        self.solver.integrate(1E-9, self.solver.freeStream())
        self.start()

    def changeWind(self, *args):
        for b in self.buttons.values():
            b.set_sensitive(False)
        self.stop()

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

        self.start()
        self.buttons['MachDisp'].set_label('Mach = {0}'.format(self.Mach))
        self.buttons['ReDisp'].set_label('Re = {0}'.format(self.Re))
        self.buttons['NEDisp'].set_label('{0} cells'.format(self.nE))
        for b in self.buttons.values():
            b.set_sensitive(True)

    def update(self):
        t, soln = self.solver.getSolnCache()
        field = physical(soln, self.combobox.get_active_text())
        self.vis.update('MIT xWind  t ={0:10.6f}'.format(t), field)
        return True

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


from matplotlib.backends.backend_gtkagg \
     import FigureCanvasGTKAgg as FigureCanvas
from matplotlib.backends.backend_gtkagg \
     import NavigationToolbar2GTKAgg as NavigationToolbar

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
buttons['ReMesh'] = gtk.Button("ReMesh")
buttons['MachDown'] = gtk.ToolButton(gtk.STOCK_GO_DOWN)
buttons['MachDisp'] = gtk.Button("")
buttons['MachUp'] = gtk.ToolButton(gtk.STOCK_GO_UP)
buttons['ReDown'] = gtk.ToolButton(gtk.STOCK_GO_DOWN)
buttons['ReDisp'] = gtk.Button("")
buttons['ReUp'] = gtk.ToolButton(gtk.STOCK_GO_UP)
buttons['NEDown'] = gtk.ToolButton(gtk.STOCK_GO_DOWN)
buttons['NEDisp'] = gtk.Button("")
buttons['NEUp'] = gtk.ToolButton(gtk.STOCK_GO_UP)

fig = Figure()
canvas = FigureCanvas(fig)  # a gtk.DrawingArea
nav = NavigationToolbar(canvas, win)
nav.set_size_request(200, 35);

sep = [gtk.SeparatorToolItem() for i in range(5)]
toolbar = gtk.HBox(False, 2)
toolbar.pack_start(nav, False, False, 0)
toolbar.pack_start(sep[0], False, False, 0)
toolbar.pack_start(combobox, False, False, 0)
toolbar.pack_start(sep[1], False, False, 0)
toolbar.pack_start(buttons['ReMesh'], False, False, 0)
toolbar.pack_start(sep[2], False, False, 0)
toolbar.pack_start(buttons['MachDown'], False, False, 0)
toolbar.pack_start(buttons['MachDisp'], False, False, 0)
toolbar.pack_start(buttons['MachUp'], False, False, 0)
toolbar.pack_start(sep[3], False, False, 0)
toolbar.pack_start(buttons['ReDown'], False, False, 0)
toolbar.pack_start(buttons['ReDisp'], False, False, 0)
toolbar.pack_start(buttons['ReUp'], False, False, 0)
toolbar.pack_start(sep[4], False, False, 0)
toolbar.pack_start(buttons['NEDown'], False, False, 0)
toolbar.pack_start(buttons['NEDisp'], False, False, 0)
toolbar.pack_start(buttons['NEUp'], False, False, 0)

vbox = gtk.VBox(False, 3)
vbox.pack_start(toolbar, False, False, 0)
vbox.add(canvas)

# =========== create XWind object ============== #
xwind = XWind(win, fig, buttons, combobox)
buttons['ReMesh'].connect("clicked", xwind.changeWind)
buttons['MachUp'].connect("clicked", xwind.MachUp)
buttons['MachDown'].connect("clicked", xwind.MachDown)
buttons['ReUp'].connect("clicked", xwind.ReUp)
buttons['ReDown'].connect("clicked", xwind.ReDown)
buttons['NEUp'].connect("clicked", xwind.nEUp)
buttons['NEDown'].connect("clicked", xwind.nEDown)

win.add(vbox)

def destroyFunc(*x):
    xwind.stop()
    gtk.main_quit()

win.connect("destroy", destroyFunc)

win.show_all()
gtk.main()

