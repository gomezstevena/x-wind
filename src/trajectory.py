# import matplotlib
# matplotlib.use('Agg')
from numpy import *
from scipy.interpolate import PiecewisePolynomial

class Trajectory:
    def __init__(self, t, u, ddt=None):
        assert t.ndim == 1 and t.size == u.shape[0]
        self.shape = u.shape[1:]
        data = u.reshape([t.size, 1, -1])
        if ddt is not None:
            dudt = zeros(u.shape)
            for i in range(t.size):
                dudt[i] = ddt(u[i])
            data = hstack([data, dudt.reshape([t.size, 1, -1])])
        self.history = PiecewisePolynomial(t, data)

    def __call__(self, t):
        shape = array(t).shape
        u = self.history(t)
        return u.reshape(shape + self.shape)

    @property
    def tlim(self):
        t = self.history.xi
        return array([min(t), max(t)])


if __name__ == '__main__':
    Mach = 0.3
    Re = 10000
    HiRes = 1.

    from navierstokes import *
    soln = []
    for i in range(1000):
        fname = 'data/navierstokesStep{0:06d}.npz'.format(i)
        if os.path.exists(fname):
            z = load(fname)
            soln.append(z['soln'])
        else: break
    geom, v, t, b = z['geom'], z['v'], z['t'], z['b']
    soln = array(soln).squeeze()
    print 'loading complete'
    solver = NavierStokes(v, t, b, Mach, Re, HiRes)
    traj0 = Trajectory(arange(soln.shape[0]), soln)
    # traj1 = Trajectory(arange(soln.shape[0]), soln, ddt=solver.ddt)

    t = linspace(traj0.tlim[0], traj0.tlim[1] * 0.1, 200)
    it = solver.mesh.a.argsort()[:1]
    plot(t, (traj0(t)[:,it,:] / solver.Wref).reshape([t.size, -1]))
    # plot(t, (traj1(t)[:,it,:] / solver.Wref).reshape([t.size, -1]))

    '''
    for i in range(soln.shape[0]):
        clf()
        solver.mesh.plotTriScalar(soln[i,:,2])
        plot(solver.mesh.xt()[it,0], solver.mesh.xt()[it,1], 'ok')
        # solver.mesh.plotMesh(alpha=0.2)
        axis([-2,4,-2.5,2.5])
        # axis([-.2,.3,-.15,.15])
        savefig('fig/trajectory_{0:06d}.png'.format(i))

    '''
