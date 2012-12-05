from numpy import *
import scipy.sparse
from scipy.sparse import linalg as splinalg

#from IPython import embed

class Ode:
    def __init__(self, ddt, J, tol=1E-5, dt0=1E-3, dtMax=0.1 ):
        self.ddt = ddt
        self.J = J
        self.tol = tol
        self.dt = dt0
        self.dtMax = dtMax

    #@profile
    def integrate(self, t, y0=None, t0=None):
        if y0 is not None:
            # set initial condition
            y0 = ravel(y0)
            self.y, self.y0, self.y00 = y0.copy(), y0.copy(), y0.copy()
            self.dt0 = 0
            self.I = scipy.sparse.eye(self.y.size, self.y.size)
            if t0 is None: t0 = 0.
            self.t = t0

        # time advance to t
        while self.t < t:
            t0 = self.t
            self.t = min(t, self.t + self.dt)
            dt, dt0 = self.t - t0, self.dt0
            print self.t, t0, self.dt
            # Coefficients, y = b0 * y0 + b00 * y00 + a * f(y)
            # BDF2 if dt == dt0, Backward Euler if dt0 == 0
            b00 = -1 / max(3., dt0 * (dt0 + 2 * dt) / dt**2)
            a, b0 = dt + b00 * dt0, 1 - b00
            # newton iteration
            nIterMin, nIterMax = 5, 12
            for i in range(nIterMax+1):
                dydt = self.ddt(self.y)


                if not isfinite(dydt).all():
                    nIter = nIterMax + 1
                    break
                res = self.y - b0 * self.y0 - b00 * self.y00 - a * dydt
                resNorm = sqrt( (res**2).sum() / (self.y**2).sum() )
                print 'iter {0} with dt={1}, res={2}'.format(i, dt, resNorm)
                if resNorm < 1E-9 or resNorm < self.tol or i >= nIterMax:
                    nIter = i + 1
                    break

                J = self.J(self.y)

                tmp = self.I - a*J;
                dy, err = splinalg.gmres( tmp, res, tol=self.tol)
                self.y -= dy

            if nIter > nIterMax:
                self.t = t0
                self.y[:] = self.y0[:]
                self.dt *= 0.8
                print 'bad, dt = ', self.dt
            else:
                self.dt0 = dt
                self.y00[:] = self.y0
                self.y0[:] = self.y
                if nIter < nIterMin:
                    self.dt = min(self.dtMax, max(dt, self.dt / 0.8))
        return self.y

