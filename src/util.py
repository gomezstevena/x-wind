from numpy import dot, einsum, isnan, all

from scipy.interpolate import griddata


def dot_all(a,b):
    return dot(a.ravel(), b.ravel() )

def dot_each(a, b):
	return einsum('ni, ni -> n', a, b)

def nnorm(a):
    return dot(a.ravel(), a.ravel())

def linNearGrid( x0, W0, xn ):
	W = griddata(x0, W0, xn, method='linear')
	i_nan = all( isnan(W), axis=1 )

	print i_nan, i_nan.sum()

	W[i_nan] = griddata(x0, W0, xn[i_nan], method='nearest' )

	assert all(  ~isnan(W) )

	return W