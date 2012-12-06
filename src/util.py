from numpy import dot, einsum


def dot_all(a,b):
    return dot(a.ravel(), b.ravel() )

def dot_each(a, b):
	return einsum('ni, ni -> n', a, b)

def nnorm(a):
    return dot(a.ravel(), a.ravel())
