ANI2DLIB_DIR = ../ani2d/lib
ANI2DLIB = -laft2D-3.0 -lmba2D-3.0

LIB = $(ANI2DLIB) -llapack -lm


default:	aftMesh mbaMesh

aftMesh:	aftMesh.o 
	gfortran -o $@ $< -L$(ANI2DLIB_DIR) $(LIB)

aftMesh.o:	aftMesh.f
	gfortran -c $<

mbaMesh:	mbaMesh.o 
	gfortran -o $@ $< -L$(ANI2DLIB_DIR) $(LIB)

mbaMesh.o:	mbaMesh.f
	gfortran -c $<

