ANI2DLIB_DIR = ../ani2d/lib
ANI2DLIB = -laft2D-3.0 -lmba2D-3.0

LIB = $(ANI2DLIB) -llapack -lm

FC=gfortran

OUT=aftMesh mbaMesh

default:	aftMesh mbaMesh

aftMesh:	aftMesh.o 
	$(FC) -o $@ $< -L$(ANI2DLIB_DIR) $(LIB)

aftMesh.o:	aftMesh.f
	$(FC) -c $<

mbaMesh:	mbaMesh.o 
	$(FC) -o $@ $< -L$(ANI2DLIB_DIR) $(LIB)

mbaMesh.o:	mbaMesh.f
	$(FC) -c $<

clean:
	rm *.o $(OUT)