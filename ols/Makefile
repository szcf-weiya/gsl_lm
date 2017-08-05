CUDAFLAGS = -lcudart -lcublas -lcusolver
GSLFLAGS = -lgsl -lgslcblas

all: test

lm: lm.cpp
	g++ lm.cpp -o lm $(shell gsl-config --libs)

test: cuLUsolve.h cuLUsolve.cu test.cpp
	nvcc cuLUsolve.cu test.cpp -o $@ $(CUDAFLAGS) $(GSLFLAGS)
