CUDAFLAGS = -lcudart -lcublas -lcusolver
GSLFLAGS = -lgsl -lgslcblas

all: test

lm: lm.cpp
	g++ lm.cpp -o lm $(shell gsl-config --libs)

test: cuLUsolve.h cuLUsolve.cpp test.cpp
	nvcc $^ -o $@ $(CUDAFLAGS) $(GSLFLAGS)
