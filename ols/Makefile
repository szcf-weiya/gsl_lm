CUDAFLAGS = -lcudart -lcublas -lcusolver
GSLFLAGS = -lgsl -lgslcblas

all: ols

lm: lm.cpp
	g++ lm.cpp -o lm $(shell gsl-config --libs)

test: cuLUsolve.h cuLUsolve.cu test.cpp
	nvcc cuLUsolve.cu test.cpp -o $@ $(CUDAFLAGS) $(GSLFLAGS)

cuLUsolve.o: cuLUsolve.h cuLUsolve.cu
	nvcc -c cuLUsolve.cu

cuMultifit.o: cuLUsolve.h cuMultifit.h cuMultifit.cu
	nvcc -c cuMultifit.cu

test.cpp: cuMultifit.h
	nvcc -c test.o

ols: cuLUsolve.o cuMultifit.o test.o
	nvcc $^ -o $@ $(CUDAFLAGS) $(GSLFLAGS)
