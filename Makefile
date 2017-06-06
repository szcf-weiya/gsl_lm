lm: lm.cpp
	g++ lm.cpp -o lm $(gsl-config --libs)
