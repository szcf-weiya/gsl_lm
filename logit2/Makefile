train: lr.h lr.cpp train.cpp
	g++ lr.cpp train.cpp -o $@

predict: lr.h lr.cpp predict.cpp
	g++ lr.cpp predict.cpp -o $@

run:
	./train -n 50 -m 1e-06 -a 1 data/train.samp data/ldf.mod
