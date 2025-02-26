all:
	nvcc main.cc conv2d.cc simulator.cc matmul.cc utils.cc -lcurand -lcublas -lcudnn -O3 -o main -std=c++11
clean:
	rm main
rocm:
	hipcc main.cc simulator.cc matmul.hip utils.cc -lrocblas -O3 -o main -std=c++11 -DWITH_ROCM -DROCM_USE_FLOAT16