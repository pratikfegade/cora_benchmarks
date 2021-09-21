tensor_times_vector: tensor_times_vector.cpp kernels.cu
	nvcc -g -O0 -L../../build/lib/ -I../../include -o tensor_times_vector tensor_times_vector.cpp kernels.cu -ltaco

clean:
	rm tensor_times_vector
