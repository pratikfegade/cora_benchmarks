tensor_times_vector: tensor_times_vector.cpp kernels.cu
	nvcc -L${TACO_HOME}/build/lib/ -I${TACO_HOME}/include -o tensor_times_vector tensor_times_vector.cpp kernels.cu -ltaco

clean:
	rm tensor_times_vector
