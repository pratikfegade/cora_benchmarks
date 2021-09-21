taco_csr_trmm: taco_csr_trmm.cpp kernels.cu
	nvcc -L${TACO_HOME}/build/lib/ -I${TACO_HOME}/include -o taco_csr_trmm taco_csr_trmm.cpp kernels.cu -ltaco

clean:
	rm taco_csr_trmm
