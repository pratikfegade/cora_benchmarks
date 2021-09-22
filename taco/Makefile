FLAGS=-O3 --use_fast_math
# FLAGS=-lineinfo

taco_csr_trmm: taco_csr_trmm.cu utils.cuh
	nvcc ${FLAGS} -L${TACO_HOME}/build/lib/ -I${TACO_HOME}/include -o taco_csr_trmm taco_csr_trmm.cu -ltaco

taco_csr_tradd: taco_csr_tradd.cu utils.cuh
	nvcc ${FLAGS} -L${TACO_HOME}/build/lib/ -I${TACO_HOME}/include -o taco_csr_tradd taco_csr_tradd.cu -ltaco


clean:
	rm -f taco_csr_trmm taco_csr_tradd
