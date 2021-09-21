#ifndef TACO_C_HEADERS
#define TACO_C_HEADERS
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <thrust/complex.h>
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
#define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
#define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)
#ifndef TACO_TENSOR_T_DEFINED
typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;

typedef struct {
  int32_t      order;         // tensor order (number of modes)
  int32_t*     dimensions;    // tensor dimensions
  int32_t      csize;         // component size
  int32_t*     mode_ordering; // mode storage ordering
  taco_mode_t* mode_types;    // mode storage types
  uint8_t***   indices;       // tensor index data (per mode)
  uint8_t*     vals;          // tensor values
  int32_t      vals_size;     // values array size
} taco_tensor_t;
#define TACO_TENSOR_T_DEFINED
#endif
#endif

float compute(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B, int32_t m, int32_t mode, int32_t iters);
