#include "stdio.h"
typedef unsigned int short uint16_t;
__global__ void mlperf_lars_multi_mp_sgd_mom_update_kernel(/*uint16_t* weights, float* grads, float* mom, float* master_weights*/) {
	printf("tid = %d\n",threadIdx.x);
};
