#include"mlperf_lars-inl.h"
extern "C"
void mlperf_lars_multi_mp_sgd_mom_update(void* weights, int num_weights){
	mlperf_lars_multi_mp_sgd_mom_update_kernel<<<num_weights,512>>>(/*(uint16_t*)weights*/);
}
