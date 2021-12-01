from ctypes import c_void_p, c_int, c_size_t, byref, CDLL
mlperf_lib = CDLL('/workspace/image_classification/cuda_graphs/mlperf_lib.so')

def mlperf_lars_multi_mp_sgd_mom_update(weights, num_weights=1):
    weights_ptr = c_void_p()
    mlperf_lib.mlperf_lars_multi_mp_sgd_mom_update(weights_ptr, c_int(num_weights))
