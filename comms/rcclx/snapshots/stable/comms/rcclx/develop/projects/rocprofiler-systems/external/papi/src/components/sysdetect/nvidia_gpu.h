#ifndef __NVIDIA_GPU_H__
#define __NVIDIA_GPU_H__

#if CUDA_VERSION >= 11000
    #define PAPI_NVML_DEV_BUFFER_SIZE NVML_DEVICE_UUID_V2_BUFFER_SIZE
#else
    #define PAPI_NVML_DEV_BUFFER_SIZE NVML_DEVICE_UUID_BUFFER_SIZE
#endif

void open_nvidia_gpu_dev_type( _sysdetect_dev_type_info_t *dev_type_info );
void close_nvidia_gpu_dev_type( _sysdetect_dev_type_info_t *dev_type_info );

#endif /* End of __NVIDIA_GPU_H__ */
