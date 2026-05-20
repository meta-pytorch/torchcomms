/**
 * @file    papi_cupti_common.h
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.)
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 */

#ifndef __PAPI_CUPTI_COMMON_H__
#define __PAPI_CUPTI_COMMON_H__

#include <stdio.h>
#include <cuda.h>
#include <cupti.h>

#include "cupti_utils.h"
#include "lcuda_debug.h"

typedef struct cuptic_info *cuptic_info_t;

extern const char *linked_cudart_path;
extern void *dl_cupti;

extern unsigned int _cuda_lock;

/* cuda driver function pointers */
extern CUresult ( *cuCtxGetCurrentPtr ) (CUcontext *);
extern CUresult ( *cuCtxSetCurrentPtr ) (CUcontext);
extern CUresult ( *cuCtxDestroyPtr ) (CUcontext);
extern CUresult ( *cuCtxCreatePtr ) (CUcontext *pctx, unsigned int flags, CUdevice dev);
extern CUresult ( *cuCtxGetDevicePtr ) (CUdevice *);
extern CUresult ( *cuDeviceGetPtr ) (CUdevice *, int);
extern CUresult ( *cuDeviceGetCountPtr ) (int *);
extern CUresult ( *cuDeviceGetNamePtr ) (char *, int, CUdevice);
extern CUresult ( *cuDevicePrimaryCtxRetainPtr ) (CUcontext *pctx, CUdevice);
extern CUresult ( *cuDevicePrimaryCtxReleasePtr ) (CUdevice);
extern CUresult ( *cuInitPtr ) (unsigned int);
extern CUresult ( *cuGetErrorStringPtr ) (CUresult error, const char** pStr);
extern CUresult ( *cuCtxPopCurrentPtr ) (CUcontext * pctx);
extern CUresult ( *cuCtxPushCurrentPtr ) (CUcontext pctx);
extern CUresult ( *cuCtxSynchronizePtr ) ();
extern CUresult ( *cuDeviceGetAttributePtr ) (int *, CUdevice_attribute, CUdevice);

/* cuda runtime function pointers */
extern cudaError_t ( *cudaGetDeviceCountPtr ) (int *);
extern cudaError_t ( *cudaGetDevicePtr ) (int *);
extern cudaError_t ( *cudaSetDevicePtr ) (int);
extern cudaError_t ( *cudaGetDevicePropertiesPtr ) (struct cudaDeviceProp* prop, int  device);
extern cudaError_t ( *cudaDeviceGetAttributePtr ) (int *value, enum cudaDeviceAttr attr, int device);
extern cudaError_t ( *cudaFreePtr ) (void *);
extern cudaError_t ( *cudaDriverGetVersionPtr ) (int *);
extern cudaError_t ( *cudaRuntimeGetVersionPtr ) (int *);

/* cupti function pointer */
extern CUptiResult ( *cuptiGetVersionPtr ) (uint32_t* );

/* utility functions to check runtime api, disabled reason, etc. */
int cuptic_init(void);
int cuptic_is_runtime_perfworks_api(void);
int cuptic_is_runtime_events_api(void);
int cuptic_device_get_count(int *num_gpus);
void cuptic_disabled_reason_set(const char *msg);
void cuptic_disabled_reason_get(const char **pmsg);
void *cuptic_load_dynamic_syms(const char *parent_path, const char *dlname, const char *search_subpaths[]);
int cuptic_shutdown(void);

/* context management interfaces */
int cuptic_ctxarr_create(cuptic_info_t *pinfo);
int cuptic_ctxarr_update_current(cuptic_info_t info, int evt_dev_id);
int cuptic_ctxarr_get_ctx(cuptic_info_t info, int gpu_idx, CUcontext *ctx);
int cuptic_ctxarr_destroy(cuptic_info_t *pinfo);

/* functions to track the occupancy of gpu counters in event sets */
int cuptic_device_acquire(cuptiu_event_table_t *evt_table);
int cuptic_device_release(cuptiu_event_table_t *evt_table);

/* device qualifier interfaces */
int cuptiu_dev_set(cuptiu_bitmap_t *bitmap, int i);
int cuptiu_dev_check(cuptiu_bitmap_t bitmap, int i);

#define DLSYM_AND_CHECK( dllib, name ) dlsym( dllib, name );  \
    if (dlerror() != NULL) {  \
        ERRDBG("A CUDA required function '%s' was not found in lib '%s'.\n", name, #dllib);  \
        return PAPI_EMISC;  \
    }

/* error handling defines for Cuda related function calls */
#define cudaCheckErrors( call, handleerror )  \
    do {  \
        CUresult _status = (call);  \
        LOGCUDACALL("\t" #call "\n");  \
        if (_status != CUDA_SUCCESS) {  \
            ERRDBG("CUDA Error %d: Error in call to " #call "\n", _status);  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);

#define cudaArtCheckErrors( call, handleerror )  \
    do {  \
        cudaError_t _status = (call);  \
        LOGCUDACALL("\t" #call "\n");  \
        if (_status != cudaSuccess) {  \
            ERRDBG("CUDART Error %d: Error in call to " #call "\n", _status);  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);

#define cuptiCheckErrors( call, handleerror ) \
    do {  \
        CUptiResult _status = (call);  \
        LOGCUPTICALL("\t" #call "\n");  \
        if (_status != CUPTI_SUCCESS) {  \
            ERRDBG("CUPTI Error %d: Error in call to " #call "\n", _status);  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);

#define nvpwCheckErrors( call, handleerror ) \
    do {  \
        NVPA_Status _status = (call);  \
        LOGCUPTICALL("\t" #call "\n");  \
        if (_status != NVPA_STATUS_SUCCESS) {  \
            ERRDBG("NVPA Error %d: Error in call to " #call "\n", _status);  \
            EXIT_OR_NOT; \
            handleerror;  \
        }  \
    } while (0);

#endif /* __CUPTI_COMMON_H__ */
