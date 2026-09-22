/**
 * @file    papi_cupti_common.c
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.)
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 */

#include <dlfcn.h>
#include <link.h>
#include <libgen.h>
#include <dirent.h>
#include <sys/stat.h>
#include <papi.h>
#include <cupti_target.h>
#include "papi_memory.h"

#include "cupti_config.h"
#include "papi_cupti_common.h"

static void *dl_drv, *dl_rt;

static char cuda_error_string[PAPI_HUGE_STR_LEN];

void *dl_cupti;

unsigned int _cuda_lock;

typedef int64_t gpu_occupancy_t;
static gpu_occupancy_t global_gpu_bitmask;

// Variables to handle partially disabled Cuda component
static int isCudaPartial = 0;
static int enabledDeviceIds[PAPI_CUDA_MAX_DEVICES];
static size_t enabledDevicesCnt = 0;

typedef enum
{
    sys_gpu_ccs_unknown = 0,
    sys_gpu_ccs_mixed,
    sys_gpu_ccs_all_lt_70,
    sys_gpu_ccs_all_eq_70,
    sys_gpu_ccs_all_gt_70,
    sys_gpu_ccs_all_lte_70,
    sys_gpu_ccs_all_gte_70
} sys_compute_capabilities_e;

// Load necessary functions from Cuda toolkit e.g. cupti or runtime 
static int util_load_cuda_sym(void);
static int load_cuda_sym(void);
static int load_cudart_sym(void);
static int load_cupti_common_sym(void);

// Unload the loaded functions from Cuda toolkit e.g. cupti or runtime
static int unload_cudart_sym(void);
static int unload_cupti_common_sym(void);
static void unload_linked_cudart_path(void);

// Functions to get library versions 
static int util_dylib_cu_runtime_version(void);
static int util_dylib_cupti_version(void);

// Functions to get cuda runtime library path
static int dl_iterate_phdr_cb(struct dl_phdr_info *info, __attribute__((unused)) size_t size, __attribute__((unused)) void *data);
static int get_user_cudart_path(void);

// Function to determine compute capabilities
static int compute_capabilities_on_system(sys_compute_capabilities_e *system_ccs);

// Functions to handle a partially disabled Cuda component
static int get_enabled_devices(void); 

// misc.
static int _devmask_events_get(cuptiu_event_table_t *evt_table, gpu_occupancy_t *bitmask);
static int verify_cuda_toolkit_supports_architectures_on_machine(void);
static const char *find_path_to_nvcc(char *path_from_env);

/* cuda driver function pointers */
CUresult ( *cuCtxGetCurrentPtr ) (CUcontext *);
CUresult ( *cuCtxSetCurrentPtr ) (CUcontext);
CUresult ( *cuCtxDestroyPtr ) (CUcontext);
CUresult ( *cuCtxGetDevicePtr ) (CUdevice *);
CUresult ( *cuDeviceGetPtr ) (CUdevice *, int);
CUresult ( *cuDeviceGetCountPtr ) (int *);
CUresult ( *cuDeviceGetNamePtr ) (char *, int, CUdevice);
CUresult ( *cuDevicePrimaryCtxRetainPtr ) (CUcontext *pctx, CUdevice);
CUresult ( *cuDevicePrimaryCtxReleasePtr ) (CUdevice);
CUresult ( *cuInitPtr ) (unsigned int);
CUresult ( *cuGetErrorStringPtr ) (CUresult error, const char** pStr);
CUresult ( *cuCtxPopCurrentPtr ) (CUcontext * pctx);
CUresult ( *cuCtxPushCurrentPtr ) (CUcontext pctx);
CUresult ( *cuCtxSynchronizePtr ) ();
CUresult ( *cuDeviceGetAttributePtr ) (int *, CUdevice_attribute, CUdevice);

/* cuda runtime function pointers */
cudaError_t ( *cudaGetDeviceCountPtr ) (int *);
cudaError_t ( *cudaGetDevicePtr ) (int *);
const char *( *cudaGetErrorStringPtr ) (cudaError_t);
cudaError_t ( *cudaSetDevicePtr ) (int);
cudaError_t ( *cudaGetDevicePropertiesPtr ) (struct cudaDeviceProp* prop, int  device);
cudaError_t ( *cudaDeviceGetAttributePtr ) (int *value, enum cudaDeviceAttr attr, int device);
cudaError_t ( *cudaFreePtr ) (void *);
cudaError_t ( *cudaDriverGetVersionPtr ) (int *);
cudaError_t ( *cudaRuntimeGetVersionPtr ) (int *);

/* cupti function pointer */
CUptiResult ( *cuptiGetVersionPtr ) (uint32_t* );
CUptiResult ( *cuptiDeviceGetChipNamePtr ) (CUpti_Device_GetChipName_Params* params);

/**@class load_cuda_sym
 * @brief Search for a variation of the shared object libcuda.
 */
int load_cuda_sym(void)
{
    int soNamesToSearchCount = 3;
    const char *soNamesToSearchFor[] = {"libcuda.so", "libcuda.so.1", "libcuda"};

    dl_drv = search_and_load_from_system_paths(soNamesToSearchFor, soNamesToSearchCount);
    if (!dl_drv) {
        ERRDBG("Loading installed libcuda.so failed. Check that cuda drivers are installed.\n");
        goto fn_fail;
    }

    cuCtxSetCurrentPtr           = DLSYM_AND_CHECK(dl_drv, "cuCtxSetCurrent");
    cuCtxGetCurrentPtr           = DLSYM_AND_CHECK(dl_drv, "cuCtxGetCurrent");
    cuCtxDestroyPtr              = DLSYM_AND_CHECK(dl_drv, "cuCtxDestroy");
    cuCtxGetDevicePtr            = DLSYM_AND_CHECK(dl_drv, "cuCtxGetDevice");
    cuDeviceGetPtr               = DLSYM_AND_CHECK(dl_drv, "cuDeviceGet");
    cuDeviceGetCountPtr          = DLSYM_AND_CHECK(dl_drv, "cuDeviceGetCount");
    cuDeviceGetNamePtr           = DLSYM_AND_CHECK(dl_drv, "cuDeviceGetName");
    cuDevicePrimaryCtxRetainPtr  = DLSYM_AND_CHECK(dl_drv, "cuDevicePrimaryCtxRetain");
    cuDevicePrimaryCtxReleasePtr = DLSYM_AND_CHECK(dl_drv, "cuDevicePrimaryCtxRelease");
    cuInitPtr                    = DLSYM_AND_CHECK(dl_drv, "cuInit");
    cuGetErrorStringPtr          = DLSYM_AND_CHECK(dl_drv, "cuGetErrorString");
    cuCtxPopCurrentPtr           = DLSYM_AND_CHECK(dl_drv, "cuCtxPopCurrent");
    cuCtxPushCurrentPtr          = DLSYM_AND_CHECK(dl_drv, "cuCtxPushCurrent");
    cuCtxSynchronizePtr          = DLSYM_AND_CHECK(dl_drv, "cuCtxSynchronize");
    cuDeviceGetAttributePtr      = DLSYM_AND_CHECK(dl_drv, "cuDeviceGetAttribute");

    Dl_info info;
    dladdr(cuCtxSetCurrentPtr, &info);
    LOGDBG("CUDA driver library loaded from %s\n", info.dli_fname);
    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

static int unload_cuda_sym(void)
{
    if (dl_drv) {
        dlclose(dl_drv);
        dl_drv = NULL;
    }
    cuCtxSetCurrentPtr           = NULL;
    cuCtxGetCurrentPtr           = NULL;
    cuCtxDestroyPtr              = NULL;
    cuCtxGetDevicePtr            = NULL;
    cuDeviceGetPtr               = NULL;
    cuDeviceGetCountPtr          = NULL;
    cuDeviceGetNamePtr           = NULL;
    cuDevicePrimaryCtxRetainPtr  = NULL;
    cuDevicePrimaryCtxReleasePtr = NULL;
    cuInitPtr                    = NULL;
    cuGetErrorStringPtr          = NULL;
    cuCtxPopCurrentPtr           = NULL;
    cuCtxPushCurrentPtr          = NULL;
    cuCtxSynchronizePtr          = NULL;
    cuDeviceGetAttributePtr      = NULL;
    return PAPI_OK;
}

/**@class search_and_load_shared_objects
 * @brief Search and load Cuda shared objects.
 *
 * @param *parentPath
 *   The main path we will use to search for the shared objects. 
 * @param *soMainName
 *   The name of the shared object e.g. libcudart. This is used
 *   to select the standardSubPaths to use.
 * @param *soNamesToSearchFor[]
 *   Varying names of the shared object we want to search for.
 * @param soNamesToSearchCount
 *   Total number of names in soNamesToSearchFor.
 */
void *search_and_load_shared_objects(const char *parentPath, const char *soMainName, const char *soNamesToSearchFor[], int soNamesToSearchCount)
{
    const char *standardSubPaths[3];
    // Case for when we want to search explicit subpaths for a shared object
    if (soMainName != NULL) {
        if (strcmp(soMainName, "libcudart") == 0) {
            standardSubPaths[0] = "%s/lib64/";
            standardSubPaths[1] = NULL;
        }
        else if (strcmp(soMainName, "libcupti") == 0) {
            standardSubPaths[0] = "%s/extras/CUPTI/lib64/";
            standardSubPaths[1] = "%s/lib64/";
            standardSubPaths[2] = NULL;
        }
        else if (strcmp(soMainName, "libnvperf_host") == 0) {
            standardSubPaths[0] = "%s/extras/CUPTI/lib64/";
            standardSubPaths[1] = "%s/lib64/";
            standardSubPaths[2] = NULL;
        }
    }
    // Case for when a user provides an exact path e.g. PAPI_CUDA_RUNTIME
    // and we do not want to search subpaths
    else{
        standardSubPaths[0] = "%s/";
        standardSubPaths[1] = NULL;     
    }

    char pathToSharedLibrary[PAPI_HUGE_STR_LEN], directoryPathToSearch[PAPI_HUGE_STR_LEN];
    void *so = NULL;
    char *soNameFound;
    int i, strLen;
    for (i = 0; standardSubPaths[i] != NULL; i++) {
        // Create path to search for dl names
        int strLen = snprintf(directoryPathToSearch, PAPI_HUGE_STR_LEN, standardSubPaths[i], parentPath);
        if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
            ERRDBG("Failed to fully write path to search for dlnames.\n");
            return NULL;
        }   

        DIR *dir = opendir(directoryPathToSearch);
        if (dir == NULL) {
            ERRDBG("Directory path could not be opened.\n");
            continue;
        }

        int j;
        for (j = 0; j < soNamesToSearchCount; j++) {
            struct dirent *dirEntry;
            while( ( dirEntry = readdir(dir) ) != NULL ) {
                int result;
                char *p = strstr(soNamesToSearchFor[j], "so");
                // Check for an exact match of a shared object name (.so and .so.1 case)
                if (p) {
                    result = strcmp(dirEntry->d_name, soNamesToSearchFor[j]);
                }
                // Check for any match of a shared object name (we could not find .so and .so.1)
                else {
                    result = strncmp(dirEntry->d_name, soNamesToSearchFor[j], strlen(soNamesToSearchFor[j]));
                }

                if (result == 0) {
                    soNameFound = dirEntry->d_name;
                    goto found;
                }
            }
            // Reset the position of the directory stream
            rewinddir(dir);
        }
    }

  exit:
    return so;
  found:
    // Construct path to shared library
    strLen = snprintf(pathToSharedLibrary, PAPI_HUGE_STR_LEN, "%s%s", directoryPathToSearch, soNameFound);
    if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
        ERRDBG("Failed to fully write constructed path to shared library.\n");
        return NULL;
    }
    so = dlopen(pathToSharedLibrary, RTLD_NOW | RTLD_GLOBAL);
   
    goto exit; 
}

/**@class search_and_load_from_system_paths
 * @brief A simple wrapper to try and search and load
 *        Cuda shared objects from system paths.
 *
 * @param *soNamesToSearchFor[]
 *   Varying names of the shared object we want to search for.
 * @param soNamesToSearchCount
 *   Total number of names in soNamesToSearchFor.
 */
void *search_and_load_from_system_paths(const char *soNamesToSearchFor[], int soNamesToSearchCount)
{
    void *so = NULL;
    int i;
    for (i = 0; i < soNamesToSearchCount; i++) {
        so = dlopen(soNamesToSearchFor[i], RTLD_NOW | RTLD_GLOBAL);
        if (so) {
            return so;
        }   
    }

    return so; 
}

/**@class load_cudart_sym
 * @brief Search for a variation of the shared object libcudart.
 *        Order of search is outlined below.
 *
 * 1. If a user sets PAPI_CUDA_RUNTIME, this will take precedent over
 *    the options listed below to be searched.
 * 2. If we fail to collect a variation of the shared object libcudart from PAPI_CUDA_RUNTIME or it is not set,
 *    we will search the path defined with PAPI_CUDA_ROOT; as this is supposed to always be set.
 * 3. If we fail to collect a variation of the shared object libcudart from steps 1 and 2, then we will search the linux
 *    default directories listed by /etc/ld.so.conf. As a note, updating the LD_LIBRARY_PATH is
 *    advised for this option.
 * 4. We use dlopen to search for a variation of the shared object libcudart.
 *    If this fails, then we failed to find a variation of the shared object
 *    libcudart.
 */
int load_cudart_sym(void)
{
    int soNamesToSearchCount = 3;
    const char *soNamesToSearchFor[] = {"libcudart.so", "libcudart.so.1", "libcudart"};

    // If a user set PAPI_CUDA_RUNTIME with a path, then search it for the shared object (takes precedent over PAPI_CUDA_ROOT)
    char *papi_cuda_runtime = getenv("PAPI_CUDA_RUNTIME");
    if (papi_cuda_runtime) {
        dl_rt = search_and_load_shared_objects(papi_cuda_runtime, NULL, soNamesToSearchFor, soNamesToSearchCount);
    }

    char *soMainName = "libcudart";
    // If a user set PAPI_CUDA_ROOT with a path and we did not already find the shared object, then search it for the shared object
    char *papi_cuda_root = getenv("PAPI_CUDA_ROOT");
    if (papi_cuda_root && !dl_rt) {
        dl_rt = search_and_load_shared_objects(papi_cuda_root, soMainName, soNamesToSearchFor, soNamesToSearchCount);
    }

    // Last ditch effort to find a variation of libcudart, see dlopen manpages for how search occurs
    if (!dl_rt) {
        dl_rt = search_and_load_from_system_paths(soNamesToSearchFor, soNamesToSearchCount);
        if (!dl_rt) {
            ERRDBG("Loading libcudart shared library failed. Try setting PAPI_CUDA_ROOT\n");
            goto fn_fail;
        }
    }

    cudaGetDevicePtr           = DLSYM_AND_CHECK(dl_rt, "cudaGetDevice");
    cudaGetDeviceCountPtr      = DLSYM_AND_CHECK(dl_rt, "cudaGetDeviceCount");
    cudaGetDevicePropertiesPtr = DLSYM_AND_CHECK(dl_rt, "cudaGetDeviceProperties");
    cudaGetErrorStringPtr      = DLSYM_AND_CHECK(dl_rt, "cudaGetErrorString");
    cudaDeviceGetAttributePtr  = DLSYM_AND_CHECK(dl_rt, "cudaDeviceGetAttribute");
    cudaSetDevicePtr           = DLSYM_AND_CHECK(dl_rt, "cudaSetDevice");
    cudaFreePtr                = DLSYM_AND_CHECK(dl_rt, "cudaFree");
    cudaDriverGetVersionPtr    = DLSYM_AND_CHECK(dl_rt, "cudaDriverGetVersion");
    cudaRuntimeGetVersionPtr   = DLSYM_AND_CHECK(dl_rt, "cudaRuntimeGetVersion");

    Dl_info info;
    dladdr(cudaGetDevicePtr, &info);
    LOGDBG("CUDA runtime library loaded from %s\n", info.dli_fname);
    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

int unload_cudart_sym(void)
{
    if (dl_rt) {
        dlclose(dl_rt);
        dl_rt = NULL;
    }
    cudaGetDevicePtr           = NULL;
    cudaGetDeviceCountPtr      = NULL;
    cudaGetDevicePropertiesPtr = NULL;
    cudaGetErrorStringPtr      = NULL;
    cudaDeviceGetAttributePtr  = NULL;
    cudaSetDevicePtr           = NULL;
    cudaFreePtr                = NULL;
    cudaDriverGetVersionPtr    = NULL;
    cudaRuntimeGetVersionPtr   = NULL;
    return PAPI_OK;
}

/**@class load_cupti_common_sym
 * @brief Search for a variation of the shared object libcupti.
 *        Order of search is outlined below.
 *
 * 1. If a user sets PAPI_CUDA_CUPTI, this will take precedent over
 *    the options listed below to be searched.
 * 2. If we fail to collect a variation of the shared object libcupti from PAPI_CUDA_CUPTI or it is not set,
 *    we will search the path defined with PAPI_CUDA_ROOT; as this is supposed to always be set.
 * 3. If we fail to collect a variation of the shared object libcupti from steps 1 and 2, then we will search the linux
 *    default directories listed by /etc/ld.so.conf. As a note, updating the LD_LIBRARY_PATH is
 *    advised for this option.
 * 4. We use dlopen to search for a variation of the shared object libcupti.
 *    If this fails, then we failed to find a variation of the shared object
 *    libcupti.
 */
int load_cupti_common_sym(void)
{
    int soNamesToSearchCount = 3;
    const char  *soNamesToSearchFor[] = {"libcupti.so", "libcupti.so.1", "libcupti"};

    // If a user set PAPI_CUDA_CUPTI with a path, then search it for the shared object (takes precedent over PAPI_CUDA_ROOT)
    char *papi_cuda_cupti = getenv("PAPI_CUDA_CUPTI");
    if (papi_cuda_cupti) {
        dl_cupti = search_and_load_shared_objects(papi_cuda_cupti, NULL, soNamesToSearchFor, soNamesToSearchCount);
    }

    char *soMainName = "libcupti";
    // If a user set PAPI_CUDA_ROOT with a path and we did not already find the shared object, then search it for the shared object
    char *papi_cuda_root = getenv("PAPI_CUDA_ROOT");
    if (papi_cuda_root && !dl_cupti) {
        dl_cupti = search_and_load_shared_objects(papi_cuda_root, soMainName, soNamesToSearchFor, soNamesToSearchCount);
    }

    // Last ditch effort to find a variation of libcupti, see dlopen manpages for how search occurs
    if (!dl_cupti) {
        dl_cupti = search_and_load_from_system_paths(soNamesToSearchFor, soNamesToSearchCount);
        if (!dl_cupti) {
            ERRDBG("Loading libcupti.so failed. Try setting PAPI_CUDA_ROOT\n");
            goto fn_fail;
        }
    }

    cuptiGetVersionPtr = DLSYM_AND_CHECK(dl_cupti, "cuptiGetVersion");
    cuptiDeviceGetChipNamePtr = DLSYM_AND_CHECK(dl_cupti, "cuptiDeviceGetChipName");

    Dl_info info;
    dladdr(cuptiGetVersionPtr, &info);
    LOGDBG("CUPTI library loaded from %s\n", info.dli_fname);
    return PAPI_OK;
fn_fail:
    return PAPI_EMISC;
}

int unload_cupti_common_sym(void)
{
    if (dl_cupti) {
        dlclose(dl_cupti);
        dl_cupti = NULL;
    }
    cuptiGetVersionPtr = NULL;
    cuptiDeviceGetChipNamePtr = NULL;
    return PAPI_OK;
}

int util_load_cuda_sym(void)
{
    int papi_errno;
    papi_errno = load_cuda_sym();
    papi_errno += load_cudart_sym();
    papi_errno += load_cupti_common_sym();
    if (papi_errno != PAPI_OK) {
        return PAPI_EMISC;
    }
    else
        return PAPI_OK;
}

int cuptic_shutdown(void)
{
    unload_cuda_sym();
    unload_cudart_sym();
    unload_cupti_common_sym();
    return PAPI_OK;
}

int util_dylib_cu_runtime_version(void)
{
    int runtimeVersion;
    cudaArtCheckErrors(cudaRuntimeGetVersionPtr(&runtimeVersion), return PAPI_EMISC);
    return runtimeVersion;
}

int util_dylib_cupti_version(void)
{
    unsigned int cuptiVersion;
    cuptiCheckErrors(cuptiGetVersionPtr(&cuptiVersion), return PAPI_EMISC);
    return cuptiVersion;
}


/** @class cuptic_device_get_count
  * @brief Get total number of gpus on the machine that are compute
  *        capable..
  * @param *num_gpus 
  *    Collect the total number of gpus.
*/
int cuptic_device_get_count(int *num_gpus)
{
    cudaError_t cuda_err;

    /* find the total number of compute-capable devices */
    cuda_err = cudaGetDeviceCountPtr(num_gpus);
    if (cuda_err != cudaSuccess) {
        cuptic_err_set_last(cudaGetErrorStringPtr(cuda_err));
        return PAPI_EMISC;
    }
    return PAPI_OK;
}

int get_gpu_compute_capability(int dev_num, int *cc)
{
    int cc_major, cc_minor;
    cudaError_t cuda_errno;
    cuda_errno = cudaDeviceGetAttributePtr(&cc_major, cudaDevAttrComputeCapabilityMajor, dev_num);
    if (cuda_errno != cudaSuccess) {
        cuptic_err_set_last(cudaGetErrorStringPtr(cuda_errno));
        return PAPI_EMISC;
    }
    cuda_errno = cudaDeviceGetAttributePtr(&cc_minor, cudaDevAttrComputeCapabilityMinor, dev_num);
    if (cuda_errno != cudaSuccess) {
        cuptic_err_set_last(cudaGetErrorStringPtr(cuda_errno));
        return PAPI_EMISC;
    }
    *cc = cc_major * 10 + cc_minor;
    return PAPI_OK;
}

int compute_capabilities_on_system(sys_compute_capabilities_e *system_ccs)
{
    int total_gpus;
    int papi_errno = cuptic_device_get_count(&total_gpus);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    int i, cc;
    int num_gpus_with_ccs_gt_cc70 = 0, num_gpus_with_ccs_eq_cc70 = 0, num_gpus_with_ccs_lt_cc70 = 0;
    for (i = 0; i < total_gpus; i++) {
        papi_errno = get_gpu_compute_capability(i, &cc);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        if (cc > 70) {
            ++num_gpus_with_ccs_gt_cc70;
        }
        if (cc == 70) {
            ++num_gpus_with_ccs_eq_cc70;
        }
        if (cc < 70) {
            ++num_gpus_with_ccs_lt_cc70;
        }
    }

    sys_compute_capabilities_e sys_ccs = sys_gpu_ccs_unknown;
    // All devices have CCs > 7.0.
    if (num_gpus_with_ccs_gt_cc70 == total_gpus) {
        sys_ccs = sys_gpu_ccs_all_gt_70;
    }
    // All devices have CCs = 7.0
    else if (num_gpus_with_ccs_eq_cc70 == total_gpus) {
        sys_ccs = sys_gpu_ccs_all_eq_70;
    }
    // All devices have CCs < 7.0
    else if (num_gpus_with_ccs_lt_cc70 == total_gpus) {
        sys_ccs = sys_gpu_ccs_all_lt_70;
    }
    // Devices can result in a partially disabled Cuda component
    else {
        sys_ccs = sys_gpu_ccs_mixed;

        int all_ccs_gte_cc70 = num_gpus_with_ccs_eq_cc70 + num_gpus_with_ccs_gt_cc70;
        if (all_ccs_gte_cc70 == total_gpus) {
            sys_ccs = sys_gpu_ccs_all_gte_70;
        }
 
        int all_ccs_lte_cc70 = num_gpus_with_ccs_eq_cc70 + num_gpus_with_ccs_lt_cc70;
        if (all_ccs_lte_cc70 == total_gpus) {
            sys_ccs = sys_gpu_ccs_all_lte_70;
        }
    }
    *system_ccs = sys_ccs;

    return PAPI_OK;
}

/** @class cuptic_err_set_last
  * @brief For the last error, set an error message.
  * @param *error_str
  *    Error message to be set.
*/
int cuptic_err_set_last(const char *error_str)
{
    int strLen = snprintf(cuda_error_string, PAPI_HUGE_STR_LEN, "%s", error_str);
    if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
        SUBDBG("Last set error message not fully written.\n");
    }
    
    return PAPI_OK;
}

/** @class cuptic_err_get_last
  * @brief Get the last error message set.
  * @param **error_str
  *    Error message to be returned.
*/
int cuptic_err_get_last(const char **error_str)
{
    *error_str = cuda_error_string;
    return PAPI_OK;
}

/** @class find_path_to_nvcc
  * @brief Attempt to reconstruct a path to bin/nvcc.
  * @param *path_from_env
  *   Path we want to use to help reconstruction to bin/nvcc.
  *
*/
const char *find_path_to_nvcc(char *path_from_env)
{
    if (path_from_env == NULL || strstr(path_from_env, "cuda") == NULL) {
        return NULL;
    }

    // Create a copy to work off of
    char path_from_env_copy[PATH_MAX] = { 0 };
    int strLen = snprintf(path_from_env_copy, sizeof(path_from_env_copy), "%s", path_from_env);
    if (strLen < 0 || (size_t) strLen >= sizeof(path_from_env_copy)) {
        SUBDBG("Failed to store %s in the buffer variable path_from_env_copy.\n", path_from_env);
        return NULL;
    }

    // If the path starts with a forward slash go ahead and capture it and move forward
    // in the string
    char basepath[PATH_MAX] = { 0 };
    if (path_from_env_copy[0] == '/') {
        strLen = snprintf(basepath, sizeof(basepath), "%c", path_from_env_copy[0]);
        if (strLen < 0 || (size_t) strLen >= sizeof(basepath)) {
            SUBDBG("Failed to store %c in the buffer variable path.\n", path_from_env_copy[0]);
            return NULL;
        }
        memmove(path_from_env_copy, path_from_env_copy + 1, strlen(path_from_env_copy) + 1);
    }

    char *reconstructed_path_to_nvcc = (char *) malloc(PATH_MAX * sizeof(char));
    if (reconstructed_path_to_nvcc == NULL) {
        SUBDBG("Failed to allocate memory for the variable reconstructed_path_to_nvcc.\n");
        return NULL;
    }

    // Begin to reconstruct paths such that we can find path/to/bin/nvcc
    char *cuda_directory_path = strtok(path_from_env_copy, "/");
    while(cuda_directory_path) {
        size_t current_length = strlen(basepath);
        size_t remaining_length = sizeof(basepath) - current_length;

        strLen = snprintf(basepath + current_length, remaining_length, "%s/", cuda_directory_path);
        if (strLen < 0 || (size_t) strLen >= remaining_length) {
            SUBDBG("Failed to store %s/ in the buffer variable basepath.\n", cuda_directory_path);
            free(reconstructed_path_to_nvcc);
            return NULL;
        }

        strLen = snprintf(reconstructed_path_to_nvcc, PATH_MAX, "%sbin/nvcc", basepath);
        if (strLen < 0 || strLen >= PATH_MAX) {
            SUBDBG("Failed to store %s/bin/nvcc in buffer variable reconstructed_path_to_nvcc.\n", basepath);
            free(reconstructed_path_to_nvcc);
            return NULL;
        }

        // Check current reconstructed path
        struct stat buf;
        int status = stat(reconstructed_path_to_nvcc, &buf);
        if (status == 0) {
            return reconstructed_path_to_nvcc;
        }
        cuda_directory_path = strtok(NULL, "/");
    }
    free(reconstructed_path_to_nvcc);

    return NULL;
}

/** @class verify_cuda_toolkit_supports_architecture_on_machine
  * @brief Cuda Toolkits have specific architectures that they support.
  *        This function gets the list of supported architectures for a
  *        Cuda Toolkit and compares that list to the devices on the machine.
  *
  *        To get the list of supported architectures we must have a path to
  *        bin/nvcc. Due to this, we reconstruct paths via the below environment variables
  *        in order of how they appear until we recreate an existing path to bin/nvcc.
  *
  *        1. PAPI_CUDA_RUNTIME
  *        2. PAPI_CUDA_CUPTI
  *        3. PAPI_CUDA_PERFWORKS
  *        4. PAPI_CUDA_ROOT
  *        5. LD_LIBRARY_PATH
  *
  *        We prioritize PAPI_CUDA_RUNTIME, PAPI_CUDA_CUPTI,and PAPI_CUDA_PERFWORKS
  *        as they are prioritized for their respective library checks.
*/
int verify_cuda_toolkit_supports_architectures_on_machine(void)
{
    const char *path_to_nvcc = NULL;
    char *papi_cuda_runtime = getenv("PAPI_CUDA_RUNTIME");
    if (papi_cuda_runtime != NULL) {
        path_to_nvcc = find_path_to_nvcc(papi_cuda_runtime); 
    }

    char *papi_cuda_cupti = getenv("PAPI_CUDA_CUPTI");
    if (papi_cuda_cupti != NULL && path_to_nvcc == NULL) {
        path_to_nvcc = find_path_to_nvcc(papi_cuda_cupti);
    }

    char *papi_cuda_perfworks = getenv("PAPI_CUDA_PERFWORKS");
    if (papi_cuda_perfworks != NULL && path_to_nvcc == NULL) {
        path_to_nvcc = find_path_to_nvcc(papi_cuda_perfworks);
    }

    char *papi_cuda_root = getenv("PAPI_CUDA_ROOT");
    if (papi_cuda_root != NULL && path_to_nvcc == NULL) {
        path_to_nvcc = find_path_to_nvcc(papi_cuda_root);
    }

    int strLen;
    char *ld_library_path = getenv("LD_LIBRARY_PATH");
    if (ld_library_path != NULL && path_to_nvcc == NULL) {
        size_t length_to_allocate = strlen(ld_library_path) + 1;
        char *ld_library_path_copy = (char *) malloc(length_to_allocate * sizeof(char));
        if (ld_library_path_copy == NULL) {
            cuptic_err_set_last("Memory allocation for ld_library_path_copy failed. Set PAPI_CUDA_ROOT.");
            return PAPI_ENOMEM;
        }

        strLen = snprintf(ld_library_path_copy, length_to_allocate, "%s", ld_library_path);
        if (strLen < 0 || (size_t) strLen >= length_to_allocate) {
            cuptic_err_set_last("Unable to store contents of LD_LIBRARY_PATH into the buffer ld_library_path_copy. Set PAPI_CUDA_ROOT.");
            free(ld_library_path_copy);
            return PAPI_EBUF;
        }

        const char *delims = ":";
        char *directory_path = strtok(ld_library_path_copy, delims);
        while(directory_path) {
            path_to_nvcc = find_path_to_nvcc(directory_path);
            if (path_to_nvcc != NULL) {
                break;
            }
            directory_path = strtok(NULL, ":");
        }
        free(ld_library_path_copy);
    }

    // By design do not hard fail if path_to_nvcc is NULL
    if (path_to_nvcc == NULL) {
        SUBDBG("Failed to obtain path leading to bin/nvcc. Set PAPI_CUDA_ROOT if the cuda component is not active.\n");
        return PAPI_OK;
    }

    char command[PATH_MAX] = { 0 };
    strLen = snprintf(command, sizeof(command), "%s --list-gpu-arch 2>/dev/null", path_to_nvcc);
    if (strLen < 0 || (size_t) strLen >= sizeof(command)) {
        cuptic_err_set_last("Path to bin/nvcc with --list-gpu-arch 2>/dev/null exceeds PATH_MAX.");
        free((char*) path_to_nvcc);
        return PAPI_EBUF;
    }
    free((char*) path_to_nvcc);

    const char *mode = "r";
    FILE *fp_virtual_dev_arches = popen(command, mode);
    if (fp_virtual_dev_arches == NULL) {
        cuptic_err_set_last("Call to popen to get list of virtual devices failed.");
        return PAPI_ESYS;
    }

    int status;
    char all_supported_virtual_dev_arches[PAPI_HUGE_STR_LEN] = { 0 };
    char name_of_virtual_dev_arch[PAPI_MAX_STR_LEN] = { 0 };
    while (fgets(name_of_virtual_dev_arch, sizeof(name_of_virtual_dev_arch), fp_virtual_dev_arches) != NULL) {
        size_t current_length = strlen(all_supported_virtual_dev_arches);
        size_t remaining_length = sizeof(all_supported_virtual_dev_arches) - current_length;

        // Strip the newline if applicable
        size_t virtual_arch_len = strlen(name_of_virtual_dev_arch);
        if (virtual_arch_len > 0 && name_of_virtual_dev_arch[virtual_arch_len - 1] == '\n') {
            name_of_virtual_dev_arch[virtual_arch_len - 1] = '\0';
        }

        strLen = snprintf(all_supported_virtual_dev_arches + current_length, remaining_length, "%s,", name_of_virtual_dev_arch);
        if (strLen < 0 || (size_t) strLen >= remaining_length) {
            cuptic_err_set_last("Unable to store all of the virtual devices in the buffer all_supported_virtual_dev_arches.");
 
            status = pclose(fp_virtual_dev_arches);
            if (status != 0){
                SUBDBG("Failed to close stream opened by popen.\n");
            }

            return PAPI_EBUF;
        }
    }

    status = pclose(fp_virtual_dev_arches);
    if (status != 0){
        SUBDBG("Failed to close stream opened by popen.\n");
    }

    if (all_supported_virtual_dev_arches[0] == '\0') {
        cuptic_err_set_last("No output obtained for nvcc --list-gpu-arch. This could be due to incorrect paths or NVIDIA removing --list-gpu-arch. Set PAPI_CUDA_ROOT.");
        return PAPI_ESYS;
    }

    int nvidia_device_count = 0;
    cudaArtCheckErrors( cudaGetDeviceCountPtr(&nvidia_device_count), return PAPI_EMISC );

    char compute_capability[PAPI_MAX_STR_LEN] = { 0 };
    int dev_idx;
    for (dev_idx = 0; dev_idx < nvidia_device_count; dev_idx++) {
        struct cudaDeviceProp prop = { { 0 } };
        cudaArtCheckErrors( cudaGetDevicePropertiesPtr(&prop, dev_idx), return PAPI_EMISC );
        strLen = snprintf(compute_capability, sizeof(compute_capability), "%d%d", prop.major, prop.minor);
        if (strLen < 0 || (size_t) strLen >= sizeof(compute_capability)) {
            cuptic_err_set_last("Unable to store the major,minor compute capability obtained from cudaGetDeviceProperties in the buffer compute_capability.");
            return PAPI_EBUF;
        }

        if (strstr(all_supported_virtual_dev_arches, compute_capability) == NULL) {
            char error_message[PAPI_HUGE_STR_LEN] = { 0 };
            strLen = snprintf(error_message, sizeof(error_message), "Your current Cuda Toolkit version cannot be used with device %d."
                              " Either user a newer Cuda Toolkit version or utilize 'export CUDA_VISIBLE_DEVICES' if on a mixed cc machine.\n", dev_idx); 
            if (strLen < 0 || (size_t) strLen >= sizeof(error_message)) { // removing could be fine such that we actually make it to PAPI_ECMP.
                SUBDBG("Failed to fully write error message for incompatible Cuda Toolkit and device.\n");
            }

            cuptic_err_set_last(error_message);
            return PAPI_ECMP;
        }
    }

    return PAPI_OK;
}

int cuptic_init(void)
{
    int papi_errno = util_load_cuda_sym();
    if (papi_errno != PAPI_OK) {
        cuptic_err_set_last("Unable to load CUDA library functions.");
        return papi_errno;
    }

    papi_errno = verify_cuda_toolkit_supports_architectures_on_machine();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    sys_compute_capabilities_e system_ccs;
    papi_errno = compute_capabilities_on_system(&system_ccs);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    // Get an array of the available devices on the system
    papi_errno = get_enabled_devices();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    // Handle a partially disabled Cuda component
    if (system_ccs == sys_gpu_ccs_mixed || system_ccs == sys_gpu_ccs_all_lte_70 || system_ccs == sys_gpu_ccs_all_gte_70) {
        char *PAPI_CUDA_API = getenv("PAPI_CUDA_API");
        char *cc_support = (PAPI_CUDA_API != NULL) ? "<=7.0" : ">=7.0";
        char *helpMsg  = (PAPI_CUDA_API != NULL) ? "To enable support for CCs >= 7.0 run 'unset PAPI_CUDA_API'" : "To enable support for CCs <= 7.0 run 'export PAPI_CUDA_API=LEGACY'";

        char errMsg[PAPI_HUGE_STR_LEN];
        int strLen = snprintf(errMsg, PAPI_HUGE_STR_LEN,
                              "System includes multiple compute capabilities: <7.0, =7.0, >7.0."
                              " Only support for CC %s enabled. %s.", cc_support, helpMsg);
        if (strLen < 0 || strLen >= PAPI_HUGE_STR_LEN) {
            SUBDBG("Failed to fully write the partially disabled error message.\n");
            return PAPI_EBUF;
        }
        cuptic_err_set_last(errMsg);

        isCudaPartial = 1;

        return PAPI_PARTIAL;
    }

    return PAPI_OK;
}

void cuptic_partial(int *isCmpPartial, int **cudaEnabledDeviceIds, size_t *totalNumEnabledDevices)
{
    *isCmpPartial = isCudaPartial;
    *cudaEnabledDeviceIds = enabledDeviceIds;
    *totalNumEnabledDevices = enabledDevicesCnt;
    return;
}

int cuptic_determine_runtime_api(void) 
{
    char *PAPI_CUDA_API = (getenv("PAPI_CUDA_API") != NULL) ? "LEGACY" : "PERFWORKS";

    unsigned int cuptiVersion = util_dylib_cupti_version();
    // For the Event and Metric API to be operational in the Cuda component,
    // users must link with a Cuda toolkit version that has a CUPTI version < 13000.
    if (cuptiVersion >= CUPTI_EVENT_AND_METRIC_MAX_SUPPORTED_VERSION && strcasecmp(PAPI_CUDA_API, "LEGACY") == 0) {
        cuptic_err_set_last("Cannot use Event and Metric APIs with Cuda Toolkit versions >= 13.0 as support was removed.\n");
        return PAPI_EINVAL;
    }
    // For the Perfworks API to be operational in the Cuda component,
    // users must link with a Cuda toolkit version that has a CUPTI version >= 13.
    else if (cuptiVersion < CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION && strcasecmp(PAPI_CUDA_API, "PERFWORKS")) {
        cuptic_err_set_last("CUPTI version >= 13 is required for Perfworks API functionality.\n");
        return PAPI_EINVAL;
    }

    // Determine the compute capabilities on the system
    sys_compute_capabilities_e system_ccs;
    int papi_errno = compute_capabilities_on_system(&system_ccs);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    // Check to make sure a user has not tried to set PAPI_CUDA_API with only devices
    // with CCs > 7.0
    if (system_ccs == sys_gpu_ccs_all_gt_70 && strcasecmp(PAPI_CUDA_API, "LEGACY") == 0) {
        cuptic_err_set_last("Devices detected have CCs > 7.0; therefore, only the Perfworks Metric API is supported. Unset PAPI_CUDA_API.\n");
        return  PAPI_EINVAL;
    }

    int cupti_api = -1;
    // Determine which CUPTI API will be in use
    switch (system_ccs) {
        // All devices have CCs < 7.0
        case sys_gpu_ccs_all_lt_70:
            cupti_api = API_LEGACY;
            break;
        // All devices have CCs > 7.0
        case sys_gpu_ccs_all_gt_70:
            cupti_api = API_PERFWORKS;
            break;
        // All devices have CCs <= 7.0
        case sys_gpu_ccs_all_lte_70:
        // All devices have CCs >= 7.0
        case sys_gpu_ccs_all_gte_70:
        // ALL devices have CC's = 7.0
        case sys_gpu_ccs_all_eq_70:
        // Devices are mixed with CC's > 7.0 and CC's < 7.0
        case sys_gpu_ccs_mixed:
            // Default will be to use Perfworks API, user can change this by setting PAPI_CUDA_API.
            cupti_api = API_PERFWORKS;
            if (PAPI_CUDA_API != NULL) {
                int result = strcasecmp(PAPI_CUDA_API, "LEGACY");
                if (result == 0)
                    cupti_api = API_LEGACY;
            }
            break;
        default:
            SUBDBG("Implemented CUPTI APIs do not support the current GPU configuration.\n");
            break;
    }

    return cupti_api;
}

int get_enabled_devices(void)
{
    int total_gpus; 
    int papi_errno = cuptic_device_get_count(&total_gpus);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }   

    int cupti_api = cuptic_determine_runtime_api();
    if (cupti_api < 0) {
        return PAPI_ECMP;
    }   

    int i, cc, collectCudaDevice;
    for (i = 0; i < total_gpus; i++) {
        collectCudaDevice = 0;
        papi_errno = get_gpu_compute_capability(i, &cc);
        if (papi_errno != PAPI_OK) {
            return papi_errno;
        }

        if (cupti_api == API_PERFWORKS && cc >= 70) {
            collectCudaDevice = 1;    
        }
        else if (cupti_api == API_LEGACY && cc <= 70) {
            collectCudaDevice = 1;
        }

        if (collectCudaDevice) {
            enabledDeviceIds[enabledDevicesCnt] = i;
            enabledDevicesCnt++; 
        }
    }

    return PAPI_OK;
}

/** @class cuptic_ctxarr_create
  * @brief Allocate memory for pinfo.
  * @param *pinfo
  *    Instance of a struct that holds read count, running, cuptic_t
  *    and cuptip_gpu_state_t.
*/
int cuptic_ctxarr_create(cuptic_info_t *pinfo)
{
    COMPDBG("Entering.\n");
    int total_gpus, papi_errno;

    /* retrieve total number of compute-capable devices */
    papi_errno = cuptic_device_get_count(&total_gpus);
    if (papi_errno != PAPI_OK) {
        return PAPI_EMISC;
    }
  
    /* allocate memory */ 
    *pinfo = (cuptic_info_t) calloc (total_gpus, sizeof(*pinfo));
    if (*pinfo == NULL) {
        return PAPI_ENOMEM;
    }

    return PAPI_OK;
}


/** @class cuptic_ctxarr_update_current
  * @brief Updating the current Cuda context.
  * @param info
  *    Struct that contains a Cuda context, that can be indexed into based
  *    on device id.
  * @param evt_dev_id
  *    Device id from an appended device qualifier (e.g. :device=#).
*/
int cuptic_ctxarr_update_current(cuptic_info_t info, int evt_dev_id)
{
    CUcontext pctx;
    CUresult cuda_err;
    CUdevice dev_id;

    // If a Cuda context already exists, get it
    cuda_err = cuCtxGetCurrentPtr(&pctx);
    if (cuda_err != CUDA_SUCCESS) {
        return PAPI_EMISC;
    }

    // Get the Device ID for the existing Cuda context
    if (pctx != NULL) {
        cuda_err = cuCtxGetDevicePtr(&dev_id);
        if (cuda_err != CUDA_SUCCESS) {
            return PAPI_EMISC;
        }
    }

    // A context is not stored for the :device=# qualifier
    if (info[evt_dev_id].ctx == NULL) {
        // Cuda context was not found or a user did not provide an appropriate Cuda context for the
        // device qualifier id that was supplied
        if (pctx == NULL || dev_id != evt_dev_id) {
            // If multiple devices are found on the machine, then we need to call cudaSetDevice
            SUBDBG("A Cuda context was not found. Therefore, one is created for device: %d\n", evt_dev_id);
            cudaArtCheckErrors(cudaSetDevicePtr(evt_dev_id), return PAPI_EMISC);
            cudaArtCheckErrors(cudaFreePtr(0), return PAPI_EMISC);

            cudaCheckErrors(cuCtxGetCurrentPtr(&info[evt_dev_id].ctx), return PAPI_EMISC);
            cudaCheckErrors(cuCtxPopCurrentPtr(&info[evt_dev_id].ctx), return PAPI_EMISC);
        }
        // Cuda context was found
        else {
            SUBDBG("A cuda context was found for device: %d\n", evt_dev_id);
            cudaCheckErrors(cuCtxGetCurrentPtr(&info[evt_dev_id].ctx), return PAPI_EMISC);
        }
    }
    // If the Cuda context has changed for a device keep the first one seen, but output a warning
    else if (pctx != NULL){
        if (evt_dev_id == dev_id) {
            if (info[dev_id].ctx != pctx) {
                ERRDBG("Warning: cuda context for device %d has changed from %p to %p\n", dev_id, info[dev_id].ctx, pctx);
            }
        }
    }

    return PAPI_OK;
}

int cuptic_ctxarr_get_ctx(cuptic_info_t info, int gpu_idx, CUcontext *ctx)
{
    *ctx = info[gpu_idx].ctx;
    return PAPI_OK;
}

int cuptic_ctxarr_destroy(cuptic_info_t *pinfo)
{
    free(*pinfo);
    *pinfo = NULL;
    return PAPI_OK;
}

int _devmask_events_get(cuptiu_event_table_t *evt_table, gpu_occupancy_t *bitmask)
{
    gpu_occupancy_t acq_mask = 0;
    long i;
    for (i = 0; i < evt_table->count; i++) {
        acq_mask |= (1 << evt_table->cuda_devs[i]);
    }
    *bitmask = acq_mask;

    return PAPI_OK;
}

int cuptic_device_acquire(cuptiu_event_table_t *evt_table)
{
    gpu_occupancy_t bitmask;
    int papi_errno = _devmask_events_get(evt_table, &bitmask);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    if (bitmask & global_gpu_bitmask) {
        return PAPI_ECNFLCT;
    }
    _papi_hwi_lock(_cuda_lock);
    global_gpu_bitmask |= bitmask;
    _papi_hwi_unlock(_cuda_lock);
    return PAPI_OK;
}

int cuptic_device_release(cuptiu_event_table_t *evt_table)
{
    gpu_occupancy_t bitmask;
    int papi_errno = _devmask_events_get(evt_table, &bitmask);
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }
    if ((bitmask & global_gpu_bitmask) != bitmask) {
        return PAPI_EMISC;
    }
    _papi_hwi_lock(_cuda_lock);
    global_gpu_bitmask ^= bitmask;
    _papi_hwi_unlock(_cuda_lock);
    return PAPI_OK;
}

/** @class cuptiu_dev_set
  * @brief For a Cuda native event, set the device ID.
  *
  * @param *bitmap
  *   Device map.
  * @param i
  *   Device ID.
*/
int cuptiu_dev_set(cuptiu_bitmap_t *bitmap, int i)
{
    *bitmap |= (1ULL << i);
    return PAPI_OK;
}

/** @class cuptiu_dev_check
  * @brief For a Cuda native event, check for a valid device ID.
  *
  * @param *bitmap
  *   Device map.
  * @param i
  *   Device ID.
*/
int cuptiu_dev_check(cuptiu_bitmap_t bitmap, int i)
{
    return (bitmap & (1ULL << i));
}

int get_chip_name(int dev_num, char* chipName)
{
    int papi_errno;
    CUpti_Device_GetChipName_Params getChipName = {
        .structSize = CUpti_Device_GetChipName_Params_STRUCT_SIZE,
        .pPriv = NULL,
        .deviceIndex = 0
    };
    getChipName.deviceIndex = dev_num;
    papi_errno = cuptiDeviceGetChipNamePtr(&getChipName);
    if (papi_errno != CUPTI_SUCCESS) {
        ERRDBG("CUPTI error %d: Failed to get chip name for device %d\n", papi_errno, dev_num);
        return PAPI_EMISC;
    }
    strcpy(chipName, getChipName.pChipName);
    return PAPI_OK;
}
