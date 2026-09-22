#ifndef __PAPI_ROCPSDK_INTERNAL_H__
#define __PAPI_ROCPSDK_INTERNAL_H__

#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/device_counting_service.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "papi.h"

#ifdef __cplusplus
extern "C"
{
#endif

#include "papi_internal.h"
#include "papi_memory.h"

#ifdef __cplusplus
}
#endif

#include <dlfcn.h>
#include <cxxabi.h>
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <set>
#include <map>
#include <unordered_map>
#include <list>
#include <mutex>
#if (__cplusplus >= 201402L) // c++14
#include <shared_mutex>
#endif
#include <condition_variable>
#include <regex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

// required, single name
#define DLL_SYM_CHECK_2(name, type)                               \
do {                                                               \
    char* __err;                                                   \
    /* clear any prior dlerror */                                  \
    dlerror();                                                     \
    name##_FPTR = (type) dlsym(dllHandle, #name);                  \
    __err = dlerror();                                             \
    if(__err != NULL) {                                            \
        return __err;                                              \
    }                                                              \
} while(0)

// required, either of two names (try alt first, then primary)
#define DLL_SYM_CHECK_3(name, alt, type)                           \
do {                                                               \
    char* __err;                                                   \
    void* __p;                                                     \
    dlerror();                                                     \
    __p = dlsym(dllHandle, alt);                                   \
    __err = dlerror();                                             \
    if(__err == NULL && __p) {                                     \
        name##_FPTR = (type) __p;                                  \
    } else {                                                       \
        dlerror();                                                 \
        name##_FPTR = (type) dlsym(dllHandle, #name);              \
        __err = dlerror();                                         \
        if(__err != NULL) {                                        \
            return __err;                                          \
        }                                                          \
    }                                                              \
} while(0)

// arity dispatcher:  DLL_SYM_CHECK(name, type)  or  DLL_SYM_CHECK(name, "alt", type)
#define __DLL_SYM_GET_MACRO(_1,_2,_3,NAME,...) NAME
#define DLL_SYM_CHECK(...) __DLL_SYM_GET_MACRO(__VA_ARGS__, DLL_SYM_CHECK_3, DLL_SYM_CHECK_2)(__VA_ARGS__)


#if defined(PAPI_ROCPSDK_DEBUG)
#define ROCPROFILER_CALL(result, msg)                                                              \
    {                                                                                              \
        rocprofiler_status_t CHECKSTATUS = result;                                                 \
        if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                              \
        {                                                                                          \
            std::string status_msg = rocprofiler_get_status_string_FPTR(CHECKSTATUS);              \
            std::cerr << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg            \
                      << " failed with error code " << CHECKSTATUS << ": " << status_msg           \
                      << std::endl;                                                                \
            std::stringstream errmsg{};                                                            \
            errmsg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] " << msg " failure ("  \
                   << status_msg << ")";                                                           \
            throw std::runtime_error(errmsg.str());                                                \
        }                                                                                          \
    }
#else
#define ROCPROFILER_CALL(result, msg) {(void)result;}
#endif

#define RPSDK_MODE_DISPATCH        (0)
#define RPSDK_MODE_DEVICE_SAMPLING (1)

#define RPSDK_AES_STOPPED (0x0)
#define RPSDK_AES_OPEN    (0x1)
#define RPSDK_AES_RUNNING (0x2)

typedef struct {
    char name[PAPI_MAX_STR_LEN];
    char descr[PAPI_2MAX_STR_LEN];
} ntv_event_t;

typedef struct {
    ntv_event_t *events;
    int num_events;
} ntv_event_table_t;

struct vendord_ctx {
    int state;
    int *event_ids;
    long long *counters;
    int num_events;
};

typedef struct vendord_ctx *vendorp_ctx_t;

static int init_ctx(int *event_ids, int num_events, vendorp_ctx_t ctx);
static int finalize_ctx(vendorp_ctx_t ctx);

extern unsigned int _rocp_sdk_lock;

#endif
