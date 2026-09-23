/**
 * @file    cupti_config.h
 */

#ifndef __LCUDA_CONFIG_H__
#define __LCUDA_CONFIG_H__

#include <cupti.h>

// Used to assign the EventSet state
#define CUDA_EVENTS_STOPPED (0x0)
#define CUDA_EVENTS_RUNNING (0x2)

#define API_PERFWORKS 1
#define CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION  (13)

#define API_LEGACY 2
#define CUPTI_EVENT_AND_METRIC_MAX_SUPPORTED_VERSION (13000)

#define PAPI_CUDA_MPX_COUNTERS 512
#define PAPI_CUDA_MAX_COUNTERS  30

#endif  /* __LCUDA_CONFIG_H__ */
