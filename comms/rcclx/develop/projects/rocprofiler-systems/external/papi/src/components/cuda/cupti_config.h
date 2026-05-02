/**
 * @file    cupti_config.h
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.)
 * @author  Anustuv Pal    anustuv@icl.utk.edu
 */

#ifndef __LCUDA_CONFIG_H__
#define __LCUDA_CONFIG_H__

#include <cupti.h>

/* used to assign the EventSet state  */
#define CUDA_EVENTS_STOPPED (0x0)
#define CUDA_EVENTS_RUNNING (0x2)

#define CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION  (13)

#if (CUPTI_API_VERSION >= CUPTI_PROFILER_API_MIN_SUPPORTED_VERSION)
#   define API_PERFWORKS 1
#endif

/*
 * TODO: When NVIDIA removes the event API #define CUPTI_EVENTS_API_MAX_SUPPORTED_VERSION
 * and set it to last version that supports it.
 * Then conditionally define the following macro if the version lies within this range.
 * Note: Introduce a runtime check in `cuptic_is_runtime_events_api()` to satisfy this.
 */
#define API_EVENTS 1

#endif  /* __LCUDA_CONFIG_H__ */
