#ifndef __ROC_PROFILER_CONFIG_H__
#define __ROC_PROFILER_CONFIG_H__

#include <stdint.h>

#define PAPI_ROCM_MAX_COUNTERS (512)

#define ROCM_PROFILE_SAMPLING_MODE (0x0)
#define ROCM_EVENTS_OPENED         (0x1)
#define ROCM_EVENTS_RUNNING        (0x2)

extern unsigned int _rocm_lock;
extern unsigned int rocm_prof_mode;

#endif /* End of __ROC_PROFILER_CONFIG_H__ */
