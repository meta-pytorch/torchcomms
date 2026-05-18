/**
 * @file   common.h
 * @author Giuseppe Congiu
 *         gcongiu@icl.utk.edu
 *
 */

#ifndef __COMMON_H__
#define __COMMON_H__

#include "papi.h"
#include "papi_test.h"
#include "matmul.h"

static inline void
hip_test_fail(const char *file __attribute__((unused)), int line,
              const char *call, hipError_t retval)
{
    const char *string;

    fprintf(stdout, "FAILED!!!");
    fprintf(stdout, "\nLine # %d ", line);

    string = hipGetErrorString(retval);
    fprintf(stdout, "Error in %s: %s\n", call, string);

    if (PAPI_is_initialized()) {
        PAPI_shutdown();
    }

    exit(1);
}

static inline int
match_expected_counter(long long expected, long long value)
{
    return (expected == value);
}

#endif /* End of __COMMON_H__ */
