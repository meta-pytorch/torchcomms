#include <inttypes.h>
#include <unistd.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>
#include <papi.h>
#include <omp.h>
#include "prepareArray.h"
#include "timing_kernels.h"

// For do_work macro in the header file
volatile double x,y;

extern int is_core;
char* eventname = NULL;

run_output_t probeBufferSize(long long active_buf_len, long long line_size, float pageCountPerBlock, int pattern, long long llc_size, uintptr_t **v, uintptr_t *rslt, int latency_only, int mode, int ONT){
    int _papi_eventset = PAPI_NULL;
    int retval, buffer = 0, status = 0;
    int error_line = -1, error_type = PAPI_OK;
    register uintptr_t *p = NULL;
    register uintptr_t p_prime;
    long long pageSize, blockSize;
    long long int counter[ONT];
    run_output_t out;
    out.status = 0;

    assert( sizeof(int) >= 4 );

    x = (double)*rslt;
    x = floor(1.3*x/(1.4*x+1.8));
    y = x*3.97;
    if( x > 0 || y > 0 )
        printf("WARNING: x=%lf y=%lf\n",x,y);

    long long countSingle = active_buf_len/line_size;
    long long threshold = 1024LL*1024LL/64LL;
    long long len = (active_buf_len > threshold) ? active_buf_len : threshold;
    long long countMax;
    if( len > llc_size ){
        countMax = 4LL*(len/line_size);
    }else{
        countMax = 64LL*(len/line_size);
    }

    // Get the size of a page of memory.
    pageSize = sysconf(_SC_PAGESIZE)/sizeof(uintptr_t);
    if( pageSize <= 0 ){
        fprintf(stderr,"Cannot determine pagesize, sysconf() returned an error code.\n");
        out.status = -1;
        return out;
    }

    // Compute the size of a block in the pointer chain and create the pointer chain.
    blockSize = (long long)(pageCountPerBlock*(float)pageSize);

    // Start of threaded benchmark.
    #pragma omp parallel private(p,retval) reduction(+:buffer) reduction(+:status) firstprivate(_papi_eventset) default(shared)
    {
        int idx = omp_get_thread_num();
        int thdStatus = 0;
        double divisor = 1.0;
        double time1=0, time2=0, dt, factor;
        long long count;

        // Initialize the result to a value indicating an error.
        // If no error occurs, it will be overwritten.
        if ( !latency_only ) {
            out.counter[idx] = -1;
        }

        status += prepareArray(v[idx], active_buf_len, line_size, blockSize, pattern);

        // We will use "p" even after the epilogue, so let's set
        // it here in case an error occurs.
        p = &v[idx][0];

        if ( !latency_only && (is_core || 0 == idx) ) {
            retval = PAPI_create_eventset( &_papi_eventset );
            if (retval != PAPI_OK ){
                error_type = retval;
                error_line = __LINE__;
                thdStatus = -1;
                // If we can't measure events, no need to run the kernel.
                goto skip_epilogue;
            }

            retval = PAPI_add_named_event( _papi_eventset, eventname );
            if (retval != PAPI_OK ){
                error_type = retval;
                error_line = __LINE__;
                thdStatus = -1;
                // If we can't measure events, no need to run the kernel.
                goto clean_up;
            }
        }

        // Make sure all threads start at about the same time so that we get high pressure on the memory subsystem.
        #pragma omp barrier

        count = countSingle;
        // Make a warm-up pass to fetch the data into the cache, if it fits in any cache.
        while(count > 0){
            N_128;
            count -= 128;
        }

        // Start the counters.
        if ( !latency_only && (is_core || 0 == idx) ) {
            retval = PAPI_start(_papi_eventset);
            if ( PAPI_OK != retval ) {
                error_type = retval;
                error_line = __LINE__;
                thdStatus = -1;
                // If we can't measure events, no need to run the kernel.
                goto clean_up;
            }
        }

        // Start the actual test.
        count = countMax;
        // Micro-kernel for memory reading.
        if( CACHE_READ_ONLY == mode || latency_only )
        {
            if( latency_only ) time1 = getticks();
            while(count > 0){
                N_128;
                count -= 128;
            }
            if( latency_only ) time2 = getticks();
        }
        // Micro-kernel for memory writing.
        else
        {
            while(count > 0){
                NW_128;
                count -= 128;
            }
        }

        if ( !latency_only && (is_core || 0 == idx) ) {
            // Stop the counters.
            retval = PAPI_stop(_papi_eventset, &counter[idx]);
            if ( PAPI_OK != retval ) {
                error_type = retval;
                error_line = __LINE__;
                thdStatus = -1;
                goto clean_up;
            }

            // Get the average event count per access in pointer chase.
            // If it is not a core event, get average count per thread.
            divisor = 1.0*countMax;
            if( !is_core && 0 == idx )
                divisor *= ONT;

            out.counter[idx] = (1.0*counter[idx])/divisor;

clean_up:
            retval = PAPI_cleanup_eventset(_papi_eventset);
            if (retval != PAPI_OK ){
                error_type = retval;
                error_line = __LINE__;
                thdStatus = -1;
            }

            retval = PAPI_destroy_eventset(&_papi_eventset);
            if (retval != PAPI_OK ){
                error_type = retval;
                error_line = __LINE__;
                thdStatus = -1;
            }

        }else{
            // Compute the duration of the pointer chase.
            dt = elapsed(time2, time1);

            // Convert time into nanoseconds.
            factor = 1000.0;

            // Number of accesses per pointer chase.
            factor /= (1.0*countMax);

            // Get the average nanoseconds per access.
            out.dt[idx] = dt*factor;
        }

skip_epilogue:
        buffer += (uintptr_t)p+(uintptr_t)(x+y);
        status += thdStatus;
    }

    // Get the collective status.
    if(status < 0) {
        error_handler(error_type, error_line);
        out.status = -1;
    }

    // Prevent compiler optimization.
    *rslt = buffer;

    return out;
}

void error_handler(int e, int line){
    int idx;
    const char *errors[26] = {
                              "No error",
                              "Invalid argument",
                              "Insufficient memory",
                              "A System/C library call failed",
                              "Not supported by component",
                              "Access to the counters was lost or interrupted",
                              "Internal error, please send mail to the developers",
                              "Event does not exist",
                              "Event exists, but cannot be counted due to counter resource limitations",
                              "EventSet is currently not running",
                              "EventSet is currently counting",
                              "No such EventSet Available",
                              "Event in argument is not a valid preset",
                              "Hardware does not support performance counters",
                              "Unknown error code",
                              "Permission level does not permit operation",
                              "PAPI hasn't been initialized yet",
                              "Component Index isn't set",
                              "Not supported",
                              "Not implemented",
                              "Buffer size exceeded",
                              "EventSet domain is not supported for the operation",
                              "Invalid or missing event attributes",
                              "Too many events or attributes",
                              "Bad combination of features",
                              "Component containing event is disabled"
    };

    idx = -e;
    if(idx >= 26 || idx < 0 )
        idx = 15;

    if( NULL != eventname )
        fprintf(stderr,"\nError \"%s\" occured at line %d when processing event %s.\n", errors[idx], line, eventname);
    else
        fprintf(stderr,"\nError \"%s\" occured at line %d.\n", errors[idx], line);

}
