#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "sde_lib.h"

// API functions
void simple_init(void);
double simple_compute(double x);

// The following counters are hidden to programs linking with
// this library, so they can not be accessed directly.
static double comp_value;
static long long int total_iter_cnt, low_wtrmrk, high_wtrmrk;
static papi_handle_t handle;

static const char *ev_names[4] = {
    "COMPUTED_VALUE",
    "TOTAL_ITERATIONS",
    "LOW_WATERMARK_REACHED",
    "HIGH_WATERMARK_REACHED"
};


void simple_init(void){

    // Initialize library specific variables
    comp_value = 0.0;
    total_iter_cnt = 0;
    low_wtrmrk = 0;
    high_wtrmrk = 0;

    // Initialize PAPI SDEs
    handle = papi_sde_init("Simple");
    papi_sde_register_counter(handle, ev_names[0], PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_double, &comp_value);
    papi_sde_register_counter(handle, ev_names[1], PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_long_long, &total_iter_cnt);
    papi_sde_register_counter(handle, ev_names[2], PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_long_long, &low_wtrmrk);
    papi_sde_register_counter(handle, ev_names[3], PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_long_long, &high_wtrmrk);

    return;
}

// Perform some nonsense computation to emulate a possible library behavior.
// Notice that no SDE routines need to be called in the critical path of the library.
double simple_compute(double x){
    double sum = 0.0;
    int lcl_iter = 0;

    if( x > 1.0 )
        x = 1.0/x;
    if( x < 0.000001 )
        x += 0.3579;

    while( 1 ){
        double y,x2,x3,x4;
        lcl_iter++;

        // Compute a function with range [0:1] so we can iterate
        // multiple times without diverging or creating FP exceptions.
        x2 = x*x;
        x3 = x2*x;
        x4 = x2*x2;
        y = 42.53*x4 -67.0*x3 +25.0*x2 +x/2.15;
        y = y*y;
        if( y < 0.01 )
            y = 0.5-y;

        // Now set the next x to be the current y, so we can iterate again.
        x = y;

        // Add y to sum unconditionally
        sum += y;

        if( y < 0.1 ){
            low_wtrmrk++;
            continue;
        }

        if( y > 0.9 ){
            high_wtrmrk++;
            continue;
        }

        // Only add y to comp_value if y is between the low and high watermarks.
        comp_value += y;

        // If some condition is met, terminate the loop
        if( 0.61 < y && y < 0.69 )
            break;
    }
    total_iter_cnt += lcl_iter;

    return sum;
}
