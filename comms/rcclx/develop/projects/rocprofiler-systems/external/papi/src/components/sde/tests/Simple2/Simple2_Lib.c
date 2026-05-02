#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
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

long long int counter_accessor_function( void *param );

void simple_init(void){

    // Initialize library specific variables
    comp_value = 0.0;
    total_iter_cnt = 0;
    low_wtrmrk = 0;
    high_wtrmrk = 0;

    // Initialize PAPI SDEs
    handle = papi_sde_init("Simple2");
    papi_sde_register_counter_cb(handle, ev_names[0], PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_double, counter_accessor_function, &comp_value);
    papi_sde_register_counter(handle, ev_names[1], PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_long_long, &total_iter_cnt);
    papi_sde_register_counter(handle, ev_names[2], PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_long_long, &low_wtrmrk);
    papi_sde_register_counter(handle, ev_names[3], PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_long_long, &high_wtrmrk);
    papi_sde_add_counter_to_group(handle, ev_names[2], "ANY_WATERMARK_REACHED", PAPI_SDE_SUM);
    papi_sde_add_counter_to_group(handle, ev_names[3], "ANY_WATERMARK_REACHED", PAPI_SDE_SUM);

    return;
}

// The following function will _NOT_ be called by other libray functions or normal
// applications. It is a hook for the utility 'papi_native_avail' to be able to
// discover the SDEs that are exported by this library.
papi_handle_t papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct){
    handle = fptr_struct->init("Simple2");
    fptr_struct->register_counter_cb(handle, ev_names[0], PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_double, counter_accessor_function, &comp_value);
    fptr_struct->register_counter(handle, ev_names[1], PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_long_long, &total_iter_cnt);
    fptr_struct->register_counter(handle, ev_names[2], PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_long_long, &low_wtrmrk);
    fptr_struct->register_counter(handle, ev_names[3], PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_long_long, &high_wtrmrk);
    fptr_struct->add_counter_to_group(handle, ev_names[2], "ANY_WATERMARK_REACHED", PAPI_SDE_SUM);
    fptr_struct->add_counter_to_group(handle, ev_names[3], "ANY_WATERMARK_REACHED", PAPI_SDE_SUM);

    fptr_struct->describe_counter(handle, ev_names[0], "Sum of values that are within the watermarks.");
    fptr_struct->describe_counter(handle, ev_names[1], "Total iterations executed by the library.");
    fptr_struct->describe_counter(handle, ev_names[2], "Number of times a value was below the low watermark.");
    fptr_struct->describe_counter(handle, ev_names[3], "Number of times a value was above the high watermark.");
    fptr_struct->describe_counter(handle, "ANY_WATERMARK_REACHED",  "Number of times a value was not between the two watermarks.");

    return handle;
}

// This function allows the library to perform operations in order to compute the value of an SDE at run-time
long long counter_accessor_function( void *param ){
    long long ll;
    double *dbl_ptr = (double *)param;

    // Scale the variable by a factor of two. Real libraries will do meaningful work here.
    double value = *dbl_ptr * 2.0;

    // Copy the bits of the result in a long long int.
    (void)memcpy(&ll, &value, sizeof(double));

    return ll;
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
