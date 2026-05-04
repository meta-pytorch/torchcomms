#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "simple2.hpp"

Simple::Simple(){
    papi_sde::PapiSde sde("Simple2_CPP");

    // Initialize library specific variables
    comp_value = 0.0;
    total_iter_cnt = 0;
    low_wtrmrk = 0;
    high_wtrmrk = 0;

    // Initialize PAPI SDEs
    sde.register_fp_counter(ev_names[0], PAPI_SDE_RO|PAPI_SDE_DELTA, counter_accessor_function, comp_value);
    sde.register_counter(ev_names[1], PAPI_SDE_RO|PAPI_SDE_DELTA, total_iter_cnt);
    sde.register_counter(ev_names[2], PAPI_SDE_RO|PAPI_SDE_DELTA, low_wtrmrk);
    sde.register_counter(ev_names[3], PAPI_SDE_RO|PAPI_SDE_DELTA, high_wtrmrk);
    sde.add_counter_to_group(ev_names[2], "ANY_WATERMARK_REACHED", PAPI_SDE_SUM);
    sde.add_counter_to_group(ev_names[3], "ANY_WATERMARK_REACHED", PAPI_SDE_SUM);

    return;
}

// This function allows the library to perform operations in order to compute the value of an SDE at run-time
long long Simple::counter_accessor_function( double *param ){
    long long ll;
    double *dbl_ptr = param;

    // Scale the variable by a factor of two. Real libraries will do meaningful work here.
    double value = *dbl_ptr * 2.0;

    // Copy the bits of the result in a long long int.
    (void)memcpy(&ll, &value, sizeof(double));

    return ll;
}

// Perform some nonsense computation to emulate a possible library behavior.
// Notice that no SDE routines need to be called in the critical path of the library.
double Simple::simple_compute(double x){
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

// The following function will _NOT_ be called by other libray functions or normal
// applications. It is a hook for the utility 'papi_native_avail' to be able to
// discover the SDEs that are exported by this library.
extern "C" papi_handle_t papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct){
    papi_handle_t handle = fptr_struct->init("Simple2_CPP");
    handle = fptr_struct->init("Simple2_CPP");
    fptr_struct->register_fp_counter(handle, Simple::ev_names[0], PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_double, NULL, NULL);
    fptr_struct->register_counter(handle, Simple::ev_names[1], PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_long_long, NULL);
    fptr_struct->register_counter(handle, Simple::ev_names[2], PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_long_long, NULL);
    fptr_struct->register_counter(handle, Simple::ev_names[3], PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_long_long, NULL);
    fptr_struct->add_counter_to_group(handle, Simple::ev_names[2], "ANY_WATERMARK_REACHED", PAPI_SDE_SUM);
    fptr_struct->add_counter_to_group(handle, Simple::ev_names[3], "ANY_WATERMARK_REACHED", PAPI_SDE_SUM);

    fptr_struct->describe_counter(handle, Simple::ev_names[0], "Sum of values that are within the watermarks.");
    fptr_struct->describe_counter(handle, Simple::ev_names[1], "Total iterations executed by the library.");
    fptr_struct->describe_counter(handle, Simple::ev_names[2], "Number of times a value was below the low watermark.");
    fptr_struct->describe_counter(handle, Simple::ev_names[3], "Number of times a value was above the high watermark.");
    fptr_struct->describe_counter(handle, "ANY_WATERMARK_REACHED",  "Number of times a value was not between the two watermarks.");

    return handle;
}
