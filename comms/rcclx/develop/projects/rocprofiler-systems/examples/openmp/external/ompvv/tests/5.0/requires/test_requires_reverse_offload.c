//===---test_requires_reverse_offload.c ---------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
// 
// This test checks to see that the reverse_offload clause on a requires directive 
// is supported, and if so, the function host_function() will made available as a 
// procedure only on the host. By specificying the 'ancestor' modifier with device 
// number of 1, we are indicating to compiler that execution is to be performed on
// the immediate parent, the host. The omp declare target statement ensures that the
// host_function will only be available on host.  
//
// Based on OpenMP 5.0 Example: target_reverse_offload.7.c
//===--------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

#pragma omp requires reverse_offload

int main() 
{
    int A[N];	
    int isOffloading;
    int errors, errors2;
    int device_num;
    int is_shared_env = 0;

    errors = 0;
   
    OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);

    OMPVV_WARNING_IF(!isOffloading, "Without offloading enabled, host execution is already guaranteed");

    device_num = omp_get_num_devices();

    for (int i = 0; i < N; i++) {
        A[i] = i;
    }

    OMPVV_WARNING_IF(device_num <= 0, "Cannot properly test reverse offload if no devices are available");
    OMPVV_TEST_AND_SET_SHARED_ENVIRONMENT(is_shared_env);
    OMPVV_WARNING_IF(is_shared_env != 0, "[WARNING] May not be able to detect errors if the target system supports shared memory.")

    errors2 = 0;
    
    #pragma omp target map(tofrom: errors2) map(to:A, is_shared_env) // Run on the default device, which is the host for device_num = 0
    {
       #pragma omp target device(ancestor:1) map(always, to: A)
       for (int j = 0; j < N; j++) {
          A[j] = 2*j;
       } 
       if( (omp_is_initial_device() == 0) && (is_shared_env == 0) ) {
          for (int i = 0; i < N; i++) {
             if( A[i] != i) { errors2 = errors2 + 1; }
          }
       }   

    }

    OMPVV_TEST_AND_SET_VERBOSE(errors, errors2 != 0);
    
    for (int i = 0; i < N; i++) {
       OMPVV_TEST_AND_SET(errors, A[i] != 2*i);
    }

    OMPVV_REPORT_AND_RETURN(errors)
}


