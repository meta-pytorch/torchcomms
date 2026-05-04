//===------------- test_pin_device.c --------------------===//
//
// OpenMP API Version 6.0
//
// This test checks whether the pin device mem allocator works
// as intended. The test creates an allocaiton with the pin_device
// trait, then accesses it and modifies on the target device.
// It checks if it was accessible on the target device and if the 
// modifications were successful.
//
//===----------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include "ompvv.h"
#define N 1024

int test_pin_device() {
    int errors = 0;
    int target_device = omp_get_initial_device();
    int *arr;
    
    omp_alloctrait_t trait = { omp_atk_pin_device, target_device };
    omp_allocator_handle_t pin_dev_alloc = omp_init_allocator(omp_default_mem_space, 1, &trait);
    
    arr = (int*)omp_alloc(N*sizeof(int), pin_dev_alloc);
    
    if (arr == NULL) {
        OMPVV_ERROR("Allocation failed when using omp_alloc with pin_device trait");
        errors++;
        return errors;
    }
    
    for (int i = 0; i < N; i++) {
        arr[i] = i;
    }
    
    #pragma omp target map(tofrom: arr[0:N])
    {
        for (int i = 0; i < N; i++) {
            arr[i] *= 2;
        }   
    }
    
    for (int i = 0; i < N; i++) {
        OMPVV_TEST_AND_SET_VERBOSE(errors, arr[i] != i * 2);
    }
    
    omp_free(arr, pin_dev_alloc);
    omp_destroy_allocator(pin_dev_alloc);
    return errors;
}

int main() {
    OMPVV_TEST_OFFLOADING;
    int errors = 0;
    OMPVV_TEST_AND_SET_VERBOSE(errors, test_pin_device() != 0);
    OMPVV_REPORT_AND_RETURN(errors);
}
