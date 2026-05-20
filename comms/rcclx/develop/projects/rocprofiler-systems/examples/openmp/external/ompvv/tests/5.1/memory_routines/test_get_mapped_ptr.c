//===--- test_get_mapped_ptr.c -----------------------------------------------===//
//
//  OpenMP API Version 5.1 Aug 2021
//
//Test for the omp_get_mapped_ptr memory routine. This routine returns the device pointer associated with a host pointer for a given device. 
//===--------------------------------------------------------------------------------------===//
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024

int errors;

int test_get_mapped_ptr() {
int * device_ptr;
int x = 4;
int num_devices = omp_get_num_devices(); 
int * arr_ptrs;
OMPVV_INFOMSG_IF(num_devices == 0, "No devices available. ");
// test to make sure it is NULL if no devices exist
if (num_devices == 0) {
	device_ptr = (int *) omp_get_mapped_ptr(&x, omp_get_initial_device());
	OMPVV_TEST_AND_SET(errors, device_ptr != &x);
	OMPVV_INFOMSG_IF(device_ptr != &x, "get_mapped_ptr() did not work with 0 devices.");
	OMPVV_INFOMSG_IF(device_ptr == &x, "get_mapped_ptr() worked with 0 devices. ");
}
for (int i = 0; i < num_devices; i ++) {
# pragma omp target enter data device(i) map(to:x)
	arr_ptrs = (int *) omp_get_mapped_ptr(&x, i);

	OMPVV_TEST_AND_SET(errors, arr_ptrs == NULL);
	OMPVV_INFOMSG_IF(arr_ptrs == NULL, "get_mapped_ptr() failed on getting device pointer. ");
	OMPVV_INFOMSG_IF(arr_ptrs != NULL, "get_mapped_ptr() mapped pointer to device. ");
# pragma omp target exit data map(from:x)
}


return 0;
}

int main() {
   errors = 0;
   OMPVV_TEST_OFFLOADING;
   OMPVV_TEST_AND_SET_VERBOSE(errors, test_get_mapped_ptr() != 0);
   OMPVV_REPORT_AND_RETURN(errors);
}
