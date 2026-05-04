//--------------- test_target_is_accessible_with_usm.c---------------------//
//
// OpenMP API Version 5.1 Aug 2021
//
// This test checks the omp_target_is_accessible device routine.
// In this test the output of the target_is_accessible call should return
// true because the requires unified shared memory is used.
//-----------------------------------------------------------------------//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 100

#pragma omp requires unified_shared_memory

int check_device(){
  int errors = 0;
  int check_test = 0;
  const int buf_size = sizeof(int) * N;
  const int dev = omp_get_default_device();

  int *ptr = (int *) malloc(buf_size);

  if(ptr == NULL){
    OMPVV_INFOMSG("Memory allocation on host failed");
    return 1;
  }
  for(int i = 0;i < N; i++)
    ptr[i] = i;

  check_test = omp_target_is_accessible(ptr, buf_size, dev);

  OMPVV_TEST_AND_SET_VERBOSE(errors, check_test == 0);

  OMPVV_INFOMSG_IF(check_test != 0, "The host memory is accessible from the default device.");

  if(check_test){
  #pragma omp target is_device_ptr(ptr) device(dev)
  {
    for(int i = 0;i < N; i++)
      ptr[i] = ptr[i] * i;
  }

  for(int i = 0;i < N; i++)
   OMPVV_ERROR_IF(ptr[i] != i * i, "Operations on the host memory that was reported as accessible produced incorrect results.");

  }
  else
  OMPVV_ERROR ("Host memory was reported inaccessible despite of 'requires unified_shared_memory' specified.");

  free(ptr);

  return errors;
}

int main(){
int errors = 0;

OMPVV_TEST_OFFLOADING;

OMPVV_TEST_AND_SET_VERBOSE(errors, check_device() != 0);

OMPVV_REPORT_AND_RETURN(errors);
	
}
