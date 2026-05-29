//===--- test_atomic_num_hint_device.c ------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
//  This test checks if atomic hints passed by enum value
//  are processed correctly for device execution. If the sync hint is not
//  yet defined in the specification, it defaults to 
//  omp_sync_hint_none (0x0). 
////===--------------------------------------------------------------------===//

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"


int test_atomic_with_used_enum_value() {
  OMPVV_INFOMSG("test_atomic_with_used_enum_value");
  int errors = 0, x = 0, num_threads = -1;

#pragma omp target map(tofrom: num_threads,x)  
#pragma omp parallel num_threads(OMPVV_NUM_THREADS_DEVICE) default(shared)
  {
    if (omp_get_thread_num() == 0) {
      num_threads = omp_get_num_threads();
    }
#pragma omp atomic hint(0X4) // corrosponds to omp_sync_hint_nonspeculative
    x++;
  }

  OMPVV_ERROR_IF(num_threads < 0, "Test ran with invalid number of threads (less than zero)");
  OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so the results are not conclusive");

  OMPVV_TEST_AND_SET_VERBOSE(errors, x != num_threads);

  return errors;
}

int test_atomic_with_unused_enum_value() {
  OMPVV_INFOMSG("test_atomic_with_unused_enum_value");
  int errors = 0, x = 0, num_threads = -1;

#pragma omp target map(tofrom: num_threads,x)  
#pragma omp parallel num_threads(OMPVV_NUM_THREADS_DEVICE) default(shared)
  {
    if (omp_get_thread_num() == 0) {
      num_threads = omp_get_num_threads();
    }
#pragma omp atomic hint(0X1024) //As of OMP Spec 5.0 only values till 0x8 have been taken
    x++;
  }

  OMPVV_ERROR_IF(num_threads < 0, "Test ran with invalid number of threads (less than zero)");
  OMPVV_WARNING_IF(num_threads == 1, "Test ran with one thread, so the results are not conclusive");

  OMPVV_TEST_AND_SET_VERBOSE(errors, x != num_threads);

  return errors;
}


int main() {
  int errors = 0;

  OMPVV_TEST_OFFLOADING;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_atomic_with_used_enum_value());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_atomic_with_unused_enum_value());

  OMPVV_REPORT_AND_RETURN(errors);
}
