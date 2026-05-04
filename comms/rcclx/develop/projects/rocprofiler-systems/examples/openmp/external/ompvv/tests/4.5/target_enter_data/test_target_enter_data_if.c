//===---- test_target_enter_data_if.c - check the if clause of target data ------===//
// 
// OpenMP API Version 4.5 Nov 2015
// 
// This test verifies the conditional data movement of the target enter data
// directive through the if clause. There are two options:
// 1. The if condition evaluates to true, in which case the data a[i] = 1 and 
// b[i] = i will be copied over to the device. Since there is alreay a and b
// mapped in the target enter data, then the target region should not map the 
// a and b arrays. 
//
// 2. The if condition evaluates to false. After the target enter data directive
// we modify the values of the host to be 0. In which case the target region will
// be in charge of the data movement and a[i] = 0 and b[i] = 0
//
// The target region will do c[i] = a[i] which will be either i + 1 or 0, 
// depending on the result of if. 
//
// this test only gets executed if there is offloading and no shared memory
// Shared memory devices would not work in this case
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include "ompvv.h"
#include <stdio.h>

#define SIZE_THRESHOLD 512
#define ARRAY_SIZE 1024

// Test for OpenMP 4.5 target enter data with if
int main() {
  int a[ARRAY_SIZE];
  int b[ARRAY_SIZE];
  int c[ARRAY_SIZE];
  int size, i = 0, errors = 0, isOffloading = 0, isSharedMemory = 0;

  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)
  OMPVV_TEST_AND_SET_SHARED_ENVIRONMENT(isSharedMemory)

  if (!isOffloading || isSharedMemory) {
    OMPVV_WARNING("It is not possible to test conditional data transfers "
                  "if the environment is shared or offloading is off. Not testing "
                  "anything")
    OMPVV_REPORT_AND_RETURN(0);
  }

  // check multiple sizes. 
  for (size = 256; size <= ARRAY_SIZE; size += 256) {
    // a,b and c arrays initialization
    for (i = 0; i < size; i++) {
      a[i] = 1;
      b[i] = i;
      c[i] = -1;
    }
#pragma omp target enter data if(size > SIZE_THRESHOLD) map(to: size) \
    map (to: a[0:size], b[0:size])
           
    // a, b arrays host side modification
    for (i = 0; i < size; i++) {
      a[i] = 0;
      b[i] = 0;
    }

    // if a and b were mapped already by the target enter data then 
    // the tofrom should be a noop
#pragma omp target map(tofrom: a[0:size], b[0:size], c[0:size])
{
        int j = 0;
        for (j = 0; j < size; j++) {
          // c[j] is zero if executed in the host
          // c[j] is 1+j if executed on the device
          c[j] = a[j] + b[j];
        }
} // end target

    // checking results 
    for (i = 0; i < size; i++) {
      if (size > SIZE_THRESHOLD) {
        OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != i + 1)
      } else {
        OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != 0)
      } //end-else 
    }
    // This is not part of the test but it is necessary to avoid conflicts
    #pragma omp target exit data if(size > SIZE_THRESHOLD) map(delete: a[0:size], b[0:size])
  } // end-for size

  OMPVV_REPORT_AND_RETURN(errors)

}
