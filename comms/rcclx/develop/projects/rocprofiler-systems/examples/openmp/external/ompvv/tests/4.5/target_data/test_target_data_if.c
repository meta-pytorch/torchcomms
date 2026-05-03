//===---- test_target_data_if.c - check the if clause of target data ------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// The if clause determines if the section should be executed in the host or
// the device. There are three things to test here:
// (a) with offloading when 'if' clause evaluates to true then code
// be executed on the device
// (b) with offloading when 'if' clause evaluates to false then code should
// be executed on the host
// (c) without offloading all the code should be executed on the device
// The if clause is evaluated on runtime which means that variables could
// determine this behavior. We use a SIZE_THRESHOLD variable to check if we
// should execute on the device or the host. Before starting the test we
// sample offloading to see if it was enabled or not. If the code is executed
// in the device, the result should be c[i] = a[i] + b[i] = i + 1.
// If the code is executed on the host the result should be c[i] = -1
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define SIZE 1024
#define SIZE_THRESHOLD 512

int test_target_data_map_if_nested (int isOffloading){
  int a[SIZE];
  int b[SIZE];
  int c[SIZE];
  int map_size, i = 0, errors[2] = {0,0}, isHost = -1;

  // a and b array initialization
  for (i = 0; i < SIZE; i++) {
    a[i] = 1;
    b[i] = i;
  }

  // check multiple sizes.
  for (map_size = 256; map_size <= SIZE; map_size += 256) {
    // C initialization
    for (i = 0; i < map_size; i++) {
      c[i] = -1;
    }
#pragma omp target data if(map_size > SIZE_THRESHOLD) map(to: map_size)  \
        map(tofrom: c[0:map_size])                                       \
        map(to: a[0:map_size], b[0:map_size])
    {
#pragma omp target if(map_size > SIZE_THRESHOLD) map(tofrom: isHost) \
        map (alloc: a[0:map_size], b[0:map_size], c[0:map_size]) // avoid default mapping
      {
        isHost = omp_is_initial_device();
        int alpha = (isHost ? 0 : 1);
        int j = 0;
        for (j = 0; j < map_size; j++) {
          // c[j] is zero if executed in the host
          // c[j] is 1+j if executed on the device
          c[j] = alpha*(a[j] + b[j]);
        }
      } // end target
    }//end-target data

    // checking results
    for (i = 0; i < map_size; i++) {
      if (isOffloading && map_size > SIZE_THRESHOLD) {
        // Should have executed on the device
        // if offloading was used
        // c[i] is zero if it was executed in the host
        OMPVV_TEST_AND_SET(errors[0], (c[i] != i + 1)); //error when executed on the device
      } else {
        // Should have executed in the host
        // with or without offloading
        OMPVV_TEST_AND_SET(errors[1], (c[i] != 0));
      } //end-else
    }
  } // end-for map_size

  if (!errors[0] && !errors[1]) {
    OMPVV_INFOMSG("Test nested if passed with offloading %s", (isOffloading ? "enabled" : "disabled"));
  } else if (errors[0]==0 && errors[1]!=0) {
    OMPVV_ERROR("Test nested if failed on host with offloading %s.", (isOffloading ? "enabled" : "disabled"));
  } else if (errors[0]!=0 && errors[1]==0) {
    OMPVV_ERROR("Test nested if failed on device with offloading %s.", (isOffloading ? "enabled" : "disabled"));
  } else if (errors[0]!=0 && errors[1]!=0) {
    OMPVV_ERROR("Test nested if failed on host and device with offloading %s.", (isOffloading ? "enabled" : "disabled"));
  }

  return errors[0] + errors[1];
}

int test_target_data_map_if_simple(int isOffloading){
  int a[SIZE];
  int b[SIZE];
  int c[SIZE];
  int map_size, i = 0, errors[3] = {0,0,0}, isHost = -1;

  // check multiple sizes.
  for (map_size = 256; map_size <= SIZE; map_size += 256) {
    // a, b, and c array initialization
    for (i = 0; i < SIZE; i++) {
      a[i] = SIZE - i;
      b[i] = i;
      c[i] = -1;
    }
#pragma omp target data if(map_size > SIZE_THRESHOLD) map(to: map_size)  \
        map(tofrom: c[0:map_size])                             \
        map(to: a[0:map_size], b[0:map_size])
    {
#pragma omp target map(tofrom: isHost) \
        map (alloc: a[0:map_size], b[0:map_size], c[0:map_size]) // avoid default mapping
      {
        isHost = omp_is_initial_device();
        int j = 0;
        for (j = 0; j < map_size; j++) {
          // This should be equal to SIZE, if target data
          // mapped the arrays a and b, otherwise it is
          // unknown but it is not used either
          c[j] += (a[j] + b[j] + 1);
          a[j] = -1; // changing memory content this should not affect original storage
          b[j] = -1; // changing memory content
        }
      } // end target

      // Check that the target region did not exec in the
      // host as the if should only affect the target data
      if (isOffloading) {
        OMPVV_TEST_AND_SET_VERBOSE(errors[0], isHost);
      }
    }//end-target data

    // checking results
    for (i = 0; i < map_size; i++) {
      if (map_size > SIZE_THRESHOLD || !isOffloading) {
        // Should have executed in the device if offloading is enabled
        // If offloading is not enabled or the system is a shared env
        // between device and host, then the value will modify the original
        // elements of c.
        OMPVV_TEST_AND_SET(errors[1], (c[i] != SIZE)); //error when executed on the device
      } else {
        // Should have executed in the host
        // with or without offloading
        OMPVV_TEST_AND_SET(errors[2], (c[i] != -1));
      } //end-else
    }
  } // end-for map_size

  if (errors[0]) {
    OMPVV_ERROR("Test did not offload to the device. 'If' clause might be affecting the target"
                " offlading as well and it should not ")
  }
  if (!errors[0] && !errors[1] && !errors[2]) {
    OMPVV_INFOMSG("Test passed with offloading %s", (isOffloading ? "enabled" : "disabled"));
  } else if (errors[1]==0 && errors[2]!=0) {
    OMPVV_ERROR("Test failed for if (false) with offloading %s.", (isOffloading ? "enabled" : "disabled"));
  } else if (errors[1]!=0 && errors[2]==0) {
    OMPVV_ERROR("Test failed for if (true) with offloading %s.", (isOffloading ? "enabled" : "disabled"));
  } else if (errors[1]!=0 && errors[2]!=0) {
    OMPVV_ERROR("Test failed for if(true) and if(false) with offloading %s.", (isOffloading ? "enabled" : "disabled"));
  }

  return errors[0] + errors[1] + errors[2];
}

// Test for OpenMP 4.5 target data with if
int main() {
  int isOffloading = 0;
  int errors = 0;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  OMPVV_WARNING_IF(!isOffloading, "Offloading is off, tests will be inconclusive. No way to tests if");

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_data_map_if_nested(isOffloading) != 0);
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_target_data_map_if_simple(isOffloading) != 0);

  OMPVV_REPORT_AND_RETURN(errors);
}
