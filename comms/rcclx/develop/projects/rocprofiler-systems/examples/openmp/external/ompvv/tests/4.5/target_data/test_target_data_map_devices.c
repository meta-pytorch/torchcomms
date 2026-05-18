//===--- test_target_data_map_devices.c--- target data map to multiple dev--===//
//
// OpenMP API Version 4.5 Nov 2015
//
//  This test checks for data mapping on multiple devices when using the target
//  data directive. It makes sure that data mapping is happening on each device
//  through the use of omp_set_default_device as well as the device() clause
//
//  Since OpenMP 4.5 does not have an API call to obtain the current device, 
//  this test does not guarantee that the execution devices are different. 
//  
//  Last modified by Jose M Monsalve Diaz, December 25, 2019
//
////===----------------------------------------------------------------------===//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1000

// Test for OpenMP 4.5 target data to multiple devices using API
int test_map_set_default_dev() {
  OMPVV_INFOMSG("test_set_default_device");

  // Get number of devices
  int num_dev = omp_get_num_devices();
  OMPVV_INFOMSG("num_devices: %d", num_dev);

  int def_dev = omp_get_default_device();
  OMPVV_INFOMSG("initial device: %d", omp_get_initial_device());
  OMPVV_INFOMSG("default device: %d", def_dev);

  // Allocate num_devices + 1 to avoid zero-sized VLA if num_devices == 0
  int sum[num_dev+1], errors = 0;
  int* h_matrix = (int*) malloc(num_dev * N * sizeof(int));

  for (int dev = 0; dev < num_dev; ++dev) {
    omp_set_default_device(dev);
#pragma omp target data map(from: h_matrix[dev*N:N])
    {
      OMPVV_TEST_AND_SET_VERBOSE(errors, dev != omp_get_default_device());
#pragma omp target map(alloc: h_matrix[dev*N:N])
      {
        for (int i = 0; i < N; ++i)
          h_matrix[dev*N + i] = dev;
      } // end target
    } // end target data
  }

  // checking results 
  for (int dev = 0; dev < num_dev; ++dev) {
    sum[dev] = h_matrix[dev*N + 0];
    for (int i = 1; i < N; ++i)
      sum[dev] += h_matrix[dev*N + i];
    OMPVV_TEST_AND_SET_VERBOSE(errors, (dev * N != sum[dev]));
  }

  omp_set_default_device(def_dev);
  free(h_matrix);
  return errors;
}

// Test for OpenMP 4.5 target data to multiple devices using directives
int test_map_device() {

  OMPVV_INFOMSG("test_map_device");

  // Get number of devices 
  int num_dev = omp_get_num_devices();
  OMPVV_INFOMSG("num_devices: %d", num_dev);

  OMPVV_INFOMSG("initial device: %d", omp_get_initial_device());
  OMPVV_INFOMSG("default device: %d", omp_get_default_device());

  // Allocate num_devices + 1 to avoid zero-sized VLA if num_devices == 0
  int sum[num_dev+1], errors = 0;
  int* h_matrix = (int*) malloc(num_dev * N * sizeof(int));

  for (int dev = 0; dev < num_dev; ++dev) {
#pragma omp target data map(from: h_matrix[dev*N:N]) device(dev)
    {
#pragma omp target map(alloc: h_matrix[dev*N:N]) device(dev)
      {
        for (int i = 0; i < N; ++i)
          h_matrix[dev*N + i] = dev;
      } // end target
    } // end target data
  }

  // checking results 
  errors = 0;
  for (int dev = 0; dev < num_dev; ++dev) {
    sum[dev] = h_matrix[dev*N + 0];
    for (int i = 1; i < N; ++i)
      sum[dev] += h_matrix[dev*N + i];
    OMPVV_TEST_AND_SET_VERBOSE(errors, (dev * N != sum[dev]));
  }

  free(h_matrix);
  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_set_default_dev());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_map_device());

  OMPVV_REPORT_AND_RETURN(errors);
}
