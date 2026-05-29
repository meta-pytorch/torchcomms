//===--- test_target_enter_exit_data_devices.c---------------------------------------===//
//
//  OpenMP API Version 4.5 Nov 2015
//
//  This test checks for data mapping on multiple devices when using the target
//  data map directive. The test uses omp_get_num_devices to confirm the number 
//  of total devices available. There are two separate functions, and both make sure 
//  that data mapping is happening on each available device. The first function does 
//  this through the use of omp_set_default_device, while the second function explicity 
//  uses the device() clause to control which device is being utilized.
//
//  Since OpenMP 4.5 does not have an API call to obtain the current device, 
//  this test does not guarantee that the execution devices are different. 
//
////===------------------------------------------------------------------------------===/

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1000

int test_set_default_dev() {

  OMPVV_INFOMSG("test_set_default_dev");

  // Get number of devices
  int num_dev = omp_get_num_devices();
  OMPVV_INFOMSG("num_devices: %d", num_dev);

  int def_dev = omp_get_default_device();
  OMPVV_INFOMSG("initial device: %d", omp_get_initial_device());
  OMPVV_INFOMSG("default device: %d", def_dev);

  // Allocate num_devices + 1 to avoid zero-sized VLA if num_devices == 0
  int sum[num_dev+1], errors = 0;
  int h_matrix[num_dev+1][N];

  for (int dev = 0; dev < num_dev; ++dev) {
    omp_set_default_device(dev);

#pragma omp target enter data map(alloc: h_matrix[dev][0 : N])
    
#pragma omp target map(alloc: h_matrix[dev][0 : N]) // map(alloc: ) to avoid target to map the entire matrix h_matrix[dev][:]
    {
      for (int i = 0; i < N; ++i)
        h_matrix[dev][i] = dev;
    }
    // unstructured exit

#pragma omp target exit data map(from: h_matrix[dev][0 : N])
  }

  // checking results
  errors = 0;
  for (int dev = 0; dev < num_dev; ++dev) {
    sum[dev] = h_matrix[dev][0];
    for (int i = 1; i < N; ++i)
      sum[dev] += h_matrix[dev][i];
    OMPVV_TEST_AND_SET_VERBOSE(errors, (dev * N != sum[dev]));
  }

  omp_set_default_device(def_dev);

  return errors;
}

int test_device() {

  OMPVV_INFOMSG("test_device");

  // Get number of devices
  int num_dev = omp_get_num_devices();
  OMPVV_INFOMSG("num_devices: %d", num_dev);

  OMPVV_INFOMSG("initial device: %d", omp_get_initial_device());
  OMPVV_INFOMSG("default device: %d", omp_get_default_device());

  // Allocate num_devices + 1 to avoid zero-sized VLA if num_devices == 0
  int sum[num_dev+1], errors = 0;
  int h_matrix[num_dev+1][N];

  for (int dev = 0; dev < num_dev; ++dev) {

#pragma omp target enter data map(alloc: h_matrix[dev][0 : N]) device(dev) 

#pragma omp target map(alloc: h_matrix[dev][0 : N]) device(dev)
    {
      for (int i = 0; i < N; ++i)
        h_matrix[dev][i] = dev;
    }

#pragma omp target exit data map(from: h_matrix[dev][0 : N]) device(dev)
  
}

  // checking results
  errors = 0;
  for (int dev = 0; dev < num_dev; ++dev) {
    sum[dev] = h_matrix[dev][0];
    for (int i = 1; i < N; ++i)
      sum[dev] += h_matrix[dev][i];
    OMPVV_TEST_AND_SET_VERBOSE(errors, (dev * N != sum[dev]));
  }

  return errors;
}

int main() {

  int errors = 0;
  
  //Test for offloading
  int is_offloading;
  OMPVV_TEST_AND_SET_OFFLOADING(is_offloading);

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_set_default_dev());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_device());

  OMPVV_REPORT_AND_RETURN(errors);
}
