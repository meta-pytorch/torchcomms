//===--- test_target_enter_data_devices.c ---------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
// 
// This file tests the target enter data directive when the device clause is
// specified, and also when the device is set by omp_set_default_device().
// The first function test_set_default_dev() does not specify the device
// clause, but instead uses omp_set_default_device() to set device.
// The second function test_device() uses the device clause on the target
// enter data directive.
// 
//===----------------------------------------------------------------------===//

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
  int sum[num_dev+1], errors = 0, isHost[num_dev+1];
  int h_matrix[num_dev+1][N], h_matrix_copy[num_dev+1][N];

  // Initialize all the matrices
  for (int dev = 0; dev < num_dev; ++dev) {
    sum[dev] = 0;
    isHost[dev] = 0;
  }

  for (int dev = 0; dev < num_dev; ++dev) {
    omp_set_default_device(dev);
    
    // unstructured mapping
    {
#pragma omp target enter data map(alloc: h_matrix[dev][0:N]) // omp_target_alloc sets ref to infinity. alloc: has effect only if ref is zero (page 217 line 21 - Version 4.5 November 2015)
      printf(""); // forcing the compiler to not moving out of the scope
    }
#pragma omp target map(alloc: h_matrix[dev][0:N]) map(tofrom: isHost[dev:1]) // map(alloc: ) to avoid target to map the entire matrix h_matrix[dev][:]
    {
      isHost[dev] = omp_is_initial_device();
      for (int i = 0; i < N; ++i) {
        h_matrix[dev][i] = dev;
      }
    }
  // Since we don't do enter exit data we copy the values from the device
#pragma omp target map(from: h_matrix_copy[dev][0:N]) map(alloc: h_matrix[dev][0:N])
    {
      for (int i = 0; i < N; ++i) {
        h_matrix_copy[dev][i] = h_matrix[dev][i];
      }
    }
}

  // checking results
  for (int dev = 0; dev < num_dev; ++dev) {
    OMPVV_INFOMSG("device %d ran on the %s", dev, (isHost[dev])? "host" : "device");
    for (int i = 0; i < N; ++i)
      sum[dev] += h_matrix_copy[dev][i];
    OMPVV_TEST_AND_SET(errors, (dev * N != sum[dev]));
  }

  omp_set_default_device(def_dev);

  // Avoiding memory leaks this is outside of testing area
  // Iterate over all the devices and delete the memory
  for (int dev = 0; dev < num_dev; ++dev) {
#pragma omp target exit data map(delete: h_matrix[dev][0:N]) device(dev)
  }

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
  int sum[num_dev+1], errors = 0, isHost[num_dev+1];
  int h_matrix[num_dev+1][N], h_matrix_copy[num_dev+1][N];

  // Initialize all the matrices
  for (int dev = 0; dev < num_dev; ++dev) {
    sum[dev] = 0;
    isHost[dev] = 0;
  }
  for (int dev = 0; dev < num_dev; ++dev) {
    // unstructured mapping
    {
#pragma omp target enter data map(alloc: h_matrix[dev][0:N]) device(dev)
        printf("");
    }
    // operation
#pragma omp target map(alloc: h_matrix[dev][0:N]) map(tofrom: isHost[dev:1]) device(dev) // map(alloc: ) to avoid target to map the entire matrix h_matrix[dev][:]
    {
      isHost[dev] = omp_is_initial_device();
      for (int i = 0; i < N; ++i)
        h_matrix[dev][i] = dev;
    }
  // Since we don't do enter exit data we copy the values from the device
#pragma omp target map(from: h_matrix_copy[dev][0:N])  map(alloc: h_matrix[dev][0:N]) device(dev)
    {
      for (int i = 0; i < N; ++i) {
        h_matrix_copy[dev][i] = h_matrix[dev][i];
      }
    }
  }

  // checking results
  for (int dev = 0; dev < num_dev; ++dev) {
    OMPVV_INFOMSG("device %d ran on the %s", dev, (isHost[dev])? "host" : "device");
    for (int i = 0; i < N; ++i)
      sum[dev] += h_matrix_copy[dev][i];
    OMPVV_TEST_AND_SET(errors, (dev * N != sum[dev]));
  }

  // Avoiding memory leaks
  // Iterate over all the devices and delete the memory
  for (int dev = 0; dev < num_dev; ++dev) {
#pragma omp target exit data map(delete: h_matrix[dev][0:N]) device(dev)
  }

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET(errors, test_set_default_dev());
  OMPVV_TEST_AND_SET(errors, test_device());

  OMPVV_REPORT_AND_RETURN(errors);
}
