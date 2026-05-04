//===--- test_target_update_devices.c--------------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test checks if the target update directive works on different devices.
// We check two different variants.
// 1. setting up the default device with the API call omp_set_default_device()
// 2. using the device clause of the target update directive.
//
// Testing metodology uses an array that gets mapped into the device at first
// through target enter data. Then on each iteration we update the array in one
// device, create a compute region in that device, and then update it back
// We also record that the compute region is not executed in the host
// with the omp_is_initial_device() API call. Unfortunately 4.5 has no device
// number API call.
//
////===----------------------------------------------------------------------===//



#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1000

/*
 * This test check if update device works well
 * when the omp_set_default_device API call is used
 * to change the default device
 */
int test_set_default_dev() {
  OMPVV_INFOMSG("test_set_default_dev");

  // Get number of devices
  int num_dev = omp_get_num_devices();
  OMPVV_INFOMSG("num_devices: %d", num_dev);

  int def_dev = omp_get_default_device();
  OMPVV_INFOMSG("initial device: %d", omp_get_initial_device());
  OMPVV_INFOMSG("default device: %d", def_dev);

  int sum = 0;
  int errors = 0;
  int isHost[num_dev];
  int h_matrix[N];

  // Mapping the array to all the devices
  for (int dev = 0; dev < num_dev; ++dev) {
    omp_set_default_device(dev);
    // unstructured mapping
    {
#pragma omp target enter data map(alloc: h_matrix[0:N])
        printf(""); // forcing the compiler to not moving out of the scope
    }
  }

  // Initialize the array
  for (int i = 0; i < N; ++i) {
    h_matrix[i] = 0;
  }

  // Each device gets updated with the current array version,
  // one gets added to each element in the array, and then
  // the host gets the updated version
  for (int dev = 0; dev < num_dev; ++dev) {
    omp_set_default_device(dev);
#pragma omp target update to(h_matrix[0:N])
    // operation
#pragma omp target map(alloc: h_matrix[0:N]) map(tofrom: isHost[dev:1]) // map(alloc: ) to avoid target to map the entire matrix h_matrix[:]
    {
      // Check on device or host execution
      isHost[dev] = omp_is_initial_device();

      for (int i = 0; i < N; ++i) {
        h_matrix[i]++;
      }
    }
#pragma omp target update from(h_matrix[0:N])
    // unstructured exit
  }

  // Unmap the matrix
  for (int dev = 0; dev < num_dev; ++dev) {
    omp_set_default_device(dev);
#pragma omp target exit data map(delete: h_matrix[0:N])
    printf("");
  }

  // checking results
  for (int dev = 0; dev < num_dev; ++dev) {
    OMPVV_INFOMSG("device %d ran on the %s", dev, (isHost[dev])? "host" : "device");
  }
  for (int i = 0; i < N; ++i) {
    sum += h_matrix[i];
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, (num_dev * N != sum));

  omp_set_default_device(def_dev);

  return errors;
}

/*
 * This test checks if using the device clause in
 * data update works fine
 */
int test_device() {
  OMPVV_INFOMSG("test_device_clause");

  // Get number of devices
  int num_dev = omp_get_num_devices();
  OMPVV_INFOMSG("num_devices: %d", num_dev);

  int def_dev = omp_get_default_device();
  OMPVV_INFOMSG("initial device: %d", omp_get_initial_device());
  OMPVV_INFOMSG("default device: %d", def_dev);

  int sum = 0;
  int errors = 0;
  int isHost[num_dev];
  int h_matrix[N];

  // Mapping the array to all the devices
  for (int dev = 0; dev < num_dev; ++dev) {
    // unstructured mapping
    {
#pragma omp target enter data map(alloc: h_matrix[0:N]) device(dev)
        printf(""); // forcing the compiler to not moving out of the scope
    }
  }

  // Initialize the array
  for (int i = 0; i < N; ++i) {
    h_matrix[i] = 0;
  }

  // Each device gets updated with the current array version,
  // one gets added to each element in the array, and then
  // the host gets the updated version
  for (int dev = 0; dev < num_dev; ++dev) {
#pragma omp target update to(h_matrix[0:N]) device(dev)
    // operation
#pragma omp target map(alloc: h_matrix[0:N]) map(tofrom: isHost[dev:1]) \
    device(dev)// map(alloc: ) to avoid target to map the entire matrix h_matrix[:]
    {
      // Check on device or host execution
      isHost[dev] = omp_is_initial_device();

      for (int i = 0; i < N; ++i) {
        h_matrix[i]++;
      }
    }
#pragma omp target update from(h_matrix[0:N]) device(dev)
    // unstructured exit
  }

  // Unmap the matrix
  for (int dev = 0; dev < num_dev; ++dev) {
#pragma omp target exit data map(delete: h_matrix[0:N]) device(dev)
    printf("");
  }

  // checking results
  for (int dev = 0; dev < num_dev; ++dev) {
    OMPVV_INFOMSG("device %d ran on the %s", dev, (isHost[dev])? "host" : "device");
  }
  for (int i = 0; i < N; ++i) {
    sum += h_matrix[i];
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, (num_dev * N != sum));


  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET(errors, test_set_default_dev());
  OMPVV_TEST_AND_SET(errors, test_device());

  OMPVV_REPORT_AND_RETURN(errors);
}
