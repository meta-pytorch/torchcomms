//===-- test_target_data_use_device_addr.c --------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This file is a test for the use_device_addr when used with the map
// clause with target data directive. This test uses a scalar and an array
// of size N which values are modified on the  device and tested in the
// host. List items that appear in a use_device_addr clause have the address
// of the corresponding object in the device data environment inside the
// construct. This test also tests that address conversions of
// use_device_addr clauses will occur as if performed after all variables
// are mapped according to those map clauses.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"


int main() {
  int errors = 0;
  int device_data = 14, host_data=0;

  OMPVV_TEST_OFFLOADING;

#pragma omp target data map(to: device_data)
  {
    int *dev_ptr;
#pragma omp target data use_device_addr(device_data)
    {
      dev_ptr = &device_data;
    }
#pragma omp target map(to:device_data) map(tofrom: errors) map(from: host_data) is_device_ptr(dev_ptr)
    {
      if(&device_data != dev_ptr) {
        errors++;
      }
    } // end target

#pragma omp target map(from: host_data) is_device_ptr(dev_ptr)
    {
      host_data = *dev_ptr;
    }

  } // end target data


  // checking results
  OMPVV_TEST_AND_SET(errors, host_data != 14);

  OMPVV_REPORT_AND_RETURN(errors);
}
