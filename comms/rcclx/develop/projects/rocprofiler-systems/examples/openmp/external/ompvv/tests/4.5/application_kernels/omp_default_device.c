//===----- omp_default_device.c ----------------------------------------------===//
// 
// OpenMP API Version 4.5 Nov 2015
//
// This test is based on a bug encounted by an application where the default device 
// would change (after explicit omp_set_default_device) after allocation via 
// omp_target_alloc).The test sets and gets the default device number and checks if 
// the default device has actually changed after a memory allocation was made on 
// the device.
//
////===-------------------------------------------------------------------------------===//

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1000

int test_omp_device() {
  OMPVV_INFOMSG("test_get_set_default_device");

  int errors = 0, setDev, iDev1, iDev2;
  double *buf1, *buf2;
  int num_devices = omp_get_num_devices();
  
  OMPVV_TEST_AND_SET(errors, num_devices <= 0);

  if(errors)
   return errors;

  // Set default device
  if(num_devices > 1){
    omp_set_default_device(num_devices-1);
    setDev = num_devices-1;
  }
  else{
    omp_set_default_device(0);
    setDev = 0;
  }

  iDev1 = omp_get_default_device();
  OMPVV_TEST_AND_SET(errors, iDev1 != setDev);


  buf1 = (double *)omp_target_alloc (sizeof(double)* N, iDev1);
  iDev2 = omp_get_default_device();
  OMPVV_TEST_AND_SET(errors, iDev2 != iDev1);

  omp_target_free (buf1, iDev1);

  return errors;
}

int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_omp_device());

  OMPVV_REPORT_AND_RETURN(errors);
}

