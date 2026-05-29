//===---- test_declare_target_device_type_nohost.c  ----------------------------------===//
// 
// OpenMP API Version 5.0 
//
// The declare target directive specifies that variables, functions(C,C++ and Fortran),
// and subroutines (Fortran) are mapped to a device. If a device_type
// clause is present on the contained declare target directive, then its argument 
// determines which versions are made available. If device_type(nohost) is present 
// only a device version of the procedure is made available. 
// The device version of the function is specified via declare variant.
// 
//===-------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 10
int errors = 0;

void target_function();
#pragma omp declare target to(target_function) device_type(nohost)

#pragma omp declare target
int a[N], b[N], c[N];  
int i = 0;
#pragma omp end declare target

#pragma omp declare variant(target_function) match(device={kind(nohost)})
void update() {
  for (i = 0; i < N; i++) {
    a[i] += 5;
    b[i] += 5;
    c[i] += 5;
  }
}

void target_function(){
  for (i = 0; i < N; i++) {
    a[i] += 1;
    b[i] += 2;
    c[i] += 3;
  }
} 

int test_declare_target_device_type_nohost() { 

  update();

  #pragma omp target update to(a,b,c)

  #pragma omp target  
  {
    update();
  }

  #pragma omp target update from (a,b,c)
  
  if (omp_get_default_device () >= 0 && omp_get_default_device () < omp_get_num_devices ()) {
    for (i = 0; i < N; i++) { //check array values on host
      if ( a[i] != 6 || b[i] != 7 || c[i] != 8 ) {
        errors++;
      }
    }
  } else {
    OMPVV_WARNING("Default device is the host device. Thus, test only ran on the host");
    for (i = 0; i < N; i++) { //check array values on host
      if ( a[i] != 10 || b[i] != 10 || c[i] != 10 ) {
        errors++;
      }
    }
  }
  
  return errors;
}

int main () {

  OMPVV_TEST_OFFLOADING;
  
  //initalize arrays on host
  for (i = 0; i < N; i++) {
    a[i] = 0;
    b[i] = 0;
    c[i] = 0;
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_declare_target_device_type_nohost());
  OMPVV_REPORT_AND_RETURN(errors);
}  
