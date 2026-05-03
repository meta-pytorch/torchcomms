//===---- test_declare_target_enter_device_type_host.c  -----------------------------===//
// 
// OpenMP API Version 5.2 
//
// The declare target directive specifies that variables, functions(C,C++ and Fortran),
// and subroutines (Fortran) are mapped to a device. If device_type(host) is present 
// only a host version of the procedure is made available. The "enter" clause was introduced 
// to replace the to clause in 5.2.
// 
//===-------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024
int errors = 0;

int a[N], b[N], c[N];  
int i = 0;

void update() { 
  for (i = 0; i < N; i++) {
    a[i] += 1;
    b[i] += 2;
    c[i] += 3;
  }
}

#pragma omp declare target enter(a,b,c,i) 
#pragma omp declare target enter(update) device_type(host) 

int test_declare_target_device_type_host() { 

  #pragma omp target update to(a,b,c)
  
  #pragma omp target   
  {
    for (i = 0; i < N; i++) {
      a[i] += i;
      b[i] += 2 * i;
      c[i] += 3 * i;
    }
  }
  #pragma omp target update from( a, b, c)

  //on host
  update();

  for (i = 0; i < N; i++) { //check array values on host
    if ( a[i] != i + 1 || b[i] != 2 * i + 2 || c[i] != 3 * i + 3 ) {
      errors++;
    }
   
  }
  return errors;
}

int main () {
  
  //initalize arrays on host
  for (i = 0; i < N; i++) {
    a[i] = 0;
    b[i] = 0;
    c[i] = 0;
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_declare_target_device_type_host());
  OMPVV_REPORT_AND_RETURN(errors);
}  
