//===---- test_target_enter_exit_data_struct.c ----------------------------------===//
// 
// OpenMP API Version 4.5 Nov 2015
//
// This test checks functionality of target enter data and target exit data using
// a struct and a typedef that each have multiple data memebers of different type 
// and length. The functions test_typedef() and test_struct() are both copying 
// structures to the device, altering public member values, copying them back
// and checking to see that values are properly updated. 
//
////===--------------------------------------------------------------------------===//

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "ompvv.h"

#define N 1000

// Test for OpenMP 4.5 target enter and target exit using a struct.
int test_struct() {
   
  OMPVV_INFOMSG("Running test_struct()");
    
  int errors = 0;
  int* pointers[6];

  struct {
    int a; // firstprivate
    int b[N]; // tofrom:b[0:N]
    int *p; // tofrom:p[0:0]
  } single, array[5];

  single.p = (int*) malloc(5 * sizeof(int));
  pointers[0] = single.p;

  for (int i = 0; i < 5; ++i) {
    array[i].p = (int*) malloc(5 * sizeof(int));
    pointers[i + 1] = array[i].p;
  }

#pragma omp target enter data map(to: single) map(to: array[0:5])

#pragma omp target map(alloc: single) map(alloc: array[0:5])
{
  single.a = 1;
  for (int i = 0; i < N; ++i)
    single.b[i] = 1;
      
  for (int i = 0; i < 5; ++i) {
    array[i].a = 1;
    for (int j = 0; j < N; ++j)
      array[i].b[j] = 1;
  }
} //end target map

#pragma omp target exit data map(from: single) map(from: array[0:5])

  // Checking results
  OMPVV_TEST_AND_SET_VERBOSE(errors, (single.a != 1)); 
  for (int i = 0; i < N; ++i)
    OMPVV_TEST_AND_SET_VERBOSE(errors, (single.b[i] != 1));
  OMPVV_TEST_AND_SET_VERBOSE(errors, (pointers[0] != single.p));
  for (int i = 0; i < 5; ++i) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, (array[i].a != 1)); 
    for (int j = 0; j < N; ++j)
      OMPVV_TEST_AND_SET_VERBOSE(errors, (array[i].b[j] != 1));
    OMPVV_TEST_AND_SET_VERBOSE(errors, (pointers[i + 1] != array[i].p));
  }

  free(single.p);
  for (int i = 0; i < 5; ++i) {
    free(array[i].p);
  }
  
  return errors;
}

// Test for OpenMP 4.5 target enter and target exit using a typedef struct.
int test_typedef() {
    
  OMPVV_INFOMSG("Running test_typedef()");
    
  int errors = 0;
  int* pointers[6];

  typedef struct {
    int a;
    int b[N];
    int *p;
  } test_struct;
 
  test_struct single, array[5];

  single.p = (int*) malloc(5 * sizeof(int));
  pointers[0] = single.p;

  for (int i = 0; i < 5; ++i) {
    array[i].p = (int*) malloc(5 * sizeof(int));
    pointers[i + 1] = array[i].p;
  }

#pragma omp target enter data map(to: single) map(to: array[0:5])

#pragma omp target map(alloc: single) map(alloc: array[0:5])
{
  single.a = 1;
  for (int i = 0; i < N; ++i)
    single.b[i] = 1;
      
  for (int i = 0; i < 5; ++i) {
    array[i].a = 1;
    for (int j = 0; j < N; ++j)
      array[i].b[j] = 1;
  }
} //end target map

#pragma omp target exit data map(from: single) map(from: array[0:5])

  // Checking results
  OMPVV_TEST_AND_SET_VERBOSE(errors, (single.a != 1)); 
  for (int i = 0; i < N; ++i)
    OMPVV_TEST_AND_SET_VERBOSE(errors, (single.b[i] != 1));
  errors |= (pointers[0] != single.p);
  for (int i = 0; i < 5; ++i) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, (array[i].a != 1)); 
    for (int j = 0; j < N; ++j)
      OMPVV_TEST_AND_SET_VERBOSE(errors, (array[i].b[j] != 1));
    OMPVV_TEST_AND_SET_VERBOSE(errors, (pointers[i + 1] != array[i].p));
  }
  
  free(single.p);
  for (int i = 0; i < 5; ++i) {
    free(array[i].p);
  }
   return errors;
}

int main () {
  
  // Check that offloading is enabled
  int isOffloading;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  
  int errors = 0;
  errors += test_struct();
  errors += test_typedef();
  OMPVV_REPORT_AND_RETURN(errors);
}
