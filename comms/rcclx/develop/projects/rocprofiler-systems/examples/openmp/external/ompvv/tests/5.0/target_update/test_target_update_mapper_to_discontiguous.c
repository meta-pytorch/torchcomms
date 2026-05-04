//===--- test_target_update_mapper_to_discontiguous.c -----------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test seeks to ensure that target update with motion-clause "to" can properly
// map data to the device by specifying a user-defined mapper. Additionally, the test
// checks a new addition to target update in OpenMP 5.0 that states "List items in the
// to or from clauses may include array sections with stride expressions." 
//
// Adopted from OpenMP 5.0 Example target_mapper.1.c
//===------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

typedef struct newvec {
  size_t len;
  double *data;
} newvec_t;

size_t i;
int errors = 0;

#pragma omp declare mapper(newvec_t v) map(v, v.data[0:v.len])

int target_update_to_mapper() {
  
  OMPVV_TEST_OFFLOADING;

  newvec_t s;

  s.data = (double *)calloc(N,sizeof(double));
  s.len = N;
  #pragma omp target data map(tofrom: s)
  { 
    //update array in host values
    for (i = 0; i < s.len; i++) {
      s.data[i] = i;
    }
    // update even array position values from host
    // This should set them to i  
    #pragma omp target update to(s.data[0:s.len/2:2])

    //update array on the device
    #pragma omp target 
    for (i = 0; i < s.len; i++) {
      s.data[i] += i;
    }
  } //end target

  for (i =0; i < N; i++) { 
    if(i%2){ //odd positions should be result[i] = i
      OMPVV_TEST_AND_SET(errors, s.data[i] != i);
    }
    else{   //even positions should be result[i] = 2i
      OMPVV_TEST_AND_SET(errors, s.data[i] != 2*i);
    }
  } 
  return errors;
  
}

int main() {
  
  OMPVV_TEST_OFFLOADING;
  
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, target_update_to_mapper());

  OMPVV_REPORT_AND_RETURN(errors);

}
