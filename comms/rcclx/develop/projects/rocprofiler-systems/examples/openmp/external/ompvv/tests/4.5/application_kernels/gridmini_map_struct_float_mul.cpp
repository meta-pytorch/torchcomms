//===-- gridmini_float_mul_offload.c ---------------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test checks that the float multiplication of members of the struct 'vec'
// in the offloaded region provides the same answer as calculated by host. 
// Since support for struct on map is implementation specific in 4.5 the test does 
// not have a fail condition. 
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include "ompvv.h"
#include "omp.h"

using namespace std;
struct vec {
  float v1;
  float v2;
};

inline  vec mult(vec x, vec y){
  vec out;
  out.v1 = x.v1*y.v1;
  out.v2 = x.v2*y.v2;
  return out;
}

int main(int argc, char* argv[]){
  OMPVV_TEST_OFFLOADING;

  int errors = 0;
  int N = 10;
  float x = (float)rand()/(float)(RAND_MAX/10.0);
  float y = (float)rand()/(float)(RAND_MAX/5.0);
  vec in1,in2;
  in1.v1 = x;
  in1.v2 = x;
  in2.v1 = y;
  in2.v2 = y;
  vec out[N];

  //calulate on host
  vec expected = mult(in1,in2);

#pragma omp target teams distribute parallel for map(to:in1,in2) map(from:out[0:N])
    for(int n = 0; n < N; n++) {
      out[n] = mult(in1,in2);
    }
    
  for(int n = 0; n < N; n++) {
    OMPVV_TEST_AND_SET(errors,out[n].v1 != expected.v1);
    OMPVV_TEST_AND_SET(errors,out[n].v2 != expected.v2);
  }
  if(errors)
    OMPVV_INFOMSG("Maping of entire struct is not supported by this OpenMP implementation.\n");	

  //No error will be reported even if it is recorded.
  OMPVV_REPORT_AND_RETURN(0);

}

