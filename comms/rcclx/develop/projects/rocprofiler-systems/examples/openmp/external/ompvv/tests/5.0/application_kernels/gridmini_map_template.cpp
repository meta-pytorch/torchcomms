//===-- gridmini_map_template.cpp - ---------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks for support for C++ template by verifying that the multiplication 
// of members of the struct 'vec' in the offloaded region provides the same answer 
// as calculated by host. 
//
//===----------------------------------------------------------------------===//

#include <cstdlib>
#include <stdio.h>
#include <iostream>
#include "ompvv.h"
#include "omp.h"

using namespace std;
template <typename T>
struct vec {
  T v1;
  T v2;
};

template <typename T> vec<T> mult(vec<T> x, vec<T> y){
  vec<T> out;
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
  vec<float> in1,in2;
  in1.v1 = x;
  in1.v2 = x;
  in2.v1 = y;
  in2.v2 = y;
  vec<float> out[N];

  //calulate on host
  vec<float> expected = mult(in1,in2);

#pragma omp target teams distribute parallel for map(to:in1,in2) map(from:out[0:N])
    for(int n = 0; n < N; n++) {
      out[n] = mult(in1,in2);
    }
    
  for(int n = 0; n < N; n++) {
    OMPVV_TEST_AND_SET(errors,out[n].v1 != expected.v1);
    OMPVV_TEST_AND_SET(errors,out[n].v2 != expected.v2);
  }


  OMPVV_REPORT_AND_RETURN(errors);

}

