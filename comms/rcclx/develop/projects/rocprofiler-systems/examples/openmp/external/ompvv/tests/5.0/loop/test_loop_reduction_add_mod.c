//===--- test_loop_reduction_add_mod.c ------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test uses the reduction clause on a loop directive, testing that the
// variables in the reduction clause are properly reduced using the add
// operator. Standard types are reduced: int, float, double, char.
// Test written in reference to test_loop_reduction_add.c.
//
//===-----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#define A -32
#define B 64
#define L 40
#define X 1

int test_add() {
  int n = A, errors=0; 
  float f = A;
  double d = A;
  char c = L;
  int exp_n = A + B;
  float exp_f = A + B;
  double exp_d = A + B;
  char exp_c = L + B;

#pragma omp parallel 
  {
  #pragma omp loop reduction(+:n,f,d,c) 
    for(int i=0;i<B;i++){
      n = n + X;
      f = f + X;  
      d = d + X;  
      c = c + X;
    }
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, (n != exp_n) || (f != exp_f) || (d != exp_d) || (c != exp_c));

  OMPVV_ERROR_IF(errors, "\nn|EXPECTED:%d, RECEIVED:%d\nf|EXPECTED:%.1f, RECEIVED:%.1f\nd|EXPECTED:%.1f, RECEIVED:%.1f\nc|EXPECTED:%c, RECEIVED:%c\n",exp_n,n,exp_f,f,exp_d,d,exp_c,c); 

  return errors;
}
int main() {
  int total_errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(total_errors, test_add() != 0);

  OMPVV_REPORT_AND_RETURN(total_errors);
}
