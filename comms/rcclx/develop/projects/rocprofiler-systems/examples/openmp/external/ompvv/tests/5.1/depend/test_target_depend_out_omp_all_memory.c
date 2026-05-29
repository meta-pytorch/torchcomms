//===------------ test_target_depend_out_omp_all_memory.c --------------------===//
//
// OpenMP API Version 5.1 Nov 2020
//
// This test checks that the special omp_all_memory locator keyword works
// as intended. The test checks this by having two variables passed into the 
// target parallel region, depending on both values. If both variables are
// properly passed through the sections then it passes. If not, the test fails.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int main() {
	int errors = 0;
	int x[N];
  	int y = 5;

	for (int i = 0; i < N; i++) {
		x[i] = i;
	}

	OMPVV_TEST_OFFLOADING;

	#pragma omp target map(tofrom: errors) map(to: x, y)
	{
	    #pragma omp parallel
	    #pragma omp single
	    {
	      #pragma omp task shared(x, y) depend(out: omp_all_memory)
	      {
		for(int i = 0; i < N; i++) {
		  x[i] += y;   
		}
	      }
	      #pragma omp task shared(x, y, errors) depend(in: x, y)
	      {
		for (int i = 0; i < N; i++) {
		  OMPVV_TEST_AND_SET(errors, x[i] != i + y);
		}
	      }
	      #pragma omp taskwait
	    }
	}

	OMPVV_REPORT_AND_RETURN(errors);
}
