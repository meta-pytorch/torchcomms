//===------------ test_target_taskloop_simd_order_reproducible_device.c ------===//
//
// OpenMP API Version 5.1 Nov 2020
//
// This test checks that the order(reproducible:concurrent) clause is properly handled.
// Leverages the previous array item to determine if the same threads executed
// in the same order over the array. If both arrays match the test passes, 
// if not it fails as the reproducible clause failed.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 24

int main() {
	int errors = 0;
	int x[N];
	int y[N];

	for (int i = 0; i < N; i++) {
		x[i] = i;
		y[i] = i;
	}

	OMPVV_TEST_OFFLOADING;

	#pragma omp target map(tofrom: x,y)
	{
		#pragma omp parallel
		{
		   #pragma omp single
		   #pragma omp taskloop simd order(reproducible:concurrent)
		      for (int i = 0; i < N; i++) {
			 x[i] = x[i] + 2;	
		      }

		   #pragma omp single
		   #pragma omp taskloop simd order(reproducible:concurrent)
		      for (int i = 0; i < N; i++) {
			y[i] = x[i] + 2;
		      }
		}

	}

	for (int i = 0; i < N; i++) {
    		OMPVV_TEST_AND_SET(errors, y[i] != i + 4);
	}

	OMPVV_REPORT_AND_RETURN(errors);
}
