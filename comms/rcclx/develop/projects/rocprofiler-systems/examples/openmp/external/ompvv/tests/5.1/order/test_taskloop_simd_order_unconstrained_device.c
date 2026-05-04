//===------ test_target_taskloop_simd_order_unconstrained_concurrent.c ------===//
//
// OpenMP API Version 5.1 Nov 2020
//
// This test checks that the order(unconstrained:concurrent) clause is properly handled.
// As per the definition the order in which they are executed does not matter. 
// Instead focus on that the correct result is calculated, regardless of execution
// order. 
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
		   #pragma omp taskloop simd order(unconstrained:concurrent)
		   for (int i = 0; i < N; i++) {
			   x[i] = x[i] + 2;	
		   }

		   #pragma omp single
		   #pragma omp taskloop simd order(unconstrained:concurrent)
		   for (int i = 0; i < N; i++) {
			   y[i] = y[i] + 2;
		   }
		}

	}

	for (int i = 0; i < N; i++) {
    		OMPVV_TEST_AND_SET(errors, y[i] != i + 2);
        	OMPVV_TEST_AND_SET(errors, x[i] != i + 2)
	}

	OMPVV_REPORT_AND_RETURN(errors);
}
