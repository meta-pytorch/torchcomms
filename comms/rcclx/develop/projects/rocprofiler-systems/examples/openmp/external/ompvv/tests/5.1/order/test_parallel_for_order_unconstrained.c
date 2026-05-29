//===------------ test_target_parallel_for_order_unconstrained.c -------------===//
//
// OpenMP API Version 5.1 Nov 2020
//
// This test checks that the order(unconstrained:concurrent) clause is properly
// handled. Since with unconstrained the values can be executed in any order
// this test simply checks that the correct values have been calculated for
// the array
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

	for (int i = 0; i < N; i++) {
		x[i] = i;
	}

	OMPVV_TEST_OFFLOADING;

	#pragma omp target map(tofrom: x)
	{
		#pragma omp parallel for order(unconstrained:concurrent)
		for (int i = 0; i < N; i++) {
			x[i] = x[i] + 2;	
		}
	}

	for (int i = 0; i < N; i++) {
    		OMPVV_TEST_AND_SET(errors, x[i] != i + 2);
	}

	OMPVV_REPORT_AND_RETURN(errors);
}
