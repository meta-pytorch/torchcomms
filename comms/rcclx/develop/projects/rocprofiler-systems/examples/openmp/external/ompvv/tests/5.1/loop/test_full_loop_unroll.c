//===--- test_full_loop_unroll.c -----------------------------------------------===//
//
// OpenMP API Version 5.1 August 2021 
//
// This test checks the behavior of the 5.1 unroll construct (2.11.9.2 in 5.1 spec). 
// "The unroll construct fully or partially unrolls a loop", which is more of a code generation
// compiler feature. 
// This test specifically checks the unroll construct with the full clause, in which the 
// loop will be completely unrolled.
// Referenced 5.1 Example Doc 7.2 
////===--------------------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

# define N 1024

int errors; 

int test_full_loop_unroll() {
	int arr[N];
	int parallel_arr[N];
	int sum = 0;
	int parallel_sum = 0;

	// sequential loop
	for (int i = 0; i < N; i++) {
		arr[i] = i;
		sum += arr[i];
	}

	// omp unroll full loop
# pragma omp unroll full
	for (int i = 0; i < N; i++) {
		parallel_arr[i] = i;
	}

	// sequential sum of parallel array
	for (int i = 0; i < N; i++) {
		parallel_sum += parallel_arr[i];
	}
	OMPVV_TEST_AND_SET_VERBOSE(errors, parallel_sum != sum);
	OMPVV_INFOMSG_IF(sum == 0, "Array was not initialized.");
	OMPVV_INFOMSG_IF(parallel_sum == 0, "Something went wrong with loop unroll.");
	OMPVV_INFOMSG_IF(parallel_sum == sum, "Test passed.");
	OMPVV_INFOMSG_IF(parallel_sum != sum, "Test did not pass.");
	return errors;
}

int main() {
	errors = 0;
	OMPVV_TEST_OFFLOADING;
	OMPVV_TEST_AND_SET_VERBOSE(errors, test_full_loop_unroll() != 0);
	OMPVV_REPORT_AND_RETURN(errors);
}
