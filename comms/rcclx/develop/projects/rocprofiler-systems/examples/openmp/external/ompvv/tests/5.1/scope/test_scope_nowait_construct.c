//-------------test_scope_nowait_construct.c-------------//
//
//OpenMP API Version 5.1 Aug 2021
//
//Tests the behavior of the scope construct with nowait
//specified. This test should remove the implied barrier
//creatd by the scope construct when nowait is specified.
//-------------------------------------------------------//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>
#include <unistd.h>

#define N 1024

int test1[N];
int test2[N];


int test_scope_nowait(){
	int errors = 0;
	for (int i = 0; i <N; i++){
		test1[i] = 0;
		test2[i] = 0;
	}
	#pragma omp parallel shared(test1, test2)
	{
		#pragma omp scope
		{
			#pragma omp for
			for (int i = 0; i < N; i++){
				test1[i] = 1;
			}
		}
		#pragma omp scope nowait
		{
			#pragma omp for
			for (int i = 0; i < N; i++){
				test2[i] = 1;
			}
		}
	}
	int sum1 = 0;
	int sum2 = 0;
	for (int i = 0; i <N; i++){
		sum1 += test1[i];
		sum2 += test2[i];
	}
	int total = sum1 + sum2;
	OMPVV_TEST_AND_SET_VERBOSE(errors, total != 2048);
        OMPVV_INFOMSG_IF(sum1 != 1024, "first function was not called");
	OMPVV_INFOMSG_IF(sum2 != 1024, "second function was not called");
	return errors;
}

int main(){
	int errors = 0;
	OMPVV_TEST_OFFLOADING;
	OMPVV_TEST_AND_SET_VERBOSE(errors, test_scope_nowait() != 0);
	OMPVV_REPORT_AND_RETURN(errors);
}
