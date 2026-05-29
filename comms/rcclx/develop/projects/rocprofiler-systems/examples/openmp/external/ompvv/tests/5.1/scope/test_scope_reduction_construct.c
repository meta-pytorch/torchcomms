//--------------- test_scope_reduction_construct.c-----------------------//
//
// OpenMP API Version 5.1 Aug 2021
//
// The objective of this test is ensure the scope construct works as 
// intended. The scope construct defines a structured block that is
// executed by all threads in a team but where additional OpenMP operation
// can be specified. This test follows the example given by the OpenMP
// 5.1 Specification Example sheet. 
//-----------------------------------------------------------------------//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024

int test_scope(int n, int a[], int s){
	int errors = 0;
	#pragma omp parallel shared(s)
	{
		int loc_s = 0;
		#pragma omp for
			for (int i = 0; i < n; i++)
				loc_s += a[i];
		#pragma omp single
		{
			s = 0;
		}
		#pragma omp scope reduction(+:s)
		{
			s += loc_s;
		}
	}
	OMPVV_TEST_AND_SET_VERBOSE(errors, s != 1024);
	OMPVV_INFOMSG_IF(s == 0, "sum was not set");
	return errors;	
}	

int main(){
	int a[N];
	int s = 0;
	int errors = 0;
	for (int i = 0; i < N; i++){
		a[i] = 1;
	}
	OMPVV_TEST_OFFLOADING;
	OMPVV_TEST_AND_SET_VERBOSE(errors, test_scope(N,a,s) != 0);
	OMPVV_REPORT_AND_RETURN(errors);
}
