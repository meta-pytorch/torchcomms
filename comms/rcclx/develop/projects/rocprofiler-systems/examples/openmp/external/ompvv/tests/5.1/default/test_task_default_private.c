//-----------test_task_default_private.c---------------------------
//
// OpenMP API Version 5.1 Aug 2021
//
// Test the behavior of the default clasue when the specified
// data-sharing attribute is private. This test focuses on the
// use of the task and target construct and should work for both
// host and target implementations. Private should allow any data
// initialized outside of the task region to have its own instance
// within the task region and retain any value changes made within
// the region.
//-----------------------------------------------------------------

#include <omp.h>
#include "ompvv.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N 1024


int test_task_default_private(){
	int errors = 0;
	int test_num= 1;
	int sum = 0;
	
	#pragma omp task shared(sum) default(private)
	{
	#pragma omp target map(tofrom: sum, test_num)
	{
		int test_num = 2;
		sum += test_num;
		
	}
	}	
	OMPVV_TEST_AND_SET_VERBOSE(errors, sum != 2);
	OMPVV_INFOMSG_IF(sum == 1, "Did not use variable delcared in task region");
	return errors;
}

int main(){
	int errors = 0;
	OMPVV_TEST_OFFLOADING;
	OMPVV_TEST_AND_SET_VERBOSE(errors, test_task_default_private() != 0);
	OMPVV_REPORT_AND_RETURN(errors);
}
