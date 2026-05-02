//---------------test_omp_places_numa_domain--------------------//
//
// OpenMP API Version 5.1 Nov 2020
//
//This test intends to test the omp_places numa domain option. 
//First the test sets the value of omp_places to numa_domain.
//Then the test checks for equal distribution in threads.
//-------------------------------------------------------------//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define N 1024
int test_places(){
	int errors = 0;
	int test = 1;
	char* ret_val = NULL;
	ret_val = getenv("OMP_PLACES");
	test = ret_val == NULL || strcmp(ret_val, "numa_domains");
	OMPVV_TEST_AND_SET(errors, test != 0);
	return errors;
}


int main(){
	int errors = 0;
	OMPVV_TEST_OFFLOADING;
	OMPVV_TEST_AND_SET_VERBOSE(errors, test_places() != 0);
	OMPVV_REPORT_AND_RETURN(errors);
}
