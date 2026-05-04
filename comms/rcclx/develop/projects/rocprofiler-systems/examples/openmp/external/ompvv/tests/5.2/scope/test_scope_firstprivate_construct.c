//--------------- test_scope_firstprivate_construct.c -----------------------------//
//
// OpenMP API Version 5.2 Nov 2021 
//
// This test checks that the scope firstprivate construct clause is properly working.
// The test itself passes a test integer into the scope pragma and ensures that
// all changes made to it are not kept outside of the scope region.
//----------------------------------------------------------------------------//


#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

int test_scope(){
	int errors = 0;
	int test_int = 1;
	#pragma omp target parallel map(tofrom: errors) 
	{
		#pragma omp scope firstprivate(test_int)
		{
			test_int += 1;
			OMPVV_TEST_AND_SET(errors,test_int != 2);
		}
	}
	OMPVV_ERROR_IF(errors, "firstprivate int is not updating correctly");
	OMPVV_TEST_AND_SET_VERBOSE(errors,test_int != 1);
	OMPVV_INFOMSG_IF(test_int == 2, "test int was not firstprivate");
	return errors;
}

int main(){
	int errors = 0;
	OMPVV_TEST_OFFLOADING;
	OMPVV_TEST_AND_SET_VERBOSE(errors, test_scope() != 0);
	OMPVV_REPORT_AND_RETURN(errors);
}
