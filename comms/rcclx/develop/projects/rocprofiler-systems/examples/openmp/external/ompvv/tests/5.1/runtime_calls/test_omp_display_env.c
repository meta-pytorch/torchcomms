//===-- test_omp_display_env.c --------------------------------------------===//
//
// OpenMP API Version 5.1 Aug 2021
//
// Tests the behavior of omp_display_env. This test should print out the OpenMP
// version number and the initial values of the ICV's associated with the
// environment variables
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

int test_omp_display_env(){
	omp_display_env(1);
	return 0;
}

int main(){
	OMPVV_REPORT_AND_RETURN(test_omp_display_env());
}
