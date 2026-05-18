//--------------- test_omp_num_teams_env_2.c---------------------//
//
// OpenMP API Version 5.1 Aug 2021
//
// The objective of this test is to check that the num_teams env
// var is set properly. It is important to note that the name of the test
// is very important to the env var being set. It follows a convention set
// up for testing env variables in run.sh. This test itself simply checks
// that the max number of teams is set to the appropriate variable value specified
// in the test name. 
//-----------------------------------------------------------------------//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

int test_case(){
	int errors = 0;
	int env = 10;
	env = omp_get_max_teams();
	OMPVV_TEST_AND_SET_VERBOSE(errors, env != 2);
	OMPVV_INFOMSG_IF(env == 0, "Environment variable not set");
	OMPVV_INFOMSG_IF(env == 10, "Call to omp_get_max_teams() did not update local variable");
	return errors;
}

int main(){
	int errors = 0;
	OMPVV_TEST_OFFLOADING;
	OMPVV_TEST_AND_SET_VERBOSE(errors, test_case() != 0);
	OMPVV_REPORT_AND_RETURN(errors);
}
