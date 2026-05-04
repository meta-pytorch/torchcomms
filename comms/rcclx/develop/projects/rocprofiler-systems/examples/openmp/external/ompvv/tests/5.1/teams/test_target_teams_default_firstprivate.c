//===--------- test_target_teams_default_firstprivate.c ----------------------===//
//
// OpenMP API Version 5.1 Nov 2020
//
// This test uses the default(firstprivate) clause on a teams directive. The test
// validates that when the default(firstprivate) clause is present all
// variables without explicit sharing rules are not avaialble outside the region
// and are private to each team. The default(firstprivate) clause is tested using
// the not_shared variable, whose value should have not changed after the target
// teams construct since all changes to the firstprivate variable should not persist
// after the construct. Additionally if there is a race condition, we know the 
// variable is not defaulting to firstprivate either.
//
//===------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

int main() {

	OMPVV_TEST_OFFLOADING;

	int errors = 0;
	int not_shared = 5;
	int num_teams = 0; 

	#pragma omp target teams default(firstprivate) map(tofrom:num_teams,errors) shared(num_teams,errors) num_teams(OMPVV_NUM_TEAMS_DEVICE)
	{
		if (omp_get_team_num() == 0) {
			num_teams = omp_get_num_teams();
			if( not_shared != 5 ) {
				errors = errors + 1;
			}
		}

		for (int i = 0; i < omp_get_num_teams(); i++) {
			not_shared += 5;
		}
	}


	OMPVV_WARNING_IF(num_teams != OMPVV_NUM_TEAMS_DEVICE, "Number of teams was unexpected, test results likely inconclusive");

	OMPVV_TEST_AND_SET(errors, (not_shared != 5));

	OMPVV_REPORT_AND_RETURN(errors);
}
