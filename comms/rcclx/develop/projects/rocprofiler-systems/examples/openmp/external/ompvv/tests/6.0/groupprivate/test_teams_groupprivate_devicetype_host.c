//===--- test_groupprivate_devicetype_host.c ---------------------------===//
//
// OpenMP API Version 6.0
// Tests the groupprivate directive ensuring proper 
// behavior on the host. 
//
////===----------------------------------------------------------------------===//


#include <omp.h>
#include <stdio.h>
#include "ompvv.h"

int group_sum;
#pragma omp groupprivate(group_sum) device_type(host)
 
int test_groupprivate_devicetype_host(){
    int errors = 0;
    int host_sum = 0;
    int team_sum = 0;

    #pragma omp teams num_teams(OMPVV_NUM_TEAMS_HOST) reduction(+: team_sum)
    {
        group_sum = omp_get_team_num();

        team_sum += group_sum;
    
    }

    for (int i = 0; i < OMPVV_NUM_TEAMS_HOST; i++){
        host_sum += i;
    }
    OMPVV_TEST_AND_SET_VERBOSE(errors, team_sum != host_sum);
    return errors;
}

int main(){
        int errors = 0;
        OMPVV_TEST_AND_SET_VERBOSE(errors, test_groupprivate_devicetype_host() != 0);
        OMPVV_REPORT_AND_RETURN(errors);
}

