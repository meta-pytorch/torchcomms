//--------------- test_target_get_max_teams.c --------------------------------//
// OpenMP API Version 5.1 Nov 2020
// Pg. 669, line 29
// ***************************
// ROUTINE: omp_get_max_teams
// ***************************
// This test uses the omp_get_max_teams routine on the host and device (if
// available) to check what the max teams capacity is for host and the current
// device. If the routine returns 0, the nteams-var ICV is 0, thus it is not
// treated as an upper bound and the behavior of the program is implementation
// defined. If the routine were to return a negative integer, the same behavior
// applies. If the routine returns a positive integer, the value of the ICV is
// treated as an upperbound. If the value is a number greater than the maximum
// capacity that the implementation supports, then the test will pass if the
// number of teams is less than that value, but the value itself has no
// functional purpose. The ICV is initialized to 0 per device. Thus, if
// omp_get_num_teams returns a positive number as expected, the test will pass
// when the ICV value is 0 or less than 0. In the case where the OMP_NUM_TEAMS
// environmental variable is set to a positive integer, or the omp_set_num_teams
// routine is given a positive integer as an argument, the test will pass if and
// only if the omp_get_num_teams routine returns an integer that is less than or
// equal to the positive ICV value.
//----------------------------------------------------------------------------//
#include "ompvv.h"
#include <omp.h>

int test_get_max_teams(int offload) {
  int errors = 0;
  int max_teams;
  int num_teams = 0; // a value that is not possible for omp_get_num_teams()

  #pragma omp target map(tofrom: max_teams) if(offload)
  {
    max_teams = omp_get_max_teams();
  }
  #pragma omp target map(tofrom : num_teams) if(offload)
  {
    #pragma omp teams 
    {
      if (omp_get_team_num() == 0)
        num_teams = omp_get_num_teams();
    }
  }

  OMPVV_ERROR_IF(
      max_teams > 0 && num_teams > max_teams,
      "Number of teams reported exceeded max number of teams (max no. > 0)");
  OMPVV_TEST_AND_SET(errors, max_teams > 0 && num_teams > max_teams);

  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET(errors, test_get_max_teams(0) != 0);
  OMPVV_REPORT(errors);
  OMPVV_TEST_OFFLOADING;
  OMPVV_TEST_AND_SET(errors, test_get_max_teams(1) != 0);
  OMPVV_REPORT(errors);
  OMPVV_RETURN(errors);
  return errors;
}
