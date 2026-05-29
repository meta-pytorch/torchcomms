//===--- test_teams.c-------------------------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test uses the teams directive on host and verifies of the 
// requested number of teams with requested number of threads were 
// created. 
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

int main() {
  int num_teams[OMPVV_NUM_TEAMS_DEVICE];
  int num_threads[OMPVV_NUM_THREADS_DEVICE];
  int errors[2] = {0,0};


  for (int x = 0; x < OMPVV_NUM_TEAMS_DEVICE; ++x) {
    num_teams[x] = -99;
  }

  for (int x = 0; x < OMPVV_NUM_THREADS_DEVICE; ++x) {
    num_threads[x] = -99;
  }

#pragma omp teams num_teams(OMPVV_NUM_TEAMS_DEVICE) thread_limit(OMPVV_NUM_THREADS_DEVICE)
  {
    num_teams[omp_get_team_num()] = omp_get_num_teams();
#pragma omp parallel master
    num_threads[omp_get_team_num()]= omp_get_num_threads();
  }

  if (num_teams[0] == 1) {
    OMPVV_WARNING("Test operated with one team. num_teams requested were %d.", OMPVV_NUM_TEAMS_DEVICE);
  } else if (num_teams[0] < 1) {
    OMPVV_ERROR("omp_get_num_teams() reported a value less than one.");
  }

  if (num_threads[0] == 1) {
    OMPVV_WARNING("Team 0 reported only 1 thread. thread_limit was set to %d.", OMPVV_NUM_THREADS_DEVICE);
  } else if (num_threads[0] < 1) {
    OMPVV_ERROR("omp_get_num_threads() reported a value below one.");
  }
  for (int x = 1; x < num_teams[0]; ++x) {
    if (num_teams[x] != num_teams[x - 1]) {
      errors[0]++;
    }
    if (num_threads[x] != num_threads[x - 1]) {
      errors[1]++;
    }
  }

  OMPVV_INFOMSG_IF(!(errors[0] + errors[1]), "Test passed with %d teams and %d threads per team.", num_teams[0], num_threads[0]);

  OMPVV_REPORT_AND_RETURN(errors[0] + errors[1]);
}
