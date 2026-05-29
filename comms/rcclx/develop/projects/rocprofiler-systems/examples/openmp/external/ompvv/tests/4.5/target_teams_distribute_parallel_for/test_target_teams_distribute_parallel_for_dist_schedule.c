//===-test_target_teams_distribute_parallel_for_dist_schedule.c-===//
//
// OpenMP API Version 4.5 Nov 2015
//
// testing the combined construct target teams distribute parallel for
// This test will check if dist_schedule(static, 4) taking effect
//
//===---------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

int arr[32] = {0};
int chunked = 1;

int Dist_ScheduleStatic() {
  int ErrCount = 0;
int num_teams, num_threads;

#pragma omp target teams distribute parallel for num_teams(2)\
        thread_limit(4)  map(tofrom: num_teams, num_threads)\
        dist_schedule(static, 4)
  for (int i = 0; i < 32; ++i) {
    if(omp_get_team_num()==0 && omp_get_thread_num() == 0){
      num_teams = omp_get_num_teams();
      num_threads = omp_get_num_threads();
    }
    arr[i] = omp_get_team_num();
  }

  if ((num_teams > OMPVV_NUM_TEAMS_DEVICE) || (num_threads > OMPVV_NUM_THREADS_DEVICE)) {
    ErrCount += 1;
  }
  if(num_teams == 1){
    OMPVV_WARNING("Only 1 team was created, the feature cannot be tested.");
    return ErrCount;
  }
  
  for(int i=0; i<32; i+=4){
    int temp = arr[i];
    for(int j=1; j< 4;j++)
      if(arr[i+j] != temp){
        chunked = 0;
        break;
      }
    if(chunked == 0) {
      ErrCount++;
      OMPVV_WARNING("The given iterations were not scheduled according to the static chunks requested");
      break;
    }
  }

  return ErrCount;
}



int main() {
  OMPVV_TEST_OFFLOADING;
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, Dist_ScheduleStatic() != 0);
  OMPVV_REPORT_AND_RETURN(errors);
}
