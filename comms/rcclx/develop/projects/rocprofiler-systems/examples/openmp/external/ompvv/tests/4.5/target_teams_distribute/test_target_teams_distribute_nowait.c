//===--- test_target_teams_distribute_nowait.c-------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses the nowait clause on a target teams distribute directive and
// uses a barrier to resyncronize the target regions.  Since we can't be sure
// that operations will be asyncronous, we can only raise a warning if they are
// not executed asynchronously.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024
#define N_TASKS 16
#define ITERATIONS 1024

int main() {
  OMPVV_WARNING("This test does not throw an error if tasks fail to execute asynchronously, as this is still correct behavior. If execution is not asynchronous, we will throw a warning.");
  int isOffloading = 0;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
  int64_t work_storage[N_TASKS][N];
  int order[N_TASKS];  // Each position marks the order in which that task executed
  int errors = 0;
  int ticket[1] = {0};


#pragma omp target enter data map(to: ticket[0:1], order[0:N_TASKS])

  for (int i = 0; i < N_TASKS; ++i) {
#pragma omp target teams distribute map(alloc: work_storage[i][0:N], ticket[0:1]) nowait
    for (int j = 0; j < N; ++j) {
      work_storage[i][j] = 0;
      for (int k = 0; k < N*(N_TASKS - i); ++k) { // Creates skewed work distribution
	work_storage[i][j] += k*i*j;              // This value will not be verified
      }
      int my_ticket = 0;
#pragma omp atomic capture
      my_ticket = ticket[0]++;
      order[i] = my_ticket;
    }
  }
#pragma omp taskwait

#pragma omp target exit data map(from:ticket[0:1], order[0:N_TASKS])

  if (ticket[0] != N_TASKS*N) {
    OMPVV_ERROR("The test registered a different number of target regions than were spawned");
    errors = 1;
  }

  int was_async = 0;
  for (int i = 1; i < N_TASKS; ++i) {
    if (order[i] <= order[i - 1]) {
      was_async = 1;
      break;
    }
  }

  OMPVV_WARNING_IF(!was_async, "We could not detect asynchronous behavior between target regions");
  OMPVV_INFOMSG_IF(was_async, "Asynchronous behavior detected, this suggests nowait had an effect");
  
  OMPVV_REPORT_AND_RETURN(errors);
}
