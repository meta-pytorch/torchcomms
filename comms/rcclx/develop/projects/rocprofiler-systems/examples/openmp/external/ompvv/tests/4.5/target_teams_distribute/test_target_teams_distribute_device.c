//===--- test_target_teams_distribute_device.c-------------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses the device clause to indicate which device should execute the
// given target regions.  The test uses the separate device data environments to
// ensure that operations are executed on the specified device.  If only one
// device is available, the test issues a warning.
//
// By having a separate initialization of the same array on each device at the
// same time, if all operations were occuring on the same device, we would
// expect the same results from each device and it wouldn't be able to give
// proper answers for each initialization.
//
////===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define ARRAY_SIZE 1024

int main() {
  int isOffloading = 0;
  OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);

  int num_devices = omp_get_num_devices();
  // Allocate num_devices + 1 to avoid zero-sized VLA if num_devices == 0
  int a[num_devices+1][ARRAY_SIZE];
  int b[ARRAY_SIZE];
  int num_teams[num_devices+1];
  int errors[num_devices+1];
  int sum_errors = 0;

  OMPVV_INFOMSG("Running tests on %d devices", num_devices);

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    for (int dev = 0; dev < num_devices; ++dev) {
      a[dev][x] = 1;
    }
    b[x] = x;
  }

  for (int x = 0; x < num_devices; ++x) {
    num_teams[x] = 0;
    errors[x] = 0;
  }

  for (int dev = 0; dev < num_devices; ++dev) {
#pragma omp target enter data map(to: a[dev][0:ARRAY_SIZE], b[0:ARRAY_SIZE], num_teams[dev:1]) device(dev)
  }

  for (int dev = 0; dev < num_devices; ++dev) {
#pragma omp target teams distribute map(alloc: a[dev][0:ARRAY_SIZE], b[0:ARRAY_SIZE], num_teams[dev:1]) device(dev)
    for (int x = 0; x < ARRAY_SIZE; ++x) {
      if (omp_get_team_num() == 0) {
        num_teams[dev] = omp_get_num_teams();
      }
      a[dev][x] += b[x] + dev;
    }
  }

  for (int dev = 0; dev < num_devices; ++dev) {
#pragma omp target exit data map(from: a[dev][0:ARRAY_SIZE], num_teams[dev:1]) map(delete: b[0:ARRAY_SIZE]) device(dev)
    for (int x = 0; x < ARRAY_SIZE; ++x) {
      OMPVV_TEST_AND_SET_VERBOSE(errors[dev], a[dev][x] != 1 + dev + b[x]);
      if (a[dev][x] != 1 + dev + b[x]) {
        break;
      }
    }
  }

  for (int x = 0; x < num_devices; ++x) {
    sum_errors += errors[x];
  }

  for (int dev = 0; dev < num_devices; ++dev) {
    OMPVV_WARNING_IF(!errors[dev] && num_teams[dev] == 1, "Test operated with one team. Parallelism of teams distribute can't be guaranteed.");
    OMPVV_ERROR_IF(errors[dev], "Test failed with device %d", dev);
  }

  OMPVV_REPORT_AND_RETURN(sum_errors);
}
