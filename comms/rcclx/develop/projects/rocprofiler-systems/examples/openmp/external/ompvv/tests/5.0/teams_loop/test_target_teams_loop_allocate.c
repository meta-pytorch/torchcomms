//===-------------- test_target_teams_loop_allocate.c---------------------===//
//
// OpenMP API Version 5.0 Nov 2018
// The tests in this file tests 'teams loop' construct coupled with map()
// clause in combination with different allocator types
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

/**
  This is a basic test to demonstrate how allocate can be used with
  target teams loop construct. Allocate clause is used to allocate a local
  variable of size int.
*/
int testTargetTeamsAllocateCl() {
  int a[N], b[N], c[N], local = 0;
  int errors = 0;
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = 2*i;  // Even
    b[i] = 2*i + 1;  // Odd
    c[i] = 0;
  }
  // Execute on target
#pragma omp target teams loop map(to: a[0:N], b[0:N]) map(from: c[0:N]) \
        uses_allocators(omp_low_lat_mem_alloc) \
        allocate(omp_low_lat_mem_alloc: local) firstprivate(local)
  for (int i = 0; i < N; i++) {
    local = a[i] + b[i];
    c[i] = local;
  }
  // Validate
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != a[i] + b[i]);
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, local != 0);
  return errors;
}

/**
  This is a basic test to demonstrate use of allocate clause with
  omp_default_mem_alloc allocator to allocate an array.
*/
int testTargetTeamsAllocateArrayDefaultAlloc() {
  int a[N], b[N], c[N], local[N];
  int errors = 0;
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = 2*i;  // Even
    b[i] = 2*i + 1;  // Odd
    local[i] = 0;
  }
  // Execute on target
#pragma omp target teams loop map(to: a[0:N], b[0:N]) map(from: c[0:N])\
        uses_allocators(omp_default_mem_alloc)\
        allocate(omp_default_mem_alloc: local) firstprivate(local)
  for (int i = 0; i < N; i++) {
    local[i] = a[i] + b[i];
    c[i] = local[i];
  }
  // Validate
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != a[i] + b[i]);
    OMPVV_TEST_AND_SET_VERBOSE(errors, local[i] != 0);
  }
  return errors;
}

/**
  This is a basic test to demonstrate use of allocate clause with
  omp_large_cap_mem_alloc allocator to allocate an array.
*/
int testTargetTeamsAllocateArrayLargeCapAlloc() {
  int a[N], b[N], c[N], local[N];
  int errors = 0;
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = 2*i;  // Even
    b[i] = 2*i + 1;  // Odd
    local[i] = 0;
  }
  // Execute on target
#pragma omp target teams loop map(to: a[0:N], b[0:N]) map(from: c[0:N])\
        uses_allocators(omp_large_cap_mem_alloc)\
        allocate(omp_large_cap_mem_alloc: local) firstprivate(local)
  for (int i = 0; i < N; i++) {
    local[i] = a[i] + b[i];
    c[i] = local[i];
  }
  // Validate
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != a[i] + b[i]);
    OMPVV_TEST_AND_SET_VERBOSE(errors, local[i] != 0);
  }
  return errors;
}

/**
  This is a basic test to demonstrate use of allocate clause with
  omp_const_mem_alloc allocator to allocate an variable.
*/
int testTargetTeamsAllocateConstAlloc() {
  int a[N], b[N];
  const int local = 3;
  int errors = 0;
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = 2*i;  // Even
    b[i] = 0;  // Odd
  }
  // Execute on target
#pragma omp target teams loop map(to: a[0:N]) map(from: b[0:N])\
        firstprivate(local) uses_allocators(omp_const_mem_alloc)\
         allocate(omp_const_mem_alloc: local)
  for (int i = 0; i < N; i++) {
    b[i] = a[i] + local;
  }
  // Validate
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, b[i] != a[i] + local);
  }
  return errors;
}

/**
  This is a basic test to demonstrate use of allocate clause with
  omp_high_bw_mem_alloc allocator to allocate an array.
*/
int testTargetTeamsAllocateArrayHighBWAlloc() {
  int a[N], b[N], c[N], local[N];
  int errors = 0;
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = 2*i;  // Even
    b[i] = 2*i + 1;  // Odd
    local[i] = 0;
  }
  // Execute on target
#pragma omp target teams loop map(to: a[0:N], b[0:N]) map(from: c[0:N])\
        uses_allocators(omp_high_bw_mem_alloc)\
        allocate(omp_high_bw_mem_alloc: local) firstprivate(local)
  for (int i = 0; i < N; i++) {
    local[i] = a[i] + b[i];
    c[i] = local[i];
  }
  // Validate
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != a[i] + b[i]);
    OMPVV_TEST_AND_SET_VERBOSE(errors, local[i] != 0);
  }
  return errors;
}

/**
  This is a basic test to demonstrate use of allocate clause with
  omp_low_lat_mem_alloc allocator to allocate an array.
*/
int testTargetTeamsAllocateArrayLowLatAlloc() {
  int a[N], b[N], c[N], local[N];
  int errors = 0;
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = 2*i;  // Even
    b[i] = 2*i + 1;  // Odd
    local[i] = 0;
  }
  // Execute on target
#pragma omp target teams loop map(to: a[0:N], b[0:N]) map(from: c[0:N])\
        uses_allocators(omp_low_lat_mem_alloc)\
        allocate(omp_low_lat_mem_alloc: local) firstprivate(local)
  for (int i = 0; i < N; i++) {
    local[i] = a[i] + b[i];
    c[i] = local[i];
  }
  // Validate
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != a[i] + b[i]);
    OMPVV_TEST_AND_SET_VERBOSE(errors, local[i] != 0);
  }
  return errors;
}

/**
  This is a basic test to demonstrate use of allocate clause with
  omp_cgroup_mem_alloc allocator to allocate an array.
*/
int testTargetTeamsAllocateArrayCGroupAlloc() {
  int a[N], b[N], c[N], local[N];
  int errors = 0;
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = 2*i;  // Even
    b[i] = 2*i + 1;  // Odd
    local[i] = 0;
  }
  // Execute on target
#pragma omp target teams loop map(to: a[0:N], b[0:N]) map(from: c[0:N])\
        uses_allocators(omp_cgroup_mem_alloc)\
        allocate(omp_cgroup_mem_alloc: local) firstprivate(local)
  for (int i = 0; i < N; i++) {
    local[i] = a[i] + b[i];
    c[i] = local[i];
  }
  // Validate
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != a[i] + b[i]);
    OMPVV_TEST_AND_SET_VERBOSE(errors, local[i] != 0);
  }
  return errors;
}

/**
  This is a basic test to demonstrate use of allocate clause with
  omp_pteam_mem_alloc allocator to allocate an array.
*/
int testTargetTeamsAllocateArrayPTeamAlloc() {
  int a[N], b[N], c[N], local[N];
  int errors = 0;
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = 2*i;  // Even
    b[i] = 2*i + 1;  // Odd
    local[i] = 0;
  }
  // Execute on target
#pragma omp target teams loop map(to: a[0:N], b[0:N]) map(from: c[0:N])\
        uses_allocators(omp_pteam_mem_alloc)\
        allocate(omp_pteam_mem_alloc: local) firstprivate(local)
  for (int i = 0; i < N; i++) {
    local[i] = a[i] + b[i];
    c[i] = local[i];
  }
  // Validate
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, c[i] != a[i] + b[i]);
    OMPVV_TEST_AND_SET_VERBOSE(errors, local[i] != 0);
  }
  return errors;
}

int main() {
  int errors = 0;
  
  OMPVV_TEST_OFFLOADING;
  
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTargetTeamsAllocateCl());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTargetTeamsAllocateArrayDefaultAlloc());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTargetTeamsAllocateArrayLargeCapAlloc());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTargetTeamsAllocateConstAlloc());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTargetTeamsAllocateArrayHighBWAlloc());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTargetTeamsAllocateArrayLowLatAlloc());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTargetTeamsAllocateArrayCGroupAlloc());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testTargetTeamsAllocateArrayPTeamAlloc());
  OMPVV_REPORT_AND_RETURN(errors);
}
