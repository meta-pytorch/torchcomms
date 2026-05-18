//===----------------- test_target_teams_loop_defaultmap.c-----------------===//
//
// OpenMP API Version 5.0 Nov 2018
// The tests in this file tests 'teams loop' construct coupled with defaultmap()
// clause.
// all the tests target the device offload using target construct
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 1024

/**
  This is a basic test to demonstrate defaultmap(tofrom)
  clause with target teams loop construct.
*/
int testDefaultMapToFrom() {
  int errors = 0;
  int device_data[N], host_data[N];
  for (int i = 0; i < N; i++) {
    device_data[i] = i;
    host_data[i] = 0;
  }
#pragma omp target teams loop defaultmap(tofrom)
  for (int i = 0; i < N; i++) {
    host_data[i] += device_data[i];
    device_data[i] = 2*i;
  }
  // checking results
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, host_data[i] != i);
    OMPVV_TEST_AND_SET(errors, device_data[i] != 2*i);
  }
  return errors;
}

/**
  This is a basic test to demonstrate defaultmap(from)
  clause with target teams loop construct.
*/
int testDefaultMapFrom() {
  int errors = 0;
  int device_data[N], host_data[N];
  for (int i = 0; i < N; i++) {
    device_data[i] = i;
    host_data[i] = 0;
  }
#pragma omp target teams loop defaultmap(from)
  for (int i = 0; i < N; i++) {
    host_data[i] = i;
  }
  // checking results
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, host_data[i] != i);
  }
  return errors;
}

/**
  This is a basic test to demonstrate defaultmap(to)
  clause with target teams loop construct.
*/
int testDefaultMapTo() {
  int errors = 0;
  int device_data[N], host_data[N];
  for (int i = 0; i < N; i++) {
    device_data[i] = i;
    host_data[i] = 0;
  }
#pragma omp target teams loop defaultmap(to) map(tofrom: host_data)
  for (int i = 0; i < N; i++) {
    host_data[i] += device_data[i];
  }
  // checking results
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, host_data[i] != i);
  }
  return errors;
}

/**
  This is a basic test to demonstrate defaultmap(default)
  clause with target teams loop construct.
*/
int testDefaultMapDefault() {
  int errors = 0;
  int device_data[N], host_data[N];
  for (int i = 0; i < N; i++) {
    device_data[i] = i;
    host_data[i] = 0;
  }
#pragma omp target teams loop defaultmap(default)
  for (int i = 0; i < N; i++) {
    host_data[i] += device_data[i];
    device_data[i] = 2*i;
  }
  // checking results
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, host_data[i] != i);
    OMPVV_TEST_AND_SET(errors, device_data[i] != 2*i);
  }
  return errors;
}

/**
  This is a basic test to demonstrate defaultmap(firstprivate)
  clause with target teams loop construct.
*/
int testDefaultMapFirstPrivate() {
  int errors = 0;
  int device_data[N], host_data[N];
  for (int i = 0; i < N; i++) {
    device_data[i] = i;
    host_data[i] = 0;
  }
#pragma omp target teams loop defaultmap(firstprivate)
  for (int i = 0; i < N; i++) {
    host_data[i] = device_data[i];
  }
  // checking results
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, host_data[i] != 0);
  }
  return errors;
}

/**
  This is a basic test to demonstrate defaultmap(tofrom:aggregate)
  clause with target teams loop construct.
*/
int testDefaultMapToFromAggr() {
  int errors = 0;
  int device_data[N], host_data[N];
  for (int i = 0; i < N; i++) {
    device_data[i] = i;
    host_data[i] = 0;
  }
#pragma omp target teams loop defaultmap(tofrom:aggregate) map(tofrom: host_data)
  for (int i = 0; i < N; i++) {
    host_data[i] += device_data[i];
    device_data[i] = 2*i;
  }
  // checking results
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, host_data[i] != i);
    OMPVV_TEST_AND_SET(errors, device_data[i] != 2*i);
  }
  return errors;
}

/**
  This is a basic test to demonstrate defaultmap(tofrom:scalar)
  clause with target teams loop construct.
*/
int testDefaultMapToFromScalar() {
  int errors = 0;
  int x = 0;
#pragma omp target teams loop defaultmap(tofrom:scalar) reduction(+:x)
  for (int i = 0; i < N; i++) {
    x += i;
  }
  // checking results
  OMPVV_TEST_AND_SET(errors, x != (N*(N-1)/2));
  return errors;
}

/**
  This is a basic test to demonstrate defaultmap(tofrom:aggregate)
  clause with target teams loop construct using structure.
*/
int testDefaultMapToFromAggrStr() {
  int errors = 0;
  struct dataS {
    int device_data[N];
    int host_data[N];
  } dataObjS;
  for (int i = 0; i < N; i++) {
    dataObjS.device_data[i] = i;
    dataObjS.host_data[i] = 0;
  }
#pragma omp target teams loop defaultmap(tofrom:aggregate)
  for (int i = 0; i < N; i++) {
    dataObjS.host_data[i] = dataObjS.device_data[i];
  }
  // checking results
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, dataObjS.host_data[i] != i);
  }
  return errors;
}

/**
  This is a basic test to demonstrate defaultmap(tofrom:pointer)
  clause with target teams loop construct.
*/
int testDefaultMapToFromPtr() {
  int errors = 0;
  int *ptr[N];
#pragma omp target teams loop defaultmap(tofrom:pointer)
  for (int i = 0; i < N; i++) {
    ptr[i] = (int*)0xefefefef;
  }
  // checking results
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, ptr[i] != ((int*)0xefefefef));
  }
  return errors;
}

/**
  This is a basic test to demonstrate defaultmap(alloc)
  clause with target teams loop construct.
*/
int testDefaultMapAlloc() {
  int errors = 0;
  int device_data[N], host_data[N];
  for (int i = 0; i < N; i++) {
    device_data[i] = i;
    host_data[i] = 0;
  }
#pragma omp target teams loop defaultmap(alloc) map(to: device_data[0:N])
  for (int i = 0; i < N; i++) {
    host_data[i] = device_data[i];
  }
  // checking results to verify host_data is unchanged
  for (int i = 0; i < N; i++) {
    OMPVV_TEST_AND_SET(errors, host_data[i] != 0);
  }
  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, testDefaultMapToFrom());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testDefaultMapFrom());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testDefaultMapTo());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testDefaultMapDefault());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testDefaultMapFirstPrivate());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testDefaultMapToFromAggr());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testDefaultMapToFromScalar());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testDefaultMapToFromAggrStr());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testDefaultMapToFromPtr());
  OMPVV_TEST_AND_SET_VERBOSE(errors, testDefaultMapAlloc());
  OMPVV_REPORT_AND_RETURN(errors);
}
