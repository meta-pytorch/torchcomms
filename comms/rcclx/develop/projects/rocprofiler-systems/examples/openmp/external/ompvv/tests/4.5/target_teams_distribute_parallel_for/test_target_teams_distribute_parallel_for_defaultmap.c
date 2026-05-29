//===---- test_target_teams_distribute_parallel_for_defaultmap.c - combined consutrct -===//
//
// OpenMP API Version 4.5 Nov 2015
//
// Testing defaultmap of different scalar values. We check when it is off and when it is
// on. The first one should not copy values back from the device of scalars. The second
// should copy the values back even if they are not mapped explicitly.
//
//===----------------------------------------------------------------------------------===//

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define ITERATIONS 10000

int test_defaultmap_on() {
  OMPVV_INFOMSG("test_defaultmap_on");

  int errors = 0;
  int i;

  // we try with all the scalars
  char scalar_char = 'a';
  char scalar_char_cpy[ITERATIONS];
  short scalar_short = 10;
  short scalar_short_cpy[ITERATIONS];
  int scalar_int = 11;
  int scalar_int_cpy[ITERATIONS];
  float scalar_float = 5.5f;
  float scalar_float_cpy[ITERATIONS];
  double scalar_double = 10.45;
  double scalar_double_cpy[ITERATIONS];
  enum { VAL1 = 1, VAL2, VAL3, VAL4} scalar_enum = VAL1, scalar_enum_cpy[ITERATIONS];


  // Testing the to behavior of the tofrom we use an array to avoid data
  // races and check that all threads get the value
#pragma omp target teams distribute parallel for defaultmap(tofrom: scalar)
  for (i = 0; i < ITERATIONS; ++i) {
    scalar_char_cpy[i] = scalar_char;
    scalar_short_cpy[i] = scalar_short;
    scalar_int_cpy[i] = scalar_int;
    scalar_float_cpy[i] = scalar_float;
    scalar_double_cpy[i] = scalar_double;
    scalar_enum_cpy[i] = scalar_enum;
  } // end of omp target

  for (i = 0; i < ITERATIONS; ++i) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_char_cpy[i] != 'a');
    OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_short_cpy[i] != 10);
    OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_int_cpy[i] != 11);
    OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(scalar_float_cpy[i] - 5.5f) > 0.0001);
    OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(scalar_double_cpy[i] - 10.45) > 0.00001);
    OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_enum_cpy[i] != VAL1);
  }

  // Map the same array to multiple devices. initialize with device number
#pragma omp target teams distribute parallel for defaultmap(tofrom: scalar)
  for (i = 0; i < ITERATIONS; ++i) {
    if (omp_get_team_num() == 0) {
      if (omp_get_thread_num() == 0) {
        scalar_char = 'b';
        scalar_short = 20;
        scalar_int = 33;
        scalar_float = 6.5f;
        scalar_double = 20.45;
        scalar_enum = VAL4;
      }
    }
  } // end of omp target

  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_char != 'b');
  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_short != 20);
  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_int != 33);
  OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(scalar_float - 6.5f) > 0.0001);
  OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(scalar_double - 20.45) > 0.00001);
  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_enum != VAL4);

  return errors;
}

int test_defaultmap_off() {
  OMPVV_INFOMSG("test_defaultmap_off");

  int errors = 0;
  int i;

  // we try with all the scalars
  char scalar_char = 'a';
  char scalar_char_cpy[ITERATIONS];
  short scalar_short = 10;
  short scalar_short_cpy[ITERATIONS];
  int scalar_int = 11;
  int scalar_int_cpy[ITERATIONS];
  float scalar_float = 5.5f;
  float scalar_float_cpy[ITERATIONS];
  double scalar_double = 10.45;
  double scalar_double_cpy[ITERATIONS];
  enum { VAL1 = 1, VAL2, VAL3, VAL4} scalar_enum = VAL1, scalar_enum_cpy[ITERATIONS];

  // Testing the copy behavior of the firstprivatization. we use an array to avoid data
  // races and check that all threads get the value
#pragma omp target teams distribute parallel for 
  for (i = 0; i < ITERATIONS; ++i) {
    scalar_char_cpy[i] = scalar_char;
    scalar_short_cpy[i] = scalar_short;
    scalar_int_cpy[i] = scalar_int;
    scalar_float_cpy[i] = scalar_float;
    scalar_double_cpy[i] = scalar_double;
    scalar_enum_cpy[i] = scalar_enum;
  } // end of omp target

  for (i = 0; i < ITERATIONS; ++i) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_char_cpy[i] != 'a');
    OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_short_cpy[i] != 10);
    OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_int_cpy[i] != 11);
    OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(scalar_float_cpy[i] - 5.5f) > 0.0001);
    OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(scalar_double_cpy[i] - 10.45) > 0.00001);
    OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_enum_cpy[i] != VAL1);
  }
  
#pragma omp target teams distribute parallel for
  for (i = 0; i < 1; ++i) {
      scalar_char = 'b';
      scalar_short = 20;
      scalar_int = 33;
      scalar_float = 6.5f;
      scalar_double = 20.45;
      scalar_enum = VAL4;
  } // end of omp target

  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_char != 'a');
  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_short != 10);
  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_int != 11);
  OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(scalar_float - 5.5f) > 0.0001);
  OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(scalar_double - 10.45) > 0.0001);
  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_enum != VAL1);

  return errors;
}
int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_defaultmap_on());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_defaultmap_off());

  OMPVV_REPORT_AND_RETURN(errors);
}
