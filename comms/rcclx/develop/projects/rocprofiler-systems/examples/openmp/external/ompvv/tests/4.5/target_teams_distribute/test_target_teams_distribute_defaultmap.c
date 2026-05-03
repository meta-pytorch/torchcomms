//===--- test_target_teams_distribute_defaultmap.c---------------------------===//
//
// OpenMP API Version 4.5 Nov 2015
//
// This test uses the defaultmap clause on a target teams distribute directive.
// This tests the following scalars: char, short, int, float, double, and enum.
// Both using the clause defaultmap(tofrom:scalar) is used. When it is used,
// the test tests the to nature by setting arrays to the value.  Then it is also
// tested that, as opposed to the default action on scalars which is to first-
// privatize them, they are shared and returned to the host.
//
// It also tests the default operation of treating scalars without the defaultmap
// clause.  The test first tests the privatization of the firstprivatized
// scalars and then separately tests the proper initialization of them separately
//
////===----------------------------------------------------------------------===//

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"
#include <math.h>

#define ARRAY_SIZE 1024
enum enum_type { VAL1 = 1, VAL2, VAL3, VAL4};

int test_defaultmap_on() {
  OMPVV_INFOMSG("test_defaultmap_on");

  int errors = 0;

  // we try with all the scalars
  char scalar_char = 'a';
  char char_array[ARRAY_SIZE];
  short scalar_short = 10;
  short short_array[ARRAY_SIZE];
  int scalar_int = 11;
  int int_array[ARRAY_SIZE];
  float scalar_float = 5.5f;
  float float_array[ARRAY_SIZE];
  double scalar_double = 10.45;
  double double_array[ARRAY_SIZE];
  enum enum_type scalar_enum = VAL1;
  enum enum_type enum_array[ARRAY_SIZE];

#pragma omp target teams distribute defaultmap(tofrom: scalar) map(from: char_array[0:ARRAY_SIZE], \
                                                                   short_array[0:ARRAY_SIZE], \
                                                                   int_array[0:ARRAY_SIZE], \
                                                                   float_array[0:ARRAY_SIZE], \
                                                                   double_array[0:ARRAY_SIZE], \
                                                                   enum_array[0:ARRAY_SIZE])
  for (int x = 0; x < ARRAY_SIZE; ++x){
    char_array[x] = scalar_char;
    short_array[x] = scalar_short;
    int_array[x] = scalar_int;
    float_array[x] = scalar_float;
    double_array[x] = scalar_double;
    enum_array[x] = scalar_enum;
  }

  for (int x = 0; x < ARRAY_SIZE; ++x){
    OMPVV_TEST_AND_SET_VERBOSE(errors, char_array[x] != 'a');
    if (char_array[x] != 1){
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x){
    OMPVV_TEST_AND_SET_VERBOSE(errors, short_array[x] != 10);
    if (short_array[x] != 2){
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x){
    OMPVV_TEST_AND_SET_VERBOSE(errors, int_array[x] != 11);
    if (int_array[x] != 3){
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x){
    OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(float_array[x] - 5.5f) > .0000001);
    if (fabs(float_array[x] - .4) > .0000000001){
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x){
    OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(double_array[x] - 10.45) > .0000000001);
    if (fabs(double_array[x] - .5) > .0000000001){
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x){
    OMPVV_TEST_AND_SET_VERBOSE(errors, enum_array[x] != VAL1);
    if (enum_array[x] != VAL2){
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x){
    char_array[x] = 1;
    short_array[x] = 2;
    int_array[x] = 3;
    float_array[x] = .4;
    double_array[x] = .5;
    enum_array[x] = VAL2;
  }

#pragma omp target teams distribute defaultmap(tofrom: scalar) map(tofrom: char_array[0:ARRAY_SIZE], \
                                                                   short_array[0:ARRAY_SIZE], \
                                                                   int_array[0:ARRAY_SIZE], \
                                                                   float_array[0:ARRAY_SIZE], \
                                                                   double_array[0:ARRAY_SIZE], \
                                                                   enum_array[0:ARRAY_SIZE])
  for (int x = 0; x < ARRAY_SIZE; ++x) {
    if (omp_get_team_num() == 0) {
      scalar_char = char_array[x];
      scalar_short = short_array[x];
      scalar_int = int_array[x];
      scalar_float = float_array[x];
      scalar_double = double_array[x];
      scalar_enum = enum_array[x];
    }
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_char != char_array[0]);
  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_short != short_array[0]);
  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_int != int_array[0]);
  OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(scalar_float - float_array[0]) > .0000001);
  OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(scalar_double - double_array[0]) > .0000000001);
  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_enum != enum_array[0]);

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, char_array[x] != 1);
    if (char_array[x] != 1) {
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, short_array[x] != 2);
    if (short_array[x] != 2) {
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, int_array[x] != 3);
    if (int_array[x] != 3) {
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(float_array[x] - .4) > .0000001);
    if (fabs(float_array[x] - .4) > .0000000001) {
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(double_array[x] - .5) > .0000000001);
    if (fabs(double_array[x] - .5) > .0000000001) {
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, enum_array[x] != VAL2);
    if (enum_array[x] != VAL2) {
      break;
    }
  }

  return errors;
}

int test_defaultmap_off() {
  OMPVV_INFOMSG("test_defaultmap_off");

  int errors = 0;

  // we try with all the scalars
  char scalar_char = 'a';
  char scalar_char_copy;
  char char_array_a[ARRAY_SIZE];
  char char_array_b[ARRAY_SIZE];
  short scalar_short = 10;
  short scalar_short_copy;
  short short_array_a[ARRAY_SIZE];
  short short_array_b[ARRAY_SIZE];
  int scalar_int = 11;
  int scalar_int_copy;
  int int_array_a[ARRAY_SIZE];
  int int_array_b[ARRAY_SIZE];
  float scalar_float = 5.5f;
  float scalar_float_copy;
  float float_array_a[ARRAY_SIZE];
  float float_array_b[ARRAY_SIZE];
  double scalar_double = 10.45;
  double scalar_double_copy;
  double double_array_a[ARRAY_SIZE];
  double double_array_b[ARRAY_SIZE];
  enum enum_type scalar_enum = VAL1;
  enum enum_type scalar_enum_copy;
  enum enum_type enum_array_a[ARRAY_SIZE];
  enum enum_type enum_array_b[ARRAY_SIZE];

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    char_array_a[x] = x%10;
    char_array_b[x] = 0;
    short_array_a[x] = x%20;
    short_array_b[x] = 0;
    int_array_a[x] = x%30;
    int_array_b[x] = 0;
    float_array_a[x] = x / 40.0;
    float_array_b[x] = 0;
    double_array_a[x] = x / 50.0;
    double_array_b[x] = 0;
    enum_array_a[x] = (enum enum_type)(x%4 + 1);
    enum_array_b[x] = VAL1;
  }


  //Testing the privatization nature of firstprivate default action
#pragma omp target teams distribute map(tofrom: char_array_a[0:ARRAY_SIZE], char_array_b[0:ARRAY_SIZE], \
                                        short_array_a[0:ARRAY_SIZE], short_array_b[0:ARRAY_SIZE], \
                                        int_array_a[0:ARRAY_SIZE], int_array_b[0:ARRAY_SIZE], \
                                        float_array_a[0:ARRAY_SIZE], float_array_b[0:ARRAY_SIZE], \
                                        double_array_a[0:ARRAY_SIZE], double_array_b[0:ARRAY_SIZE], \
                                        enum_array_a[0:ARRAY_SIZE], enum_array_b[0:ARRAY_SIZE])
  for (int x = 0; x < ARRAY_SIZE; ++x) {
    scalar_char = 0;
    for (int y = 0; y < char_array_a[x]; ++y) {
      scalar_char += 1;
    }
    char_array_b[x] = scalar_char;
    scalar_short = 0;
    for (int y = 0; y < short_array_a[x]; ++y) {
      scalar_short += 1;
    }
    short_array_b[x] = scalar_short;
    scalar_int = 0;
    for (int y = 0; y < int_array_a[x]; ++y) {
      scalar_int += 1;
    }
    int_array_b[x] = scalar_int;
    scalar_float = 0;
    for (int y = 0; y < ((int)float_array_a[x]); ++y) {
      scalar_float += .7f;
    }
    float_array_b[x] = scalar_float;
    scalar_double = 0;
    for (int y = 0; y < ((int)double_array_a[x]); ++y) {
      scalar_double += .9;
    }
    double_array_b[x] = scalar_double;
    scalar_enum = VAL1;
    for (int y = 1; y < enum_array_a[x]; ++y) {
      scalar_enum = (enum enum_type)(scalar_enum + 1);
    }
    enum_array_b[x] = scalar_enum;
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, char_array_b[x] != char_array_a[x]);
    if (char_array_b[x] != char_array_a[x]) {
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, short_array_b[x] != short_array_a[x]);
    if (short_array_b[x] != short_array_a[x]) {
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, int_array_b[x] != int_array_a[x]);
    if (int_array_b[x] != int_array_a[x]) {
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(float_array_b[x] - (((int) float_array_a[x]) * .7)) > .00001);
    if (fabs(float_array_b[x] - (((int) float_array_a[x]) * .7)) > .00001) {
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(double_array_b[x] - (((int) double_array_a[x]) * .9)) > .000000001);
    if (fabs(double_array_b[x] - (((int) double_array_a[x]) * .9)) > .000000001) {
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, enum_array_b[x] != enum_array_a[x]);
    if (enum_array_b[x] != enum_array_a[x]) {
      break;
    }
  }

  scalar_char = 26;
  scalar_short = 126;
  scalar_int = 5126;
  scalar_float = 5.126;
  scalar_double = 51.26;
  scalar_enum = VAL2;

  scalar_char_copy = scalar_char;
  scalar_short_copy = scalar_short;
  scalar_int_copy = scalar_int;
  scalar_float_copy = scalar_float;
  scalar_double_copy = scalar_double;
  scalar_enum_copy = scalar_enum;

  // Testing the copy of scalar values to the device
#pragma omp target teams distribute map(tofrom: char_array_a[0:ARRAY_SIZE], short_array_a[0:ARRAY_SIZE], \
                                        int_array_a[0:ARRAY_SIZE], float_array_a[0:ARRAY_SIZE], \
                                        double_array_a[0:ARRAY_SIZE], enum_array_a[0:ARRAY_SIZE])
  for (int x = 0; x < ARRAY_SIZE; ++x) {
    char_array_a[x] = scalar_char;
    short_array_a[x] = scalar_short;
    int_array_a[x] = scalar_int;
    float_array_a[x] = scalar_float;
    double_array_a[x] = scalar_double;
    enum_array_a[x] = scalar_enum;
  }

  // Testing the fact that values should not be modified
  // at the host (unless shared memory or running on the host)
#pragma omp target teams distribute
  for (int x = 0; x < ARRAY_SIZE; ++x) {
    scalar_char = 0;
    scalar_short = 0;
    scalar_int = 0;
    scalar_float = 0;
    scalar_double = 0;
    scalar_enum = VAL3;
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, char_array_a[x] != scalar_char_copy);
    if (char_array_a[x] != scalar_char_copy) {
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, short_array_a[x] != scalar_short_copy);
    if (short_array_a[x] != scalar_short_copy) {
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, int_array_a[x] != scalar_int_copy);
    if (int_array_a[x] != scalar_int_copy) {
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(float_array_a[x] - scalar_float_copy) > .00001);
    if (fabs(float_array_a[x] - scalar_float_copy) > .00001) {
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(double_array_a[x] - scalar_double_copy) > .000000001);
    if (fabs(double_array_a[x] - scalar_double_copy) > .000000001) {
      break;
    }
  }

  for (int x = 0; x < ARRAY_SIZE; ++x) {
    OMPVV_TEST_AND_SET_VERBOSE(errors, enum_array_a[x] != scalar_enum_copy);
    if (enum_array_a[x] != scalar_enum_copy) {
      break;
    }
  }

  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_char != scalar_char_copy);
  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_short != scalar_short_copy);
  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_int != scalar_int_copy);
  OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(scalar_float - scalar_float_copy) > .00001);
  OMPVV_TEST_AND_SET_VERBOSE(errors, fabs(scalar_double - scalar_double_copy) > .000000001);
  OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_enum != scalar_enum_copy);

  return errors;
}
int main() {
  OMPVV_TEST_OFFLOADING;

  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_defaultmap_on());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_defaultmap_off());

  OMPVV_REPORT_AND_RETURN(errors);
}
