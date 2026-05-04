//--------------test_assume_noopenmpconstructs.c----------------------------------//
// OpenMP API Version 6.0 November 2024
// Pg. 900, line 22
// ***********
// DIRECTIVE:assume
// CLAUSE:no_openmp_constructs
// ***********
// The no_openmp_constructs clause is used to guarantee that the section of code
// which applies to the assumes directive does not contain any OpenMP
// constructs. The argument supplied to the clause is the "can_assume" int
// variable. If the variable is 1, the no_openmp_constructs assumes that the
// assumption is true. If the variable is 0, the clause assumes that the
// assumption is false. If the argument is not specified (empty) then the default
// assumption is true. If the clause is working correctly, then the array A will
// be updated with the expected values and the sum of those values will be
// checked for correctness.
//---------------------------------------------------------------------------------//
#include "ompvv.h"
#include <omp.h>

#define N 10
#define can_assume 1
#define cannot_assume 0

int test_assumes_no_openmp_constructs() {
  int A[N] = {0};
  int errors = 0, sum = 0;
  int i;

  //parallel for construct
  #pragma omp assume no_openmp_constructs(cannot_assume)
  {
    #pragma omp parallel for
    for (i = 0; i < N; ++i) {
      A[i] = i + 1;
    }
  }

  for (i = 0; i < N; ++i) {
    sum += A[i];
    A[i] = 0;
  }

  if (sum != N * (N + 1) / 2)
    ++errors;

  sum = 0;

  //no constructs
  #pragma omp assume no_openmp_constructs(can_assume)
  {
    for (i = 0; i < N; ++i) {
      A[i] = 2 * (i + 1);
    }
  }

  for (i = 0; i < N; ++i) {
    sum += A[i];
    A[i] = 0;
  }
  if (sum != 2 * N * (N + 1) / 2)
    ++errors;

  sum = 0;
  //omp.h routine
  #pragma omp assume no_openmp_constructs(can_assume)
  {
    for (i = 0; i < N; ++i) {
      A[i] = 2 * (i + 1);
    }

    for (i = 0; i < N; ++i) {
      if (omp_in_parallel() == 0){
        sum += A[i];
        A[i] = 0;
      }
    }
  }

  if (sum != 2 * N * (N + 1) / 2)
    ++errors;

  sum = 0;

  //no constructs
  #pragma omp assume no_openmp_constructs
  {
    for (i = 0; i < N; ++i) {
      A[i] = 3 * (i + 1);
    }
  }

  for (i = 0; i < N; ++i) {
    sum += A[i];
  }

  if (sum != 3 * N * (N + 1) / 2)
    ++errors;

  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET(errors, test_assumes_no_openmp_constructs() != 0);
  OMPVV_REPORT_AND_RETURN(errors);
  return errors;
}
