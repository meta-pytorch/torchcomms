//-------------- test_fuse_apply_looprange.c----------------------------------//
// OpenMP API Version 6.0 August 2024
// Pg. 868, line 16
// ***********
// DIRECTIVE:fuse
// CLAUSE:apply, looprange
// ***********
// The looprange clause for the fuse directive is being tested with the apply
// clause. The looprange clause takes count of the number of loops (4) and
// begins fusion at loop 2 (left value). 2 loops are specified as the right
// value, so only 2 consecutive loops will be merged. Thus, the number of
// remaining loops will be 3. The apply clause performs an unroll transformation
// on the fused loop resulting from the application of looprange. If the array
// updates occur incorrectly during fusion, the test will fail.
//----------------------------------------------------------------------------//
#include "ompvv.h"

#define N 4

int test_apply_looprange() {
  int A[4][N] = {0};
  int errors = 0;
  int j;

  #pragma omp fuse apply(fused : unroll) looprange(2, 2)
  {
    for (j = 0; j < N; ++j)
      A[0][j] = j + 1;
    for (j = 0; j < N; ++j)
      A[1][j] = N + j + 1;
    for (j = 0; j < N; ++j)
      A[2][j] = 2 * N + j + 1;
    for (j = 0; j < N; ++j)
      A[3][j] = 3 * N + j + 1;
  }

  for (int i = 0; i < 4; ++i) {
    for (j = 0; j < N; ++j) {
      if (A[i][j] != i * N + j + 1)
        ++errors;
    }
  }

  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET(errors, test_apply_looprange() != 0);
  OMPVV_REPORT_AND_RETURN(errors);
  return errors;
}
