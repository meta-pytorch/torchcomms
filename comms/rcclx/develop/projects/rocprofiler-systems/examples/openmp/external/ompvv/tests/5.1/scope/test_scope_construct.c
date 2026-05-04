//------------ test_scope_construct.c --------------------//
// OpenMP API Version 5.1 Nov 2020
// *****************
// DIRECTIVE: scope
// *****************
// The scope directive is being tested. Scope binds to the 
// innermost parallel region. However, since no
// clauses are specified, there should be no modification
// to the behavior of code execution. Therefore, the test
// checks that the the total is updated correctly as if the
// scope directive were not present.
//--------------------------------------------------------//

#include <omp.h>
#include "ompvv.h"

#define N 64

int test_scope() {
  int errors = 0;
  int total = 0;
  #pragma omp target parallel shared(total) map(tofrom : total)
  {
    #pragma omp scope
    {
      #pragma omp for
      for (int i = 0; i < N; ++i) {
        #pragma omp atomic update
        ++total;
      }
    }
  }
  OMPVV_TEST_AND_SET_VERBOSE(errors, total != N);
  return errors;
}
int main() {
  int errors = 0;
  OMPVV_TEST_OFFLOADING;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_scope() != 0);
  OMPVV_REPORT_AND_RETURN(errors);
}
