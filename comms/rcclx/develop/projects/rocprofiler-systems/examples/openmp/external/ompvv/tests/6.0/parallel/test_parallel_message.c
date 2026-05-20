//--------------- test_parallel_message.c------------------------------------//
// OpenMP API Version 6.0 August 2024
// Pg. 870, line 25
// ***********
// DIRECTIVE:parallel
// CLAUSE:message
// ***********
// The message clause for the parallel directive is being tested. If the clause
// executes correctly, than the user will see a message corresponding to the top
// parallel region. Otherwise, the user should notice a message corresponding to
// the second region. Additionally, the return value of the test function should
// be non-zero.
//----------------------------------------------------------------------------//
#include "ompvv.h"
#include <omp.h>

int test_message() {
  int return_value = 1;
  if (_OPENMP) {
    #pragma omp parallel message("Entering first parallel region.")
    {
      if (omp_get_thread_num() == 0)
        --return_value;
    }
  } else {
    #pragma omp parallel message("Entering second parallel region.")
    {
    }
  }

  return return_value;
}

int main() {
  int errors = 0;
  OMPVV_TEST_AND_SET(errors, test_message() != 0);
  OMPVV_REPORT_AND_RETURN(errors);
  return errors;
}
