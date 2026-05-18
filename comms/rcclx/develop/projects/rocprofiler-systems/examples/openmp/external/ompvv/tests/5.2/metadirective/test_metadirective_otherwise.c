//--------------- test_metadirective_otherwise.c ----------------------------//
// OpenMP API Version 5.2 Nov 2021
// *************************
// DIRECTIVE: metadirective
// CLAUSES: when, otherwise
// *************************
// The otherwise clause of the metadirective construct is being tested. The
// metadirective construct contains a when clause, which itself contains a
// context-selector (device = {kind(nohost)}) and a directive-variant (nothing).
// The metadirective construct also contains an otherwise clause, which itself
// contains a directive-variant (target map(tofrom : on_host)).
// The when clause of the metadirective construct is
// evaluated for a valid context-selector pertaining to a device related
// characteristic of a certain kind (nohost), defined in the OpenMP additional
// definitions. If a target construct were specified, the evaluation would
// be true, and the directive-variant nothing would be executed as if
// #pragma omp nothing were present. Target is not present, so the
// directive-variant of the otherwise clause will be executed. This would not
// occur if device = {kind(host)} were specified. If omp_is_initial_device()
// evaluates to false (0), then the test will pass.
//----------------------------------------------------------------------------//

#include "ompvv.h"
#include <omp.h>

int test_otherwise() {
  int errors = 0;
  int on_host = 1;
  int num_devices = omp_get_num_devices();

  #pragma omp metadirective \
    when(device = {kind(nohost)}: nothing) \
    otherwise(target map(tofrom : on_host))
    {
      on_host = omp_is_initial_device();
    }

  if (num_devices > 0) {
    OMPVV_ERROR_IF(
        on_host == 1,
        "A device was available, but the target directive was not executed");
    OMPVV_TEST_AND_SET_VERBOSE(errors, on_host == 1);
  } else {
    OMPVV_WARNING(
        "NO DEVICES ARE AVAILABLE, OTHERWISE CLAUSE WAS ACCESSED ON HOST");
  }

  return errors;
}

int main() {
  int errors = 0;
  OMPVV_TEST_OFFLOADING;
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_otherwise() != 0);
  OMPVV_REPORT_AND_RETURN(errors);
}
