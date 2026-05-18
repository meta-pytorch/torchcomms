!===--- test_loop_lastprivate_device.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test uses the lastprivate clause with a loop directive. According to
! specification, the list items on a lastprivate clause in this context may
! only contain loop iteration variables of loops associated with the loop
! directive. This test checks that the loop iteration variables associated
! with a loop directive and a loop directive with collapse(2) have valid
! values after the parallel region containing the loop. This test checks the
! above in a target context.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define NSIZE 1024
#define NSIZE2 512

PROGRAM test_loop_lastprivate_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_one_loop_level() .ne. 0)
  OMPVV_TEST_VERBOSE(test_two_loop_levels() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_one_loop_level()
    INTEGER,DIMENSION(NSIZE):: a, b
    INTEGER:: errors, lp_errors, x
    CHARACTER(len=400) :: msgHelper

    errors = 0
    lp_errors = 0
    x = 0

    DO x = 1, NSIZE
       a(x) = 1
       b(x) = x
    END DO

    !$omp target parallel num_threads(OMPVV_NUM_THREADS_DEVICE) map(tofrom: a, b, x)
    !$omp loop lastprivate(x)
    DO x = 1, NSIZE
      a(x) = a(x) + b(x)
    END DO
    !$omp end loop
    !$omp end target parallel

    OMPVV_TEST_AND_SET_VERBOSE(lp_errors, x .ne. (NSIZE + 1))
    WRITE(msgHelper, *) "Loop iteration variable in loop construct ended with invalid value."
    OMPVV_ERROR_IF(x .ne. (NSIZE + 1), msgHelper)

    DO x = 1, NSIZE
      OMPVV_TEST_AND_SET_VERBOSE(errors, a(x) - b(x) .ne. 1)
    END DO

    test_one_loop_level = errors + lp_errors
  END FUNCTION test_one_loop_level

  INTEGER FUNCTION test_two_loop_levels()
    INTEGER,DIMENSION(NSIZE2,NSIZE):: a, b
    INTEGER:: errors, lp_errors_x, lp_errors_y, x, y
    CHARACTER(len=400) :: msgHelper

    errors = 0
    lp_errors_x = 0
    lp_errors_y = 0
    x = 0
    y = 0

    DO x = 1, NSIZE
       DO y = 1, NSIZE2
          a(y,x) = 1
          b(y,x) = x + y
       END DO
    END DO

    !$omp target parallel num_threads(OMPVV_NUM_THREADS_DEVICE) map(tofrom: a, b, x, y)
    !$omp loop lastprivate(x, y) collapse(2)
    DO x = 1, NSIZE
       DO y = 1, NSIZE2
          a(y,x) = a(y,x) + b(y,x)
       END DO
    END DO
    !$omp end loop
    !$omp end target parallel

    OMPVV_TEST_AND_SET_VERBOSE(lp_errors_x, x .ne. (NSIZE + 1))
    OMPVV_TEST_AND_SET_VERBOSE(lp_errors_y, y .ne. (NSIZE2 + 1))
    WRITE(msgHelper, *) "Outer loop iteration variable in loop directive with collapse ended with invalid value."
    OMPVV_ERROR_IF(x .ne. (NSIZE + 1), msgHelper)
    WRITE(msgHelper, *) "Inner loop iteration variable in loop directive with collapse ended with invalid value."
    OMPVV_ERROR_IF(y .ne. (NSIZE2 + 1), msgHelper)

    DO x = 1, NSIZE
       DO y = 1, NSIZE2
          OMPVV_TEST_AND_SET_VERBOSE(errors, a(y,x) - b(y,x) .ne. 1)
       END DO
    END DO

    test_two_loop_levels = errors + lp_errors_x + lp_errors_y
  END FUNCTION test_two_loop_levels
END PROGRAM test_loop_lastprivate_device
