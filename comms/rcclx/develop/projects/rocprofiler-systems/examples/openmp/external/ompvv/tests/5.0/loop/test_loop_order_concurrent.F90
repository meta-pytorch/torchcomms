!===--- test_loop_order_concurrent.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks the loop directive with the order(concurrent) clause.
! The order(concurrent) clause is assumed to be present if it is not
! present, so this test covers the standalone loop directive as well. The
! test creates a parallel region with a loop construct nested within, and
! performs simple operations on an int array which are then checked for
! correctness. Additionally, since loop binds to a parallel region, the test
! checks randomly that other threads wait before proceeding out of the loop
! region. The number of threads is checked in the parallel region but after
! the loop construct because runtime API calls are not permitted in loop
! directive regions.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_loop_order_concurrent
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_loop() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_loop()
    INTEGER,DIMENSION(N):: a, b, c
    INTEGER,DIMENSION(OMPVV_NUM_THREADS_HOST):: rand_indexes
    INTEGER:: errors, x, wait_errors, num_threads
    REAL:: curr_rand

    errors = 0
    wait_errors = 0

    DO x = 1, N
       a(x) = 1
       b(x) = x + 1
       c(x) = 2*(x + 1)
    END DO

    CALL RANDOM_SEED()

    DO x = 1, OMPVV_NUM_THREADS_HOST
       CALL RANDOM_NUMBER(curr_rand)
       rand_indexes(x) = MODULO(INT(curr_rand*10),N) + 1
    END DO

    !$omp parallel num_threads(OMPVV_NUM_THREADS_HOST)
    !$omp loop order(concurrent)
    DO x = 1, N
       a(x) = a(x) + b(x)*c(x)
    END DO
    IF (a(rand_indexes(omp_get_thread_num() + 1)) .eq. 1) THEN
       !$omp atomic update
       wait_errors = wait_errors + 1
    END IF
    IF (omp_get_thread_num() .eq. 0) THEN
       num_threads = omp_get_num_threads()
    END IF
    !$omp end parallel

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, a(x) .ne. 1 + (b(x)*c(x)))
    END DO

    OMPVV_WARNING_IF(num_threads .eq. 1, "Test ran with one thread. Cannot guarantee loop construct parallelism.")
    OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads .lt. 1)
    OMPVV_ERROR_IF(wait_errors .gt. 0, "Threads in parallel region did not wait for loop region to finish.")

    test_loop = errors + wait_errors
  END FUNCTION test_loop
END PROGRAM test_loop_order_concurrent
