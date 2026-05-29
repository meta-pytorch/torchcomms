!//===------ test_taskloop_reduction.F90 ----------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks the taskloop directive with the reduction clause specified.
!
!//===---------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_taskloop_reduction
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(test_taskloop_red() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_taskloop_red()
    INTEGER:: errors, test_sum, num_threads, real_sum, i
    INTEGER, DIMENSION(N):: a, b

    errors = 0
    test_sum = 0
    real_sum = 0
    num_threads = -1

    DO i = 1, N
       a(i) = 5
       b(i) = i * 2
    END DO

    !$omp parallel num_threads(OMPVV_NUM_THREADS_HOST) shared(a, b, num_threads, test_sum)
       !$omp single
       !$omp taskloop reduction(+:test_sum)
       DO i = 1, N
          !$omp atomic
          test_sum = test_sum + a(i)*b(i)
          !$omp end atomic
       END DO
       !$omp end taskloop
       !$omp end single
       num_threads = omp_get_num_threads()

       !$omp single
       !$omp taskloop reduction(+:test_sum)
       DO i = 1, N
          !$omp atomic
          test_sum = test_sum + 1
          !$omp end atomic
       END DO
       !$omp end taskloop
       !$omp end single
    !$omp end parallel

    real_sum = real_sum + N

    DO i = 1, N
       real_sum = real_sum + a(i)*b(i)
    END DO

    OMPVV_TEST_AND_SET_VERBOSE(errors, test_sum .NE. real_sum)

    OMPVV_WARNING_IF(num_threads .EQ. 1, "Test ran with one thread, so parallelism of taskloop can't be guaranteed.")
    OMPVV_ERROR_IF(num_threads .LT. 1, "Test returned an invalid number of threads.")
    OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads .LT. 1)

    test_taskloop_red = errors
  END FUNCTION test_taskloop_red
END PROGRAM test_taskloop_reduction
