!//===------ test_atomic_hint.F90 --------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_atomic_hint
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(test_atomic_hint_uncontended() .ne. 0)
  OMPVV_TEST_VERBOSE(test_atomic_hint_contended_nonspec() .ne. 0)
  OMPVV_TEST_VERBOSE(test_atomic_hint_speculative() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_atomic_hint_uncontended()
    INTEGER:: x, num_threads, errors

    errors = 0
    x = 0
    num_threads = -1

    OMPVV_INFOMSG("test_atomic_hint_uncontended")

    !$omp parallel num_threads(2) default(shared)
       IF (omp_get_thread_num() .EQ. 0) THEN
          num_threads = omp_get_num_threads()
       END IF
       !$omp atomic hint(omp_sync_hint_uncontended)
          x = x + 1
       !$omp end atomic 
    !$omp end parallel

    OMPVV_ERROR_IF(num_threads .LT. 0, "Test ran with invalid number of threads (less than zero)")
    OMPVV_WARNING_IF(num_threads .EQ. 1, "Test ran with one thread, so the results are not conclusive")

    OMPVV_TEST_AND_SET_VERBOSE(errors, x .NE. num_threads)

    test_atomic_hint_uncontended = errors
  END FUNCTION test_atomic_hint_uncontended

  INTEGER FUNCTION test_atomic_hint_contended_nonspec()
    INTEGER:: x, num_threads, errors

    errors = 0
    x = 0
    num_threads = -1

    OMPVV_INFOMSG("test_atomic_hint_contended_nonspec")

    !$omp parallel num_threads(OMPVV_NUM_THREADS_HOST) default(shared)
       IF (omp_get_thread_num() .EQ. 0) THEN
          num_threads = omp_get_num_threads()
       END IF
       !$omp atomic hint(omp_sync_hint_contended+omp_sync_hint_nonspeculative)
          x = x + 1
       !$omp end atomic 
    !$omp end parallel

    OMPVV_ERROR_IF(num_threads .LT. 0, "Test ran with invalid number of threads (less than zero)")
    OMPVV_WARNING_IF(num_threads .EQ. 1, "Test ran with one thread, so the results are not conclusive")

    OMPVV_TEST_AND_SET_VERBOSE(errors, x .NE. num_threads)

    test_atomic_hint_contended_nonspec = errors
  END FUNCTION test_atomic_hint_contended_nonspec

  INTEGER FUNCTION test_atomic_hint_speculative()
    INTEGER:: num_threads, errors, i
    INTEGER, DIMENSION(N):: a

    errors = 0
    num_threads = -1

    DO i = 1, N
      a(i) = 1
    END DO

    OMPVV_INFOMSG("test_atomic_hint_speculative")

    !$omp parallel do num_threads(OMPVV_NUM_THREADS_HOST) default(shared)
    DO i = 1, N
       IF (i .EQ. 1) THEN
          num_threads = omp_get_num_threads()
          !$omp atomic hint(omp_sync_hint_speculative)
             a(2) = a(2) + 1
          !$omp end atomic 
       END IF
       !$omp atomic hint(omp_sync_hint_speculative)
          a(i) = a(i) + i
       !$omp end atomic 
    END DO
    !$omp end parallel do

    OMPVV_ERROR_IF(num_threads .LT. 0, "Test ran with invalid number of threads (less than zero)")
    OMPVV_WARNING_IF(num_threads .EQ. 1, "Test ran with one thread, so the results are not conclusive")

    DO i = 1, N
       IF (i .EQ. 2) THEN
          OMPVV_TEST_AND_SET_VERBOSE(errors, a(i) .NE. 4)
       ELSE
          OMPVV_TEST_AND_SET_VERBOSE(errors, a(i) .NE. i+1)
       END IF
    END DO


    test_atomic_hint_speculative = errors
  END FUNCTION test_atomic_hint_speculative
END PROGRAM test_atomic_hint
