!/===--- test_masked_filter.F90 -------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test utilizes the masked construct to ensure that a section of code
! only runs on a certain thread. When the filter clause is not present, the
! primary thread runs the structured block.
!
!/===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_masked
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(masked() .ne. 0)
  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION masked()
    INTEGER:: errors = 0
    INTEGER:: total = 10
    INTEGER:: ct = 0
    INTEGER:: threads, tot

    OMPVV_INFOMSG("test_masked")
    threads = OMPVV_NUM_THREADS_HOST
    !$omp parallel num_threads(threads) private(tot)
    DO WHILE (.true.)
      !$omp atomic read
      tot = total
      if (tot <= 0) &
        exit
      !$omp masked filter(3)
        OMPVV_TEST_AND_SET_VERBOSE(errors, omp_get_thread_num() .ne. 3) ! primary thread
        ct = ct + 1
        !omp atomic
        total = total - 1
      !$omp end masked
    END DO
    !$omp end parallel
    
    OMPVV_TEST_AND_SET_VERBOSE(errors, ct .ne. 10)
    masked = errors
  END FUNCTION masked
END PROGRAM test_masked
