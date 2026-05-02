!//===------ test_requires_default_mem_order_seq_cst.F90 --------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks for support of the atomic_default_mem_order clause on the 
! requires directive. This clause determines the default memory behavior for
! atomic constructs. These behaviors are seq_cst, acq_rel, and relaxed.
! This test checks the seq_cst behavior, which is also the default.
!
! Adapted from 5.0 OpenMP example acquire_release.2.f90
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

PROGRAM test_requires_default_mem_order_seq_cst
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  !$omp requires atomic_default_mem_order(seq_cst)

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_atomic_seq_cst() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_atomic_seq_cst()
    INTEGER:: x, y, thrd, tmp, errors

    x = 0
    y = 0
    errors = 0

    OMPVV_INFOMSG("test_atomic_seq_cst")
    
    call omp_set_dynamic(.false.)
    call omp_set_num_threads(2)

    !$omp parallel private(thrd, tmp)
       thrd = omp_get_thread_num()
       IF (thrd .EQ. 0) THEN
          x = 10 
          !$omp atomic write 
          y = 1
          !$omp end atomic
       ELSE
          tmp = 0
          DO WHILE (tmp .EQ. 0) 
            !$omp atomic read 
            tmp = y
            !$omp end atomic 
          END DO
          OMPVV_TEST_AND_SET_VERBOSE(errors, x .NE. 10)
       END IF
    !$omp end parallel

    test_atomic_seq_cst = errors
  END FUNCTION test_atomic_seq_cst
END PROGRAM test_requires_default_mem_order_seq_cst
