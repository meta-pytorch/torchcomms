!/===-- test_task_affinity.F90 --------------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This is a test of the affinity clause on a task construct. The affinity
! clause indicates to the compiler that the task should execute physically
! near to the memory location of the list items in the clause. This test
! checks that the affinity clause can be used in the appropriate context
! of a task construct but cannot guarantee that the compiler provides any
! exact semantics for the clause.
!
!/===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_task_affinity
  USE iso_c_binding
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(run_tasks() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION run_tasks()
    INTEGER:: errors, i
    INTEGER,ALLOCATABLE:: A(:), B(:)

    errors = 0

    allocate(A(N))
    allocate(B(N))
    DO i = 1, N
       A(i) = i
    END DO

    !$omp task depend(out: B) shared(B) affinity(A)
    DO i = 1, N
       B(i) = A(i)
    END DO
    !$omp end task

    !$omp task depend(in: B) shared(B) affinity(A)
    DO i = 1, N
       B(i) = B(i) + A(i)
    END DO
    !$omp end task

    !$omp taskwait

    DO i = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, B(i) .ne. i*2)
    END DO

    run_tasks = errors
  END FUNCTION run_tasks
END PROGRAM test_task_affinity
