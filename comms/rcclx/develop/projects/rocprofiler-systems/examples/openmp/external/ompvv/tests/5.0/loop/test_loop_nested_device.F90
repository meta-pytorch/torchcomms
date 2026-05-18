!===--- test_loop_nested_device.F90 ----------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks the loop directive without any clauses with nested loops.
! The construct is applied in a teams and a parallel construct and the
! correctness of array operations in the nested loops in the loop construct
! is checked. Compare with test_loop_bind.c, which performs the same
! operations in a nested loop with a bind clause. This test checks the above
! all in an offloading (target) context.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 32

PROGRAM test_loop_nested_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(test_loop_nested_teams() .NE. 0)
  OMPVV_TEST_VERBOSE(test_loop_nested_parallel() .NE. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_loop_nested_teams()
    INTEGER,DIMENSION(N,N):: x
    INTEGER,DIMENSION(N):: y
    INTEGER,DIMENSION(N):: z
    INTEGER:: errors, num_teams, i, j

    OMPVV_INFOMSG("test_loop_nested_teams")

    errors = 0
    num_teams = -1

    DO i = 1, N
       DO j = 1, N
          x(j,i) = 1
       END DO
       y(i) = i
       z(i) = 2*i
    END DO

    !$omp target teams num_teams(OMPVV_NUM_TEAMS_DEVICE) thread_limit(OMPVV_NUM_THREADS_DEVICE) map(tofrom: x, y, z, num_teams)
    !$omp loop private(j)
    DO i = 1, N
       DO j = 1, N
          x(j,i) = x(j,i) + y(i)*z(i)
       END DO
    END DO
    !$omp end loop
    !$omp parallel if(.FALSE.)
    IF ( omp_get_team_num() .EQ. 0 ) THEN
       num_teams = omp_get_num_teams()
    END IF
    !$omp end parallel
    !$omp end target teams

    DO i = 1, N
       DO j = 1, N
          OMPVV_TEST_AND_SET_VERBOSE(errors, x(j,i) .NE. (1 + y(i)*z(i)))
       END DO
    END DO

    OMPVV_WARNING_IF(num_teams .EQ. 1, "Test ran with one team, so parallelism of loop construct can't be guaranteed.")
    OMPVV_TEST_AND_SET_VERBOSE(errors, num_teams .LT. 1)
    OMPVV_ERROR_IF(num_teams .LT. 1, "omp_get_num_teams() returned an invalid number of teams.")

    test_loop_nested_teams = errors
  END FUNCTION test_loop_nested_teams

  INTEGER FUNCTION test_loop_nested_parallel()
    INTEGER,DIMENSION(N,N):: x
    INTEGER,DIMENSION(N):: y
    INTEGER,DIMENSION(N):: z
    INTEGER:: errors, num_threads, i, j

    OMPVV_INFOMSG("test_loop_nested_parallel")

    errors = 0
    num_threads = -1

    DO i = 1, N
       DO j = 1, N
          x(j,i) = 1
       END DO
       y(i) = i
       z(i) = 2*i
    END DO

    !$omp target parallel num_threads(OMPVV_NUM_THREADS_HOST) map(tofrom: x, y, z, num_threads)
    !$omp loop
    DO i = 1, N
       DO j = 1, N
          x(j,i) = x(j,i) + y(i)*z(i)
       END DO
    END DO
    !$omp end loop
    IF ( (omp_get_thread_num() .EQ. 0) .AND. (omp_get_team_num() .EQ. 0) ) THEN
       num_threads = omp_get_num_threads()
    END IF
    !$omp end target parallel

    DO i = 1, N
       DO j = 1, N
          OMPVV_TEST_AND_SET_VERBOSE(errors, x(j,i) .NE. (1 + y(i)*z(i)))
       END DO
    END DO

    OMPVV_WARNING_IF(num_threads .EQ. 1, "Test ran with one thread, so parallelism of loop construct can't be guaranteed.")
    OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads .LT. 1)
    OMPVV_ERROR_IF(num_threads .LT. 1, "omp_get_num_threads() returned an invalid number of threads.")

    test_loop_nested_parallel = errors
  END FUNCTION test_loop_nested_parallel

END PROGRAM test_loop_nested_device
