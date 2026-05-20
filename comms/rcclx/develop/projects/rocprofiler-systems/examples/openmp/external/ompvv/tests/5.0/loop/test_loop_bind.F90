!===--- test_loop_bind.F90 -------------------------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This test checks the loop directive with the bind(binding) clause. The bind
! clause indicates that the loop construct should apply in the context of the
! given binding, one of teams, parallel, or thread. Each of these bindings
! is tested in an appropriate context and the correctness of results of
! array operations in the nested loop is checked.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 32

PROGRAM test_loop_bind
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_VERBOSE(test_bind_teams() .NE. 0)
  OMPVV_TEST_VERBOSE(test_bind_parallel() .NE. 0)
  OMPVV_TEST_VERBOSE(test_bind_thread_teams() .NE. 0)
  OMPVV_TEST_VERBOSE(test_bind_thread_parallel() .NE. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_bind_teams()
    INTEGER,DIMENSION(N,N):: x
    INTEGER,DIMENSION(N):: y
    INTEGER,DIMENSION(N):: z
    INTEGER:: errors, num_teams, i, j

    OMPVV_INFOMSG("test_loop_bind_teams")

    errors = 0
    num_teams = -1

    DO i = 1, N
       DO j = 1, N
          x(j,i) = 1
       END DO
       y(i) = i
       z(i) = 2*i
    END DO

    !$omp teams num_teams(OMPVV_NUM_TEAMS_HOST) thread_limit(OMPVV_NUM_THREADS_HOST)
    !$omp loop bind(teams) private(j)
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
    !$omp end teams

    DO i = 1, N
       DO j = 1, N
          OMPVV_TEST_AND_SET_VERBOSE(errors, x(j,i) .NE. (1 + y(i)*z(i)))
       END DO
    END DO

    OMPVV_WARNING_IF(num_teams .EQ. 1, "Test ran with one team, so parallelism of loop construct with bind(teams) can't be guaranteed.")
    OMPVV_TEST_AND_SET_VERBOSE(errors, num_teams .LT. 1)
    OMPVV_ERROR_IF(num_teams .LT. 1, "omp_get_num_teams() returned an invalid number of teams.")

    test_bind_teams = errors
  END FUNCTION test_bind_teams

  INTEGER FUNCTION test_bind_parallel()
    INTEGER,DIMENSION(N,N):: x
    INTEGER,DIMENSION(N):: y
    INTEGER,DIMENSION(N):: z
    INTEGER:: errors, num_threads, i, j

    OMPVV_INFOMSG("test_loop_bind_parallel")

    errors = 0
    num_threads = -1

    DO i = 1, N
       DO j = 1, N
          x(j,i) = 1
       END DO
       y(i) = i
       z(i) = 2*i
    END DO

    !$omp parallel num_threads(OMPVV_NUM_THREADS_HOST)
    !$omp loop bind(parallel)
    DO i = 1, N
       DO j = 1, N
          x(j,i) = x(j,i) + y(i)*z(i)
       END DO
    END DO
    !$omp end loop
    IF ( (omp_get_thread_num() .EQ. 0) .AND. (omp_get_team_num() .EQ. 0) ) THEN
       num_threads = omp_get_num_threads()
    END IF
    !$omp end parallel

    DO i = 1, N
       DO j = 1, N
          OMPVV_TEST_AND_SET_VERBOSE(errors, x(j,i) .NE. (1 + y(i)*z(i)))
       END DO
    END DO

    OMPVV_WARNING_IF(num_threads .EQ. 1, "Test ran with one thread, so parallelism of loop construct can't be guaranteed.")
    OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads .LT. 1)
    OMPVV_ERROR_IF(num_threads .LT. 1, "omp_get_num_threads() returned an invalid number of threads.")

    test_bind_parallel = errors
  END FUNCTION test_bind_parallel

  INTEGER FUNCTION test_bind_thread_teams()
    INTEGER,DIMENSION(N):: y
    INTEGER,DIMENSION(N):: z
    INTEGER,DIMENSION(OMPVV_NUM_TEAMS_DEVICE,N,N):: outData
    INTEGER,DIMENSION(N,N):: x
    INTEGER:: errors, num_teams, i, j, k

    OMPVV_INFOMSG("test_loop_bind_thread_teams")

    errors = 0
    num_teams = -1

    DO i = 1, N
       DO j = 1, N
          DO k = 1, OMPVV_NUM_TEAMS_DEVICE
             outData(k,j,i) = 1
          END DO
       END DO
       y(i) = i
       z(i) = 2*i
    END DO

    !$omp teams num_teams(OMPVV_NUM_TEAMS_HOST) thread_limit(OMPVV_NUM_THREADS_HOST) private(x)
    DO i = 1, N
       DO j = 1, N
          x(j,i) = 1
       END DO
    END DO
    !$omp loop bind(thread)
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
    DO i = 1, N
       DO j = 1, N
          outData(omp_get_team_num()+1,j,i) = x(j,i)
       END DO
    END DO
    !$omp end teams

    DO i = 1, N
       DO j = 1, N
          DO k = 1, num_teams
             OMPVV_TEST_AND_SET_VERBOSE(errors, outData(k,j,i) .NE. (1 + y(i)*z(i)))
          END DO
       END DO
    END DO

    OMPVV_WARNING_IF(num_teams .EQ. 1, "Test ran with one team, so parallelism of loop construct can't be guaranteed.")
    OMPVV_TEST_AND_SET_VERBOSE(errors, num_teams .LT. 1)
    OMPVV_ERROR_IF(num_teams .LT. 1, "omp_get_num_teams() returned an invalid number of teams.")

    test_bind_thread_teams = errors
  END FUNCTION test_bind_thread_teams

  INTEGER FUNCTION test_bind_thread_parallel()
    INTEGER,DIMENSION(N):: y
    INTEGER,DIMENSION(N):: z
    INTEGER,DIMENSION(OMPVV_NUM_THREADS_HOST,N,N):: outData
    INTEGER,DIMENSION(N,N):: x
    INTEGER:: errors, num_threads, i, j, k

    OMPVV_INFOMSG("test_loop_bind_thread_parallel")

    errors = 0
    num_threads = -1

    DO i = 1, N
       DO j = 1, N
          DO k = 1, OMPVV_NUM_THREADS_HOST
             outData(k,j,i) = 1
          END DO
       END DO
       y(i) = i
       z(i) = 2*i
    END DO

    !$omp parallel shared(outData) num_threads(OMPVV_NUM_THREADS_HOST) private(x)
    DO i = 1, N
       DO j = 1, N
          x(j,i) = 1
       END DO
    END DO
    !$omp loop bind(thread)
    DO i = 1, N
       DO j = 1, N
          x(j,i) = x(j,i) + y(i)*z(i)
       END DO
    END DO
    !$omp end loop
    IF ( omp_get_thread_num() .EQ. 0 ) THEN
       num_threads = omp_get_num_threads()
    END IF
    DO i = 1, N
       DO j = 1, N
          outData(omp_get_thread_num()+1,j,i) = x(j,i)
       END DO
    END DO
    !$omp end parallel

    DO i = 1, N
       DO j = 1, N
          DO k = 1, num_threads
             OMPVV_TEST_AND_SET_VERBOSE(errors, outData(k,j,i) .NE. (1 + y(i)*z(i)))
          END DO
       END DO
    END DO

    OMPVV_WARNING_IF(num_threads .EQ. 1, "Test ran with one thread, so parallelism of loop construct can't be guaranteed.")
    OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads .LT. 1)
    OMPVV_ERROR_IF(num_threads .LT. 1, "omp_get_num_threads() returned an invalid number of threads.")

    test_bind_thread_parallel = errors
  END FUNCTION test_bind_thread_parallel

END PROGRAM test_loop_bind
