!===------ test_target_teams_distribute_dist_schedule.F90 ----------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test checks that the dist_schedule clause (which must have kind
! static) correctly causes CHUNK_SIZE iterations to be split among the
! number of teams the test is run with (in a round-robin fashion in order
! of the team number) when a chunk size is given. The test also confirms
! that when no chunk size is given, that each team receives no more than
! one "chunk" of implementation-defined size.
!
!===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024
#define CHUNK_SIZE 64

PROGRAM main
  USE iso_fortran_env
  USE ompvv_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_SHARED_ENVIRONMENT

  OMPVV_TEST_VERBOSE(test_dist_schedule() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_dist_schedule()
    INTEGER:: errors, num_teams, x, counter
    LOGICAL:: err_cond
    INTEGER,DIMENSION(N):: a, b

    errors = 0
    counter = -1

    DO x = 1, N
       a(x) = -1
       b(x) = -1
    END DO

    !$omp target teams distribute map(from: num_teams) map(tofrom: a(1:N))&
    !$omp& dist_schedule(static, CHUNK_SIZE)
    DO x = 1, N
       IF (omp_get_team_num() .eq. 0) THEN
          num_teams = omp_get_num_teams()
       END IF
       a(x)= omp_get_team_num()
    END DO
    !$omp end target teams distribute

    OMPVV_WARNING_IF(num_teams .eq. 1, "Cannot test because num_teams was 1.")
    OMPVV_TEST_AND_SET_VERBOSE(errors, num_teams .lt. 1)

    DO x = 1, N
       IF (MOD(x - 1, CHUNK_SIZE) .eq. 0) THEN
          counter = MOD(counter + 1, num_teams)
       END IF
       OMPVV_TEST_AND_SET_VERBOSE(errors, a(x) .ne. counter)
       OMPVV_ERROR_IF(a(x) .ne. counter, "Iterations improperly scheduled for dist(static, chunk_size)")
    END DO

    num_teams = -1

    !$omp target teams distribute map(from: num_teams) &
    !$omp& map(tofrom: b(1:N)) dist_schedule(static)
    DO x = 1, N
       IF (omp_get_team_num() .eq. 0) THEN
          num_teams = omp_get_num_teams()
       END IF
       b(x) = omp_get_team_num()
    END DO
    !$omp end target teams distribute

    OMPVV_WARNING_IF(num_teams .eq. 1, "Cannot test because num_teams was 1.")
    OMPVV_TEST_AND_SET_VERBOSE(errors, num_teams .lt. 1)

    DO x = 2, N
       err_cond = (b(x) .lt. b(x - 1)) .or. (b(x) .gt. (b(x - 1) + 1))
       OMPVV_ERROR_IF(err_cond, "Iterations improperly scheduled for dist(static)")
       OMPVV_TEST_AND_SET_VERBOSE(errors, err_cond);
       IF (err_cond) THEN
          exit
       END IF
    END DO

    test_dist_schedule = errors
  END FUNCTION test_dist_schedule
END PROGRAM main
