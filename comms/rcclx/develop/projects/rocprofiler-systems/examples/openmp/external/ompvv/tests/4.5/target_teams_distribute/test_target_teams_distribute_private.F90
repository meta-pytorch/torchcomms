!===--- test_target_teams_distribute_private.F90----------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the private clause on a target teams distribute directive to
! indicate that the variable in the private clause should be made private to
! each team executing the teams distribute region.  The test then operates on
! the privatized variable in such a way that would most likely cause competing
! operations if the variable is not privatized.  If the computation completes
! without errors, we assume that the privatization occured.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

      PROGRAM test_target_teams_distribute_device
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none
        INTEGER :: errors
        errors = 0
        OMPVV_TEST_OFFLOADING
        OMPVV_TEST_VERBOSE(test_private() .ne. 0)

        OMPVV_REPORT_AND_RETURN()
      CONTAINS
        INTEGER FUNCTION test_private()
          INTEGER,DIMENSION(N):: a, b, c, d
          INTEGER:: privatized, errors, x, y, num_teams
          errors = 0
          num_teams = -1

          DO x = 1, N
            a(x) = 1
            b(x) = x
            c(x) = 2 * x
            d(x) = 0
          END DO

          !$omp target teams distribute private(privatized) map(to: a(1:N), &
          !$omp& b(1:N), c(1:N)) map(from: d(1:N)) map(tofrom: num_teams) &
          !$omp& num_teams(OMPVV_NUM_TEAMS_DEVICE)
          DO x = 1, N
            IF (omp_get_team_num() .eq. 0) THEN
              num_teams = omp_get_num_teams()
            END IF
            privatized = 0
            DO y = 1, a(x) + b(x)
              privatized = privatized + 1
            END DO
            d(x) = c(x) * privatized
          END DO

          DO x = 1, N
            OMPVV_TEST_AND_SET(errors, (d(x) .ne. (1 + x) * 2 * x))
          END DO

          OMPVV_WARNING_IF(num_teams .eq. 1, "Test ran with one team. Results of private test are inconclusive.")
          OMPVV_TEST_AND_SET_VERBOSE(errors, num_teams .lt. 1)

          test_private = errors
        END FUNCTION test_private
      END PROGRAM test_target_teams_distribute_device
