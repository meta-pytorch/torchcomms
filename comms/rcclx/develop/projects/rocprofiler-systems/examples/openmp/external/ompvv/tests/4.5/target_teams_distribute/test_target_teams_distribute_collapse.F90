!===--- test_target_teams_distribute_collapse.F90 --------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the collapse clause and tests that for loops out of the scope
! of the collapsed loops are not parallelized.  This test tests using one and
! two collapsed loops.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 128

      PROGRAM test_target_teams_distribute_collapse
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none
        INTEGER :: errors
        errors = 0
        OMPVV_TEST_OFFLOADING

        OMPVV_WARNING("Collapse cannot be postively tested.")
        OMPVV_WARNING("Test only verifies that the collapse doesn't ")
        OMPVV_WARNING("collapse too many loops.")
        OMPVV_TEST_VERBOSE(test_collapse1() .ne. 0)
        OMPVV_TEST_VERBOSE(test_collapse2() .ne. 0)
        OMPVV_REPORT_AND_RETURN()

        CONTAINS
          INTEGER FUNCTION test_collapse1()
            INTEGER :: a(N, N), b(N, N + 1)
            INTEGER :: errors
            INTEGER :: temp_total
            INTEGER :: num_teams
            INTEGER :: x, y
            errors = 0

            DO x = 1, N
              b(x, 1) = 0
              DO y = 1, N
                a(x, y) = x + y
              END DO
            END DO

            !Collapse is only for one loop.  Second loop should be sequential

            !$omp target teams distribute map(to: a(1:N, 1:N)) map(tofrom: &
            !$omp& b(1:N, 1:N+1)) map(from: num_teams) collapse(1) &
            !$omp& num_teams(OMPVV_NUM_TEAMS_DEVICE)
            DO x = 1, N
              DO y = 1, N
                b(x, y + 1) = b(x, y) + a(x, y)
                IF (omp_get_team_num() .eq. 0) THEN
                   num_teams = omp_get_num_teams()
                END IF
              END DO
            END DO
            !$omp end target teams distribute

            DO x = 1, N
              temp_total = 0
              DO y = 1, N + 1
                OMPVV_TEST_AND_SET(errors, temp_total-b(x, y) .ne. 0)
                IF (y .ne. N + 1) THEN
                  temp_total = temp_total + a(x, y)
                END IF
              END DO
            END DO

            test_collapse1 = errors
          END FUNCTION test_collapse1

          INTEGER FUNCTION test_collapse2()
            INTEGER, ALLOCATABLE :: a(:,:,:), b(:,:,:)
            INTEGER :: errors
            INTEGER :: x, y, z
            INTEGER :: temp_total
            errors = 0

            ALLOCATE(a(N,N,N))
            ALLOCATE(b(N,N,N+1))

            DO x = 1, N
              DO y = 1, N
                b(x, y, 1) = 0
                DO z = 1, N
                  a(x, y, z) = x + y + z
                END DO
              END DO
            END DO

            !$omp target teams distribute map(to: a(1:N, 1:N, 1:N)) &
            !$omp& map(tofrom: b(1:N, 1:N, 1:N+1)) collapse(2) &
            !$omp& num_teams(OMPVV_NUM_TEAMS_DEVICE)
            DO x = 1, N
              DO y = 1, N
                DO z = 1, N
                  b(x, y, z+1) = b(x, y, z) + a(x, y, z)
                END DO
              END DO
            END DO
            !$omp end target teams distribute

            DO x = 1, N
              DO y = 1, N
                temp_total = 0
                DO z = 1, N
                  OMPVV_TEST_AND_SET(errors, (temp_total - b(x, y, z)) .ne. 0)
                  IF (z .ne. N + 1) THEN
                    temp_total = temp_total + a(x, y, z)
                  END IF
                END DO
              END DO
            END DO

            test_collapse2 = errors
          END FUNCTION test_collapse2
      END PROGRAM test_target_teams_distribute_collapse
