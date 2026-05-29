!===--- test_target_teams_distribute_firstprivate.F90-----------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the firstprivate clause and tests it in two separate parts.
! The test first tests the privatization of the firstprivatized scalars and
! then separately tests the proper initialization of them separately
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_firstprivate
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  INTEGER :: errors
  OMPVV_TEST_OFFLOADING
  errors = 0

  OMPVV_TEST_VERBOSE(test_firstprivate_private() .ne. 0)
  OMPVV_TEST_VERBOSE(test_firstprivate_first() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_firstprivate_private()
    INTEGER:: errors, x, y, z, privatized
    INTEGER,DIMENSION(N):: a, b, c, d, num_teams
    INTEGER,DIMENSION(10):: privatized_array

    errors = 0
    privatized = 0

    DO x = 1, N
       a(x) = 1
       b(x) = x
       c(x) = 2 * x
       d(x) = 0
       num_teams(x) = -1
    END DO

    DO x = 1, 10
       privatized_array(x) = 0
    END DO

    !$omp target data map(from: d(1:N)) map(to: a(1:N), b(1:N), c(1:N))
    !$omp target teams distribute firstprivate(privatized, privatized_array)&
    !$omp& map(alloc: a(1:N), b(1:N), c(1:N), d(1:N)) num_teams(OMPVV_NUM_TEAMS_DEVICE)
    DO x = 1, N
       num_teams(x) = omp_get_num_teams()
       DO y = 1, a(x) + b(x)
          privatized = privatized + 1
          DO z = 1, 10
             privatized_array(z) = privatized_array(z) + 1
          END DO
       END DO
       d(x) = c(x) * privatized
       DO z = 1, 10
          d(x) = d(x) + privatized_array(z)
       END DO
       privatized = 0
       DO z = 1, 10
          privatized_array(z) = 0
       END DO
    END DO
    !$omp end target data

    DO x = 1, N
       IF (d(x) .ne. 10*(1 + x) + (1 + x)*2*x) THEN
          errors = errors + 1
       END IF
       OMPVV_WARNING_IF(num_teams(x) .eq. 1, "Did not create enough teams to check for potential data races.")
    END DO

    test_firstprivate_private = errors
  END FUNCTION test_firstprivate_private
  INTEGER FUNCTION test_firstprivate_first()
    INTEGER:: errors, x, privatized
    INTEGER,DIMENSION(N):: a, b, c, d, num_teams
    INTEGER,DIMENSION(10):: privatized_array

    errors = 0
    privatized = 1

    DO x = 1, N
       a(x) = 1
       b(x) = x
       c(x) = 2 * x
       d(x) = 0
       num_teams(x) = -1
    END DO

    DO x = 1, 10
       privatized_array(x) = x
    END DO

    !$omp target data map(from: d(1:N)) map(to: a(1:N), b(1:N), c(1:N))
    !$omp target teams distribute firstprivate(privatized_array, privatized) &
    !$omp map(alloc: a(1:N), b(1:N), c(1:N), d(1:N)) num_teams(OMPVV_NUM_TEAMS_DEVICE)
    DO x = 1, N
       num_teams(x) = omp_get_num_teams()
       d(x) = a(x) + b(x) + c(x) + privatized_array(MOD(x, 10)+1) + privatized
    END DO
    !$omp end target data

    DO x = 1, N
       IF (d(x) .ne. 2 + 3*x + MOD(x, 10)+1) THEN
          errors = errors + 1
       END IF
       OMPVV_WARNING_IF(num_teams(x) .eq. 1, "Did not create enough teams to check for potential data races.")
    END DO

    test_firstprivate_first = errors
  END FUNCTION test_firstprivate_first
END PROGRAM test_target_teams_distribute_firstprivate
