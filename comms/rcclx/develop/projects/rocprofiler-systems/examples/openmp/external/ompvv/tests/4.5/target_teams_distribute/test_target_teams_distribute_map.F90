!===--- test_target_teams_distribute_map.F90--------------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the map clause on a target teams distribute clause to test that
! the indicated data is maped on the target in the fashion that is indicated
! with the map-type.  Each map-type that is valid in this situation (to, from,
! alloc, and tofrom) are each used and tested.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_map
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING
  OMPVV_TEST_SHARED_ENVIRONMENT
  OMPVV_TEST_VERBOSE(test_map_to() .ne. 0)
  OMPVV_TEST_VERBOSE(test_map_from() .ne. 0)
  OMPVV_TEST_VERBOSE(test_map_alloc() .ne. 0)
  OMPVV_TEST_VERBOSE(test_map_tofrom() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_map_to()
    INTEGER,DIMENSION(N):: a, b
    INTEGER:: errors, x
    errors = 0

    DO x = 1, N
       a(x) = x
       b(x) = 0
    END DO

    !$omp target enter data map(alloc: b(1:N))
    !$omp target teams distribute map(to: a(1:N), b(1:N))
    DO x = 1, N
       b(x) = a(x)
    END DO
    !$omp target exit data map(from: b(1:N))

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, a(x) .ne. b(x))
       IF (a(x) .ne. b(x)) THEN
          exit
       END IF
    END DO

    test_map_to = errors
  END FUNCTION test_map_to

  INTEGER FUNCTION test_map_from()
    INTEGER,DIMENSION(N):: a, b
    INTEGER:: errors, x
    errors = 0

    DO x = 1, N
       a(x) = x
       b(x) = 0
    END DO

    !$omp target enter data map(to: a(1:N))
    !$omp target teams distribute map(from: b(1:N))
    DO x = 1, N
       b(x) = a(x)
    END DO
    !$omp target exit data map(delete: a(1:N))

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, a(x) .ne. b(x))
       IF (a(x) .ne. b(x)) THEN
          exit
       END IF
    END DO

    test_map_from = errors
  END FUNCTION test_map_from

  INTEGER FUNCTION test_map_alloc()
    INTEGER,DIMENSION(N):: a, b, c
    INTEGER:: errors, x
    errors = 0

    DO x = 1, N
       a(x) = x
       b(x) = 0
       c(x) = 0
    END DO

    !$omp target enter data map(to: a(1:N)) map(alloc: b(1:N))
    !$omp target teams distribute map(alloc: a(1:N), b(1:N), c(1:N))
    DO x = 1, N
       c(x) = a(x)
       b(x) = c(x)
    END DO
    !$omp target exit data map(delete: a(1:N)) map(from: b(1:N))

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, a(x) .ne. b(x))
       IF (a(x) .ne. b(x)) THEN
          exit
       END IF
    END DO

    test_map_alloc = errors
  END FUNCTION test_map_alloc

  INTEGER FUNCTION test_map_tofrom()
    INTEGER,DIMENSION(N):: a, b
    INTEGER:: errors, x
    errors = 0

    DO x = 1, N
       a(x) = x
       b(x) = x
    END DO

    !$omp target teams distribute map(tofrom: a(1:N), b(1:N))
    DO x = 1, N
       b(x) = b(x) + a(x)
    END DO

    DO x = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, b(x) .ne. 2*x)
       IF (b(x) .ne. 2*x) THEN
          exit
       END IF
    END DO

    test_map_tofrom = errors
  END FUNCTION test_map_tofrom
END PROGRAM test_target_teams_distribute_map
