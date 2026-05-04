!===--- test_target_teams_distribute_default_none.F90-----------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This tests uses the default(none) clause on a target teams distribute test.
! The test aims to validate that all values will not have default data sharing
! attributes.
!
!===------------------------------------------------------------------------===//


#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_default_none
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  CHARACTER(len=300):: infoMessage

  OMPVV_TEST_OFFLOADING

  WRITE(infoMessage, *) "Test only uses the default(none) clause and&
       & does not guarantee that the default(none) is enforced."
  OMPVV_WARNING(infoMessage)

  OMPVV_TEST_VERBOSE(default_none1() .ne. 0)  !check read/write to private
  OMPVV_TEST_VERBOSE(default_none2() .ne. 0)  !check atomic update of shared
  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION default_none1()
    INTEGER :: a(N), b(N), c(N), d(N)
    INTEGER :: privatized, x, y, errors
    errors = 0

    DO x = 1, N
       a(x) = 1
       b(x) = x
       c(x) = 2*x
       d(x) = 0
    END DO

    !$omp target data map(from: d(1:N)) map(to: a(1:N), b(1:N), c(1:N))
    !$omp target teams distribute default(none) shared(a, b, c, d) &
    !$omp& private(x, y, privatized) num_teams(OMPVV_NUM_TEAMS_DEVICE)
    DO x = 1, N
       privatized = 0
       DO y = 1, a(x) + b(x)
          privatized = privatized + 1
       END DO
       d(x) = c(x) * privatized
    END DO
    !$omp end target data

    DO x = 1, N
       OMPVV_TEST_AND_SET(errors, (d(x) .ne. (1 + x)*2*x))
    END DO

    default_none1 = errors
  END FUNCTION default_none1

  INTEGER FUNCTION default_none2()
    INTEGER :: a(N)
    INTEGER :: share, x, errors
    errors = 0
    share = 0

    DO x = 1, N
       a(x) = x
    END DO

    !$omp target data map(to: a(1:N)) map(tofrom: share)
    !$omp target teams distribute default(none) private(x) &
    !$omp& shared(share, a) defaultmap(tofrom:scalar) num_teams(OMPVV_NUM_TEAMS_DEVICE)
    DO x = 1, N
       !$omp atomic
       share = share + a(x)
    END DO
    !$omp end target data

    DO x = 1, N
       share = share - x
    END DO

    OMPVV_TEST_AND_SET(errors, (share .ne. 0))
    default_none2 = errors
  END FUNCTION default_none2
END PROGRAM test_target_teams_distribute_default_none
