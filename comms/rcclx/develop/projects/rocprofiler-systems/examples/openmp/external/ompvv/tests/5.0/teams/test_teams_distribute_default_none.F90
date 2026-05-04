!===--- test_teams_distribute_default_none.F90 -----------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This tests uses the default(none) clause on a target teams distribute
! test.
! The test aims to validate that all values will not have default data
! sharing
! attributes.
!
!===------------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 128

PROGRAM test_teams_distribute_default_none
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
    INTEGER :: privatized, x, y, errors, num_teams, share
    errors = 0
    share = 0

    DO x = 1, N
       a(x) = 1
       b(x) = x
       c(x) = 2*x
       d(x) = 0
    END DO

    !$omp teams distribute num_teams(OMPVV_NUM_TEAMS_DEVICE) default(none) shared(a,b,c,d,num_teams) private(privatized)
    DO x = 1, N
       privatized = 0
       DO y = 1, a(x) + b(x)
          privatized = privatized + 1
       END DO
       d(x) = c(x) * privatized
       IF (omp_get_team_num() .eq. 0) THEN
          num_teams = omp_get_num_teams()
       END IF
    END DO

    OMPVV_WARNING_IF(num_teams .eq. 1, "The number of teams was 1, is not a specification error but we could not guarantee parallelism of teams.")

    DO x = 1, N
       OMPVV_TEST_AND_SET(errors, (d(x) .ne. (1 + x)*2*x))
    END DO

    default_none1 = errors
  END FUNCTION default_none1

  INTEGER FUNCTION default_none2()
     INTEGER :: g(N)
     INTEGER :: share, x, num_teams, errors
     errors = 0
     share = 0

     DO x = 1, N
        g(x) = x
     END DO

     !$omp teams distribute num_teams(OMPVV_NUM_TEAMS_DEVICE) default(none) shared(share,g,num_teams)
     DO x = 1, N
        !$omp atomic update 
        share = share + g(x)
        !$omp end atomic
        IF (omp_get_team_num() .eq. 0) THEN
           num_teams = omp_get_num_teams()
        END IF
     END DO

     OMPVV_WARNING_IF(num_teams .eq. 1, "The number of teams was 1. This is not a specification error but we could not guarantee parallelism of teams.")

     DO x = 1, N
        share = share - x;
     END DO

     OMPVV_TEST_AND_SET(errors, (share .ne. 0))
     default_none2 = errors
   END FUNCTION default_none2
END PROGRAM test_teams_distribute_default_none
