!===--- test_target_teams_distribute_thread_limit.F90------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the thread_limit clause on a target teams distribute directive to
! indicate a requested number of threads to execute the teams distribute region.
! The specifications indicate that the number of threads that are given can be any
! number that is equal to or less than the indicated value.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_threads() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_threads()
    INTEGER:: num_threads, errors, x
    errors = 0
    num_threads = -9

    !$omp target teams distribute thread_limit(4) map(from: num_threads)
    DO x = 1, N
       IF (omp_get_team_num() .eq. 0) THEN
               !$omp parallel
               IF (omp_get_thread_num() .eq. 0) THEN
                       num_threads = omp_get_num_threads()
               END IF
               !$omp end parallel
       END IF
    END DO

    OMPVV_TEST_AND_SET(errors, num_threads .gt. 4)

    test_threads = errors
  END FUNCTION test_threads
END PROGRAM test_target_teams_distribute_device
