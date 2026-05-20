!===--- test_target_teams_distribute_parallel_for_private.F90 ---------------===//
!
! OpenMP API Version 4.5 Nov 2015
! 
! This test check for a private variable within a pragma omp target teams 
! distribute parallel for. We use a private variable within a for loop and 
! assign it every iteration hoping that we won't get into data races. We do this 
! multiple times to improve testing. We assign a large number of threads and
! teams to try to increase parallelism and contention on the privatized variable
!
!===-------------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_parallel_for_private 
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

   OMPVV_TEST_OFFLOADING

   OMPVV_TEST_VERBOSE(target_teams_distribute_parallel_for_private() .ne. 0)

   OMPVV_REPORT_AND_RETURN()

CONTAINS
   INTEGER FUNCTION target_teams_distribute_parallel_for_private()
      INTEGER, DIMENSION(N) :: a, b, c, d, num_teams, num_threads
      INTEGER :: privatized, errors, i, j, warning_threads, warning_teams
      
      errors = 0
      j = 0

      DO i = 1, N
         a(i) = 1
         b(i) = i
         c(i) = 2*i
         d(i) = 0
         num_teams(i) = -1
         num_threads(i) = -1
      END DO

      !$omp target data map(to: a, b, c) map(from: d)
         !$omp target teams distribute parallel do private(privatized, i)&
         !$omp& num_threads(OMPVV_NUM_THREADS_DEVICE) num_teams(OMPVV_NUM_TEAMS_DEVICE)
         DO j = 1, N
            num_teams(j) = omp_get_num_teams()
            num_threads(j) = omp_get_num_threads()

            privatized = 0
            
            DO i = 1, (a(j) + b(j))
               privatized = privatized + 1
            END DO
            
            d(j) = c(j) * privatized
         END DO
      !$omp end target data

      warning_threads = 0
      warning_teams = 0

      DO i = 1, N
         OMPVV_TEST_AND_SET(errors, d(i) .ne. ((1 + i) * 2 * i))
         warning_teams = warning_teams + num_teams(i)
         warning_threads = warning_threads + num_threads(i)
      END DO

      OMPVV_WARNING_IF(warning_teams .eq. N, "There was a single team across the target region. Privatization cannot be tested at the teams level");
      OMPVV_WARNING_IF(warning_threads .eq. N, "All the parallel regions ran with a single thread. Privatization cannot be tested at the thread level");

      target_teams_distribute_parallel_for_private = errors
   END FUNCTION target_teams_distribute_parallel_for_private 
END PROGRAM test_target_teams_distribute_parallel_for_private
