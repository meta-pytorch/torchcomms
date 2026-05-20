!===--- test_target_teams_distribute_parallel_for_firstprivate.F90 ----------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test check for a private variable within a pragma omp target teams 
! distribute parallel for that is initialized from the host through firstprivate
! clause. We use a private variable within a for loop and asign it every iteration
! hoping that we won't get into data races. We do this multiple times to improve
! testing.
!
!===-------------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_parallel_for_firstprivate
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

   OMPVV_TEST_VERBOSE(target_teams_distribute_parallel_for_firstprivate() .ne. 0)

   OMPVV_REPORT_AND_RETURN()

CONTAINS
   INTEGER FUNCTION target_teams_distribute_parallel_for_firstprivate()
      INTEGER, DIMENSION(N) :: a, b, c, d, num_teams, num_threads, team_num
      INTEGER :: firstized, privatized, errors, i, j, warning_threads

      firstized = 10
      errors = 0
      privatized = 0
      j = 0
       DO i = 1, N
         a(i) = 1
         b(i) = i
         c(i) = 2*i
         d(i) = 0
      END DO

      !$omp target data map(to: a, b, c)
         !$omp target teams distribute parallel do&
         !$omp& firstprivate(privatized, firstized, i)&
         !$omp& num_threads(OMPVV_NUM_THREADS_DEVICE) num_teams(OMPVV_NUM_TEAMS_DEVICE)
            DO j = 1, N
               num_teams(j) = omp_get_num_teams()
               num_threads(j) = omp_get_num_threads()
               team_num = omp_get_team_num()
               privatized = 0

               DO i = 1, (a(j) + b(j))
               privatized = privatized + 1
               END DO
               
               privatized = privatized + firstized
               d(j) = c(j) * privatized
            END DO
      !$omp end target data
      
      OMPVV_WARNING_IF(num_teams(1) .eq. 1, "Number of teams reported was 1, test cannot assert privatization across teams")
      warning_threads = 0
      DO i = 1, N
         IF (num_threads(i) .eq. 1) THEN
            warning_threads = warning_threads + 1
         END IF
         IF (i .gt. 1) THEN
            OMPVV_ERROR_IF(num_teams(i) .ne. num_teams(i-1), "Discrepancy in the reported number of teams across teams")
            IF ((team_num(i) .eq. team_num(i-1)) .and. (num_threads(i) .ne. num_threads(i-1))) THEN
               OMPVV_ERROR("Discrepancy in the reported number of threads inside a single team")
            END IF
         END IF
      END DO

      OMPVV_WARNING_IF(warning_threads .eq. N, "Number of threads was 1 for all teams test cannot assert privatization across teams")
      
      DO i = 1, N
         OMPVV_TEST_AND_SET(errors, d(i) .ne. (10 + 1+ i) * (2 * i))
      END DO

      target_teams_distribute_parallel_for_firstprivate = errors
   END FUNCTION target_teams_distribute_parallel_for_firstprivate
END PROGRAM test_target_teams_distribute_parallel_for_firstprivate   

