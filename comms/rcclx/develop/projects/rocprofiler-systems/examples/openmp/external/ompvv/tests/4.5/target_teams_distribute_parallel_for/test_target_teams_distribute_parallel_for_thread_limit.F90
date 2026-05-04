!===--- test_target_teams_distribute_parallel_for_thread_limit.F90 ----------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! Test to check the thread_limit clause. This clause changes the upper limit
! of the number of threads inside each of the contention groups created in
! the teams region. This upper limit is different to the num_threads. 
!
!===-------------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_parallel_for_thread_limit
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

   OMPVV_TEST_OFFLOADING
   OMPVV_TEST_VERBOSE(target_teams_distribute_parallel_for_thread_limit() .ne. 0)

   OMPVV_REPORT_AND_RETURN()

CONTAINS 
   INTEGER FUNCTION target_teams_distribute_parallel_for_thread_limit()
      INTEGER :: errors, i, nt, tl, prevThreadLimit
      INTEGER, DIMENSION(4) :: tested_num_threads = (/1, 10, 100, 10000/)
      INTEGER, DIMENSION(4) :: tested_thread_limit = (/1, 10, 100, 10000/)
      INTEGER, DIMENSION(N) :: num_threads, thread_limit
      CHARACTER(len=400) :: msgHelper1, msgHelper2
      errors = 0

      ! Testing multiple num_threads and thread_limits values from 1 to large num.
      ! The number of threads should never be larger than the thread limit

      DO nt = 1, 4 
         DO tl = 1, 4
            WRITE(msgHelper1, *) "Testing thread limit (",tested_thread_limit(tl), ") num_threads(", tested_num_threads(nt), ") clauses."
            OMPVV_INFOMSG(msgHelper1)
            DO i = 1, N
               num_threads(i) = -1
               thread_limit(i) = -1
            END DO

            !$omp target teams distribute parallel do map(tofrom:num_threads)&
            !$omp& num_threads(tested_num_threads(nt)) thread_limit(tested_thread_limit(tl))
               DO i = 1, N
                  num_threads(i) = omp_get_num_threads()
                  thread_limit(i) = omp_get_thread_limit()
               END DO
            
               prevThreadLimit = -1
 
               DO i = 1, N
                  WRITE(msgHelper2, *) "  reported thread limit =", thread_limit(i)
                  OMPVV_INFOMSG_IF(prevThreadLimit .ne. thread_limit(i), msgHelper2)
                  prevThreadLimit = thread_limit(i)
                  OMPVV_TEST_AND_SET_VERBOSE(errors, (thread_limit(i) .gt. tested_thread_limit(tl)) .or. (thread_limit(i) .le. 0))
                  OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads(i) .gt. tested_thread_limit(tl))
                  OMPVV_TEST_AND_SET_VERBOSE(errors, num_threads(i) .gt. tested_num_threads(nt))
               END DO
         END DO
      END DO
      
      target_teams_distribute_parallel_for_thread_limit = errors
   END FUNCTION target_teams_distribute_parallel_for_thread_limit 
END PROGRAM test_target_teams_distribute_parallel_for_thread_limit
