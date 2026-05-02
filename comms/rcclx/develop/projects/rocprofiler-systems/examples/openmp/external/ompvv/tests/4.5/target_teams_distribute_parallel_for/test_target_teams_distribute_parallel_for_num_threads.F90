!===-- test_target_teams_distribute_parallel_for_num_threads.F90 ------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! Test to check the num_threads clause. This clause changes the upper limit
! of the number of threads inside the parallel region, when used with the 
! combined construct target teams distribute parallel do.
!
!===-------------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_teams_distribute_parallel_for_num_threads
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

   OMPVV_TEST_VERBOSE(target_teams_distribute_parallel_for_num_threads() .ne. 0)

   OMPVV_REPORT_AND_RETURN()

CONTAINS
   INTEGER FUNCTION target_teams_distribute_parallel_for_num_threads()
      INTEGER :: errors, i, nt, oneThreadWarning, difNumWarning, prevNumThreads 
      INTEGER, DIMENSION(4) :: tested_num_threads = (/1, 10, 100, 10000/)
      INTEGER, DIMENSION(N) :: num_threads
      CHARACTER(len=400) :: difNumMessage, oneThreadMessage
      WRITE(difNumMessage, *) "When testing num_threads, the &
       &actual number of threads was different. This is not a &
       &compliance error with the specs."
     
      WRITE(oneThreadMessage, *) "The number of threads was &
      &always one, regardless of the num_threads clause. This is not a &
      &compliance error in the specs"
       
     

      errors = 0
      oneThreadWarning = 0

      DO nt = 1, 4
       
         Print *, "Testing for num_threads(", tested_num_threads(nt), ")" 

         ! Initializing the num_threads array
         DO i = 1, N
            num_threads(i) = -1
         END DO


         !$omp target teams distribute parallel do map(tofrom: num_threads)&
         !$omp& num_threads(tested_num_threads(nt))
            DO i = 1, N 
               num_threads(i) = omp_get_num_threads()
            END DO
          
            difNumWarning = 0
            prevNumThreads = -1
    
            DO i = 1, N
               OMPVV_INFOMSG_IF(prevNumThreads .ne. num_threads(i), "Improper number of threads reported")
               prevNumThreads = num_threads(i)
               ! If the number of threads is larger than the specified number,
               ! throw an error
               OMPVV_TEST_AND_SET(errors, (num_threads(i) .le. 0) .or. (num_threads(i) > tested_num_threads(nt)))
               IF (tested_num_threads(nt) .ne. num_threads(i)) THEN
                  difNumWarning = 1
               END IF
               IF ((tested_num_threads(nt) .ne. 1) .and. (num_threads(i) .eq. 1)) THEN
                  oneThreadWarning = oneThreadWarning + 1
               END IF
            END DO
            OMPVV_WARNING_IF(difNumWarning .ne. 0, difNumMessage)
      END DO
      OMPVV_WARNING_IF(oneThreadWarning .eq. (4*N), oneThreadMessage)
 
      target_teams_distribute_parallel_for_num_threads = errors
   END FUNCTION target_teams_distribute_parallel_for_num_threads
END PROGRAM test_target_teams_distribute_parallel_for_num_threads
   
