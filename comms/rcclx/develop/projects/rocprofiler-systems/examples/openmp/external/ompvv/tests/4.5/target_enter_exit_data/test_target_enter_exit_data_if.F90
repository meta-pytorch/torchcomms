!===---- test_target_enter_exit_data.F90 - if clause of target enter/exit data ------===//
! 
! OpenMP API Version 4.5 Nov 2015
!
! The if clause for the target enter and target exit data defines if the
! data mapping is executed or not. This test check if the if clause
! works for both enter and exit data 
!
!===----------------------------------------------------------------------===//

#include "ompvv.F90" 

#define SIZE_ARRAY 1024
#define THRESHOLD 512

      PROGRAM tests_target_enter_exit_data_if
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none
       
         LOGICAL :: isOffloading, isSharedEnv
         INTEGER :: a(1:SIZE_ARRAY)
         INTEGER :: b(1:SIZE_ARRAY)
         INTEGER :: c(1:SIZE_ARRAY)
         INTEGER :: i, j, s
         CHARACTER(len=300) :: infoMessage
        
         OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)
         OMPVV_TEST_AND_SET_SHARED_ENVIRONMENT(isSharedEnv)
         IF (isSharedEnv) THEN 
           WRITE(infoMessage, *) " working on a shared memory &
           & environment. Testing data mapping not possible"
           OMPVV_WARNING(infoMessage)
           OMPVV_REPORT_AND_RETURN()
         END IF
         IF (.NOT. isOffloading) THEN 
           WRITE(infoMessage, *) "Offloading is off. Might not be&
           & possible to test if clause correctly "
           OMPVV_WARNING(infoMessage)
          OMPVV_REPORT_AND_RETURN()
         END IF

         ! We only test if it is offloading and it is not shared 
         ! memory. As wiehtout these two conditions would not 
         ! be possible to test data transfers
         OMPVV_TEST_VERBOSE(test_target_enter_exit_data_if() .NE. 0) 
         
         OMPVV_REPORT_AND_RETURN()

       CONTAINS 
         INTEGER FUNCTION test_target_enter_exit_data_if()
           ! check multiple sizes. 
           DO s = 256, SIZE_ARRAY, 256
             ! a, b and c arrays initialization
             a(:) = (/ (SIZE_ARRAY - i, i = 1, SIZE_ARRAY) /)
             b(:) = (/ (i - 1, i = 1, SIZE_ARRAY) /) 
             c(:) = -1
             !$omp target enter data if(s > THRESHOLD) &
             !$omp map(to: a(1:s), b(1:s)) 

             ! change the value at the host
             a(:) = 0
             b(:) = 0
                               
             ! If the target enter data did not map, then the 
             ! map(to...) will get the updated value, but if it did
             ! map then the map(to...) should be a no_op

             !$omp target map(to: a(1:s), b(1:s)) &
             !$omp map(tofrom: c(1:s)) map(tofrom: s)
               c(1:s) = (/ (a(j) + b(j) + 1, j = 1, s) /)
               a(1:s) = 20
               b(1:s) = 20
             !$omp end target

             !$omp target exit data if(s > THRESHOLD) &
             !$omp map(from: a(1:s), b(1:s)) 

             ! Checking results
             IF (s > THRESHOLD) THEN
               ! check for all values that should have been mapped 
               ! on the target enter data
               OMPVV_TEST_VERBOSE(ANY(c(:s) /= SIZE_ARRAY))
               OMPVV_TEST_VERBOSE(ANY(a(:s) /= 20))
               OMPVV_TEST_VERBOSE(ANY(b(:s) /= 20))
             ELSE
               ! checking for values that should have been 
               ! mapped by the target region and not the target enter
               ! data
               OMPVV_TEST_VERBOSE(ANY(c(:s) /= 1))
               OMPVV_TEST_VERBOSE(ANY(a(:s) /= 0))
               OMPVV_TEST_VERBOSE(ANY(b(:s) /= 0))
               WRITE(infoMessage, *) "s = ", s
             END IF 


           END DO ! s

           OMPVV_GET_ERRORS(test_target_enter_exit_data_if)
         END FUNCTION test_target_enter_exit_data_if
           
      END PROGRAM tests_target_enter_exit_data_if
