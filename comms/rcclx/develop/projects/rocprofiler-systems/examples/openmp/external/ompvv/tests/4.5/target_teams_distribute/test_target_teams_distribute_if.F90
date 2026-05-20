!===--- test_target_teams_distribute_if.F90---------------------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the if clause to specify whether the target teams distribute
! directve should execute in the device or the hosts through the if clause of the 
! taget directive. The test uses omp_is_initial_device through the different 
! threads, assigning an expected value to a matrix according to 1) offloading is 
! enabled and working, 2) the if clause evaluates to true and the code executes in
! the offloading device, and 3) the if clause evaluates to false and the code executes
! in the offloading devices. 
!
! If not operating on a device, the test has a minimal test
! of the basic use of the if clause with both a true and a false parameter.
! However, the execution is identical to that of host operation in both cases.
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024
#define NUM_ATTEMPTS 100
#define ATTEMPT_THRESHOLD 70

PROGRAM test_target_teams_distribute_if
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  INTEGER :: errors
  OMPVV_TEST_OFFLOADING
  errors = 0

  OMPVV_TEST_VERBOSE(test_main() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_main()
    INTEGER,DIMENSION(N):: a
    INTEGER:: x, errors, attempt, sum
    LOGICAL:: isOffloading

    OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)
    OMPVV_WARNING_IF(.not. isOffloading, "With offloading off, it is not possible to test if")

    errors = 0

    DO x = 1, N
       a(x) = 1
    END DO
    
    DO attempt = 1, NUM_ATTEMPTS
       !$omp target teams distribute if(attempt .gt. ATTEMPT_THRESHOLD) map(tofrom: a(1:N))
       DO x = 1, N
          IF (attempt .gt. ATTEMPT_THRESHOLD) THEN
             IF (isOffloading .and. omp_is_initial_device()) THEN
                a(x) = a(x) + 10
             ELSE
                a(x) = a(x) + 0
             END IF
          ELSE
             IF (omp_is_initial_device()) THEN
                a(x) = a(x) + 1
             ELSE            
                a(x) = a(x) + 100
             END IF
          END IF
       END DO
    END DO
    
    DO x = 1, N
       IF (a(x) .ne. (1 + ATTEMPT_THRESHOLD)) THEN
          errors = errors + 1
       END IF
    END DO

    IF (errors .gt. 0) THEN
       DO x = 1, N
          sum = sum + a(x)
       END DO
       IF (sum .eq. N*(100*ATTEMPT_THRESHOLD + 1)) THEN
          OMPVV_ERROR("Error in if. The execution was expected to occur on the host, but it occurred on the device.")
       ELSE IF (sum .eq. N*(ATTEMPT_THRESHOLD + 10*(NUM_ATTEMPTS - ATTEMPT_THRESHOLD) + 1)) THEN
          OMPVV_ERROR("Error in if. The execution was expected to occur on the device, but it occurred on the host.")
       ELSE
          OMPVV_ERROR("Error in if. The execution occurred inconsistently on the host or on the device.")
       END IF
    END IF

    test_main = errors
  END FUNCTION test_main
END PROGRAM test_target_teams_distribute_if
