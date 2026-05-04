!===--- test_target_teams_distribute_parallel_for_if_target_modifier.F90 -----===!
!
! OpenMP API Version 4.5 Nov 2015
!
! In this test we want to try to check if the if clause is working 
! when used with the combined construct target teams distribute
! parallel for and the target modifier is specified.
! To do this we check if offloading is working, if it is not, it won't
! be possible for us to tell if the test passed or not, since it depends
! on offloading capabilities.
!
! The if has a directive-name-modifier that specifies to which directive
! the if applies to (either both directives, to the target or to the parallel). 
! We create three tests, one for no directive, another one for the target 
! directive and another one for the parallel directive.
!
!===--------------------------------------------------------------------------===!

#include "ompvv.F90"

#define ATTEMPT_THRESHOLD 70
#define NUM_ATTEMPTS 100
#define N 1024

PROGRAM test_target_teams_distribute_parallel_for_if_target_modifier
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

   ! Offloading is checked in checkPreconditions() function

   OMPVV_TEST_VERBOSE(target_teams_distribute_if_target_modifier() .ne. 0)

   OMPVV_REPORT_AND_RETURN()

CONTAINS
   LOGICAL FUNCTION checkPreconditions()
      LOGICAL :: isOffloading
      OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)
      OMPVV_WARNING_IF(isOffloading .eqv. .false., "With offloading off, it is not possible to test if")
   checkPreconditions = isOffloading
   END FUNCTION checkPreconditions

   INTEGER FUNCTION target_teams_distribute_if_target_modifier()
      INTEGER :: attempt, errors, i, raiseWarning
      LOGICAL :: isOffloading
      INTEGER, DIMENSION(N) :: a, warning

      errors = 0

      isOffloading = checkPreconditions()

      DO i = 1, N
         a(i) = 1
         warning(i) = 0
      END DO

      DO attempt = 1, NUM_ATTEMPTS
         !$omp target teams distribute parallel do if(target: attempt .gt. ATTEMPT_THRESHOLD)&
         !$omp& map (tofrom: a) num_threads(OMPVV_NUM_THREADS_DEVICE)
            DO i = 1, N
               IF (omp_get_num_threads() .eq. 1) THEN
                  warning(i) = warning(i) + 1
               END IF
               IF (attempt .gt. ATTEMPT_THRESHOLD) THEN
                  IF ((isOffloading .eqv. .TRUE.) .and. (omp_is_initial_device() .eqv. .true.)) THEN
                     a(i) = a(i) + 10
                  END IF
               ELSE
                  IF (omp_is_initial_device() .eqv. .TRUE.) THEN
                     a(i) = a(i) + 1
                  ELSE 
                     a(i) = a(i) + 100
                  END IF
               END IF
            END DO
      END DO
      
      raiseWarning = 0
      DO i = 1, N
         OMPVV_TEST_AND_SET(errors, a(i) .ne. 1 + (ATTEMPT_THRESHOLD))
         IF (warning(i) .ne. 0) THEN
            raiseWarning = 1
         END IF
      END DO

      OMPVV_WARNING_IF(raiseWarning .ne. 0, "The number of threads was 1 even though we expected it to be more than 1. Not a compliance error in the specs")
      OMPVV_ERROR_IF(errors .ne. 0, "error in if(target: modifier). The execution was expected to occur in the device, but it happened in the host when if(false), or the other way around")
 
      target_teams_distribute_if_target_modifier = errors
      END FUNCTION target_teams_distribute_if_target_modifier
END PROGRAM test_target_teams_distribute_parallel_for_if_target_modifier
