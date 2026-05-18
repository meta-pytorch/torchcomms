!===-- test_target_teams_distribute_parallel_for_num_teams.F90 ------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! Test to check the num_teams clause. This clause changes the upper limit of 
! the number of teams inside the target teams region. 
!
!===-------------------------------------------------------------------------===//

#include "ompvv.F90"
#define N 1024

PROGRAM test_target_teams_distribute_parallel_for_num_teams
   USE iso_fortran_env
   USE ompvv_lib
   USE omp_lib
   implicit none

   OMPVV_TEST_VERBOSE(target_teams_distribute_parallel_for_num_teams() .ne. 0)

   OMPVV_REPORT_AND_RETURN()

CONTAINS
   INTEGER FUNCTION target_teams_distribute_parallel_for_num_teams()
      INTEGER :: errors, i, nt, raiseWarningOneTeam, raiseWarningDifNum, prevNumTeams 
      INTEGER, DIMENSION(4) :: tested_num_teams = (/1, 10, 100, 10000/)
      INTEGER, DIMENSION(N) :: num_teams
      CHARACTER(len=400) :: numTeamsMsg, difNumReportedMsg, difNumTeamsMsg
      errors = 0
      i = 0
      raiseWarningOneTeam = 0

      DO nt = 1, 4
         WRITE(numTeamsMsg, *) "Testing for num_teams(", tested_num_teams(nt), ")" 
         OMPVV_INFOMSG(numTeamsMsg)

         ! initialize the num_teams array
         DO i = 1, N
            num_teams(i) = -1
         END DO

         !$omp target teams distribute parallel do map(tofrom: num_teams) num_teams(tested_num_teams(nt))
            DO i = 1, N
               num_teams(i) = omp_get_num_teams()
            END DO

         raiseWarningDifNum = 0
         prevNumTeams = -1

         DO i = 1, N
            WRITE(difNumReportedMsg, *) num_teams(i), "teams reported"
            OMPVV_INFOMSG_IF(prevNumTeams .ne. num_teams(i), difNumReportedMsg)
            prevNumTeams = num_teams(i)
            OMPVV_TEST_AND_SET(errors, num_teams(i) .le. 0 .or. (num_teams(i) .gt. tested_num_teams(nt)))
            IF (num_teams(i) .ne. tested_num_teams(nt)) THEN
               raiseWarningDifNum = 1
            END IF
            IF (num_teams(i) == 1) THEN
               raiseWarningOneTeam = raiseWarningOneTeam + 1
            END IF
         END DO

         !We want to raise a warning when the number of teams does not match the desired value
         WRITE(difNumTeamsMsg, *) "When testing for num_teams(", tested_num_teams(nt), "), the actual &
                 &number of teams was different. Not a compliance error with the specification."
         OMPVV_WARNING_IF(raiseWarningDifNum .ne. 0, difNumTeamsMsg) 
      END DO

      OMPVV_WARNING_IF(raiseWarningOneTeam .eq. 4*N, "The num_teams clause always resulted in a single team. Although &
              &this is compliant with the specification, it is not expected.")

      target_teams_distribute_parallel_for_num_teams = errors
   END FUNCTION target_teams_distribute_parallel_for_num_teams
END PROGRAM test_target_teams_distribute_parallel_for_num_teams
     
