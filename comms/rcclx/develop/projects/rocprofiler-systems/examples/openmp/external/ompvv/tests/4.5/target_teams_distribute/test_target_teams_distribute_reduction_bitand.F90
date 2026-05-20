!===--- test_target_teams_distribute_reduction_bitand.F90-------------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test uses the reduction clause on a target teams distribute
! directive, testing, for the bitand operator, that the variable in the
! reduction clause is properly reduced.
!
!//===----------------------------------------------------------------------===//
#include "ompvv.F90"

#define N 1024
#define THRESHOLD 512

PROGRAM test_target_teams_distribute_device
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_bitand() .ne. 0)

  OMPVV_REPORT_AND_RETURN()
CONTAINS
  INTEGER FUNCTION test_bitand()
    INTEGER,DIMENSION(N):: a
    REAL(8),DIMENSION(N, 32):: randoms
    INTEGER:: result, host_result, x, y, errors, itr_count
    LOGICAL:: tested_true, tested_false
    REAL(8):: false_margin
    result = 0
    errors = 0

    tested_true = .FALSE.
    tested_false = .FALSE.
    itr_count = 0

    CALL RANDOM_SEED()
    false_margin = exp(log(.5) / N)

    DO WHILE ((.NOT. tested_true .OR. .NOT. tested_false) &
         & .AND. (itr_count .lt. THRESHOLD))
       CALL RANDOM_NUMBER(randoms)
       DO x = 1, N
          a(x) = 0
          DO y = 1, 16
             IF (randoms(x, y) .lt. false_margin) THEN
                a(x) = a(x) + (2**y)
                tested_true = .TRUE.
             ELSE
                tested_false = .TRUE.
             END IF
          END DO
       END DO

       result = 0
       host_result = 0
       DO y = 1, 16
          result = result + (2**y)
          host_result = host_result + (2**y)
       END DO

       DO x = 1, N
          host_result = iand(a(x), host_result)
       END DO

       !$omp target teams distribute defaultmap(tofrom:scalar) &
       !$omp& reduction(iand:result)
       DO x = 1, N
          result = iand(a(x), result)
       END DO

       OMPVV_TEST_AND_SET_VERBOSE(errors, result .ne. host_result)

       itr_count = itr_count + 1
    END DO

    OMPVV_WARNING_IF(.NOT. tested_true, "Did not test true case")
    OMPVV_WARNING_IF(.NOT. tested_false, "Did not test false case")

    test_bitand = errors
  END FUNCTION test_bitand
END PROGRAM test_target_teams_distribute_device
