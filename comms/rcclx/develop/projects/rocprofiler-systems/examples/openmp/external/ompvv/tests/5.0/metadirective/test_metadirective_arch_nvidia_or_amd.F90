!===--- test_metadirective_arch_nvidia_or_amd.F90 --------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! Test for metadirectives based on OpenMP 5.0 examples metadirective.1-3.c
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_metadirective_arch_nvidia_or_amd
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_metadirective2() .NE. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_metadirective2()
    INTEGER,DIMENSION(N):: a
    INTEGER:: device_num
    LOGICAL:: initial_device
    INTEGER:: errors, i

    errors = 0

    DO i = 1, N
       a(i) = 0
    END DO

    device_num = 0
    DO WHILE( (device_num .EQ. 0) .OR. (device_num .LT. omp_get_num_devices()) )
       !$omp target map(from:initial_device) device(device_num)
       !$omp begin metadirective when(implementation={vendor(nvidia)}: teams num_teams(512) thread_limit(32)) &
       !$omp when(implementation={vendor(amd)}: teams num_teams(512) thread_limit(64)) default(teams)
       !$omp distribute parallel do
       DO i = 1, N
       !$omp atomic write
          initial_device = omp_is_initial_device()
       !$omp end atomic
          a(i) = i
       END DO
       !$omp end distribute parallel do
       !$omp end metadirective
       !$omp end target
       device_num = device_num + 1
    END DO

    OMPVV_TEST_AND_SET_VERBOSE(errors, initial_device)
    OMPVV_ERROR_IF(initial_device, "NVIDIA and AMD architecture not available, ran on host")

    DO i = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, a(i) .NE. i)
    END DO

    test_metadirective2 = errors
  END FUNCTION test_metadirective2

END PROGRAM test_metadirective_arch_nvidia_or_amd
