!===--- test_metadirective_arch_is_nvidia.F90 -------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! Test for metadirectives based on OpenMP 5.0 examples metadirective.1-3.c
!
!//===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_metadirective_arch_is_nvidia
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none

  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(test_metadirective1() .NE. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  INTEGER FUNCTION test_metadirective1()
    INTEGER,DIMENSION(N):: v1, v2, v3
    INTEGER:: target_device_num, host_device_num, default_device
    INTEGER:: errors, i, j

    errors = 0

    DO i = 1, N
       v1(i) = i+1
       v2(i) = -(i+1)
    END DO

    host_device_num = omp_get_initial_device()
    
    default_device = omp_get_default_device()

    !$omp target map(to:v1,v2) map(from:v3,target_device_num) device(default_device)
    !$omp begin metadirective when(device={arch("nvptx")}: teams distribute parallel do) default(parallel do)
    DO i = 1, N
    !$omp atomic write
       target_device_num = omp_get_device_num()
    !$omp end atomic
       v3(i) = v1(i) * v2(i)
    END DO
    !$omp end metadirective
    !$omp end target

    OMPVV_TEST_AND_SET(errors, host_device_num .EQ. target_device_num)
    OMPVV_ERROR_IF(host_device_num .EQ. target_device_num, "Device number that executes target region is the same as the device number on the host")

    DO i = 1, N
       OMPVV_TEST_AND_SET_VERBOSE(errors, v3(i) .NE. v1(i) * v2(i))
    END DO

    test_metadirective1 = errors
  END FUNCTION test_metadirective1

END PROGRAM test_metadirective_arch_is_nvidia
