!/===-- test_target_data_use_device_ptr.F90 -------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This file is a test for the use_device_ptr when used with the map clause
! with target data directive. This test uses arrays of size N whose values
! are modified on the device and tested in the host. Once the array has
! been mapped to the device, the use_device_ptr should be able to be used
! with the ptr to the array and subsequently modify values on the device.
! This test ensures address conversions of use_device_ptr clauses will
! occur as if performed after all variables are mapped according to those
! map clauses.
!
!/===----------------------------------------------------------------------===//
#include "ompvv.F90"

#define N 1024

PROGRAM test_target_data_use_device_ptr
  USE iso_fortran_env
  USE ompvv_lib
  USE omp_lib
  implicit none
  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(use_device_ptr() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  SUBROUTINE run_target_region(host_data, device_data)
    INTEGER :: host_data
    INTEGER,INTENT(in) :: device_data

    !$omp target data map(to: device_data) use_device_ptr(device_data)
    !$omp target is_device_ptr(device_data) map(tofrom: host_data)
    host_data = host_data + device_data
    !$omp end target
    !$omp end target data

  END SUBROUTINE run_target_region
  INTEGER FUNCTION use_device_ptr()
    INTEGER :: errors
    INTEGER :: scalar_device, scalar_host

    errors = 0
    scalar_device = N
    scalar_host = 0

    call run_target_region(scalar_host, scalar_device)

    OMPVV_TEST_AND_SET_VERBOSE(errors, scalar_host .ne. N)

    use_device_ptr = errors
  END FUNCTION use_device_ptr
END PROGRAM test_target_data_use_device_ptr
