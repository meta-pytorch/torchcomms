!/===-- test_target_data_use_device_addr.F90 ------------------------------===//
!
! OpenMP API Version 5.0 Nov 2018
!
! This file is a test for the use_device_addr when used with the map
! clause with target data directive. This test uses a scalar whose value is
! modified on the device and tested on the host. List items that appear in
! a use_device_addr clause have the address of the corresponding object in
! the device data environment inside the construct. This test also tests
! that address conversions of use_device_addr clauses will occur as if
! performed after all variables are mapped according to those map clauses.
! Thanks to Tobias Burnus for providing the workaround for scalars in
! use_device_addr in Fortran OpenMP 5.0.
!
!/===----------------------------------------------------------------------===//

#include "ompvv.F90"

#define N 1024

PROGRAM test_target_data_use_device_addr
  USE iso_c_binding
  USE ompvv_lib
  USE omp_lib
  implicit none
  OMPVV_TEST_OFFLOADING

  OMPVV_TEST_VERBOSE(use_device_addr() .ne. 0)

  OMPVV_REPORT_AND_RETURN()

CONTAINS
  SUBROUTINE run_target_region(device_data, device_out)
    INTEGER,TARGET:: device_data(:), device_out(:)

    !$omp target is_device_ptr(device_data, device_out)
    device_out(1) = device_data(1)*N
    !$omp end target

  END SUBROUTINE run_target_region
  INTEGER FUNCTION use_device_addr()
    INTEGER:: errors, host_data
    INTEGER,TARGET:: device_data, device_out
    INTEGER,POINTER:: device_data_ptr(:), device_out_ptr(:)

    errors = 0
    device_data = N
    host_data = N*N

    !$omp target data map(to: device_data) map(from: device_out) use_device_addr(device_data, device_out)
    call c_f_pointer(c_loc(device_data), device_data_ptr, shape=[1])
    call c_f_pointer(c_loc(device_out), device_out_ptr, shape=[1])

    call run_target_region(device_data_ptr, device_out_ptr)
    !$omp end target data

    OMPVV_TEST_AND_SET_VERBOSE(errors, device_out .ne. host_data)

    use_device_addr = errors
  END FUNCTION use_device_addr
END PROGRAM test_target_data_use_device_addr
