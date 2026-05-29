!===---- test_target_data_map_set_default_device.F90 - set_default_device() API --===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test check if by calling the set_default_device the target_data
! is mapped to the default device selected by this API call
!
!===-------------------------------------------------------------------------===//
#include "ompvv.F90"

#define N 5000

      PROGRAM test_target_data_map_set_default_device
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none
        LOGICAL :: isOffloading
        INTEGER :: num_dev
        CHARACTER(len=500) :: msgHelper

        OMPVV_TEST_AND_SET_OFFLOADING(isOffloading)

        ! Reporting the number of devices
        num_dev = omp_get_num_devices()
        WRITE(msgHelper, '(A,I0)') "number of devices = ", num_dev
        OMPVV_INFOMSG(msgHelper)

        ! Warning if only one device
        WRITE(msgHelper, *) "The number of devices is 1, this test &
        &is inconclussive"
        OMPVV_WARNING_IF(num_dev == 1, msgHelper)

        !starting the tests
        OMPVV_TEST_VERBOSE(test_set_default_device() .NE. 0)

        OMPVV_REPORT_AND_RETURN()

        CONTAINS
          ! Testing set default device API
          INTEGER FUNCTION test_set_default_device()
            INTEGER :: def_dev, dev_data, dev_comp
            INTEGER, dimension(N) :: anArray

            OMPVV_INFOMSG("using set_default_device")

            def_dev = omp_get_default_device()
            WRITE(msgHelper, '(A,I0)') "default device = ", def_dev
            OMPVV_INFOMSG(msgHelper)

            ! Initialize the array
            anArray(:) = 1

            ! Iterate over all the devices
            DO dev_data = 0, (num_dev - 1)
              ! set default device
              CALL omp_set_default_device(dev_data)
              !$omp target data map(tofrom: anArray(1:N))

                ! Iterate over all the devices doing comp
                ! to guarantee that the right data env is
                ! being copied back and forth
                DO dev_comp = 0, (num_dev - 1)
                  !$omp target map(alloc: anArray(1:N)) &
                  !$omp device(dev_comp)

                    ! Increase only in the current tested device
                    IF (dev_comp == dev_data) THEN
                      anArray(1:N) = anArray(1:N) + 1
                    ELSE
                      ! This should not affect result as it should happen in
                      ! other devices
                      anArray(1:N) = anArray(1:N) - 10
                    END IF

                  !$omp end target
                END DO ! dev_comp
              !$omp end target data
            END DO ! dev_data

            OMPVV_TEST_VERBOSE(ANY(anArray /= num_dev + 1))

            ! return the default device to the
            ! original default
            CALL omp_set_default_device(def_dev)

            OMPVV_GET_ERRORS(test_set_default_device)

          END FUNCTION test_set_default_device
      END PROGRAM test_target_data_map_set_default_device
