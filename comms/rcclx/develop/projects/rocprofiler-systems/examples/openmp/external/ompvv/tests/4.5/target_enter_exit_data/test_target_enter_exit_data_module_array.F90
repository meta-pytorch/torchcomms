!===--test_target_enter_exit_data_module_array.F90 -------mapping of arrays inmodules--===!
! 
! OpenMP API Version 4.5 Nov 2015
! 
! This tests checks for pairs target enter data and target exit data for
! the modifiers alloc and constructs in array that are defined in
! modules and maps that are made in modules 
!
!!===----------------------------------------------------------------------===!
#include "ompvv.F90"

#define N 100
      
      MODULE testing_module
        USE ompvv_lib
        USE omp_lib
        INTEGER :: array_1d(N)
        INTEGER :: array_2d(N,N)
        INTEGER :: array_3d(N,N,N)
        ! helper arrays 
        INTEGER :: mod_helper_array_1d(N)
        INTEGER :: mod_helper_array_2d(N,N)
        INTEGER :: mod_helper_array_3d(N,N,N)
        INTEGER :: mod_mod_errs
        CHARACTER (len = 500) :: mod_helperMsg

       CONTAINS 
        ! Testing a function for mapping to the device inside the module
        INTEGER FUNCTION map_arrays_to_from_device(isSharedEnv)
          LOGICAL :: isSharedEnv
            OMPVV_INFOMSG("testing map(to/from) inside module function")

            mod_errs = 0

            array_1d(:) = 10
            array_2d(:,:) = 10
            array_3d(:,:,:) = 10

            ! Mapping arrays from module
            !$omp target enter data map(to: array_1d) &
            !$omp map(to: array_2d, array_3d) 

            ! Modify the host version to check that it does
            ! not get copied over the device again
            IF (.NOT. isSharedEnv) THEN 
              array_1d(:) = 0
              array_2d(:,:) = 0
              array_3d(:,:,:) = 0
            END IF

            !$omp target map(from: mod_helper_array_1d) &
            !$omp map(from: mod_helper_array_2d, mod_helper_array_3d)
              mod_helper_array_1d = array_1d
              mod_helper_array_2d = array_2d
              mod_helper_array_3d = array_3d

              ! We modify the array
              array_1d(:) = 11
              array_2d(:,:) = 11
              array_3d(:,:,:) = 11
            !$omp end target

            ! Guarantee that the data was copied to
            OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(mod_helper_array_1d /= 10))
            OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(mod_helper_array_2d /= 10))
            OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(mod_helper_array_3d /= 10))

            ! check that the data has not be copied back (this should
            ! happen with the exit data and not hte target region)
            IF (.not. isSharedEnv) THEN
              OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(array_1d /= 0))
              OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(array_2d /= 0))
              OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(array_3d /= 0))
            ELSE
              OMPVV_WARNING("Part of test ommited: shared data env")
            END IF

            !$omp target exit data map(from: array_1d, array_2d) &
            !$omp map(from: array_3d)

            OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(array_1d /= 11))
            OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(array_2d /= 11))
            OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(array_3d /= 11))

            map_arrays_to_from_device = mod_errs
        END FUNCTION map_arrays_to_from_device

        ! Testing a function for maping inside of the module 
        INTEGER FUNCTION  map_arrays_alloc_delete_device(isSharedEnv)
          LOGICAL :: isSharedEnv
            WRITE (mod_helperMsg, *) "testing map(alloc/delete) inside &
            &module function. "
            OMPVV_INFOMSG(mod_helperMsg)

            mod_errs = 0

            ! Mapping arrays from module
            !$omp target enter data map(alloc: array_1d) &
            !$omp map(alloc: array_2d, array_3d) 

            ! Initialize the target copy 
            !$omp target
              array_1d(:) = 10
              array_2d(:,:) = 10
              array_3d(:,:,:) = 10
            !$omp end target

            ! Initialize the host version, it should not affect the
            ! device version
            IF (.NOT. isSharedEnv) THEN 
              array_1d(:) = 0
              array_2d(:,:) = 0
              array_3d(:,:,:) = 0
            END IF

            !$omp target map(from: mod_helper_array_1d) &
            !$omp map(from: mod_helper_array_2d, mod_helper_array_3d)
              mod_helper_array_1d = array_1d
              mod_helper_array_2d = array_2d
              mod_helper_array_3d = array_3d
            !$omp end target

            OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(mod_helper_array_1d /= 10))
            OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(mod_helper_array_2d /= 10))
            OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(mod_helper_array_3d /= 10))

            ! check that it did not copy back
            IF (.not. isSharedEnv) THEN
              OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(array_1d /= 0))
              OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(array_2d /= 0))
              OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(array_3d /= 0))
            ELSE
              OMPVV_WARNING("Part of test ommited: shared data env")
            END IF

            ! Deleting the target version should allow the host data to
            ! be copied over the device with another target region
            !$omp target exit data map(delete: array_1d, array_2d) &
            !$omp map(delete: array_3d)

            ! Setting a known host value 
            array_1d(:) = 99
            array_2d(:,:) = 99
            array_3d(:,:,:) = 99

            !$omp target map(from: mod_helper_array_1d) &
            !$omp map(from: mod_helper_array_2d, mod_helper_array_3d) &
            !$omp map(tofrom: array_1d, array_2d, array_3d)
              mod_helper_array_1d = array_1d
              mod_helper_array_2d = array_2d
              mod_helper_array_3d = array_3d
              ! Change the arrays values
              array_1d(:) = 0
              array_2d(:,:) = 0
              array_3d(:,:,:) = 0
            !$omp end target

            OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(mod_helper_array_1d /= 99))
            OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(mod_helper_array_2d /= 99))
            OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(mod_helper_array_3d /= 99))

            ! check that it did not copy back
            OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(array_1d /= 0))
            OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(array_2d /= 0))
            OMPVV_TEST_AND_SET_VERBOSE(mod_errs, ANY(array_3d /= 0))

            map_arrays_alloc_delete_device = mod_errs
        END FUNCTION map_arrays_alloc_delete_device
      END MODULE testing_module
      
      PROGRAM test_target_map_module_array
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        USE testing_module
        implicit none
        LOGICAL :: isSharedEnv
        INTEGER :: errs
        ! helper arrays 
        INTEGER :: helper_array_1d(N)
        INTEGER :: helper_array_2d(N,N)
        INTEGER :: helper_array_3d(N,N,N)
        CHARACTER (len = 500) :: helperMsg
        
        OMPVV_TEST_OFFLOADING
        OMPVV_TEST_AND_SET_SHARED_ENVIRONMENT(isSharedEnv)

        OMPVV_TEST_VERBOSE(test_module_map_to_from_array() .ne. 0)
        OMPVV_TEST_VERBOSE(test_module_map_alloc_delete_array() .ne. 0)
        OMPVV_TEST_VERBOSE(map_arrays_to_from_device(isSharedEnv) .ne. 0)
        OMPVV_TEST_VERBOSE(map_arrays_alloc_delete_device(isSharedEnv) .ne. 0)

        OMPVV_REPORT_AND_RETURN()


        CONTAINS 
          ! Function to tests the map (to:/from:...)
          INTEGER FUNCTION  test_module_map_to_from_array()
            OMPVV_INFOMSG("testing map(to/from) of array in module")

            errs = 0

            array_1d(:) = 10
            array_2d(:,:) = 10
            array_3d(:,:,:) = 10

            ! Mapping arrays from module
            !$omp target enter data map(to: array_1d) &
            !$omp map(to: array_2d, array_3d) 

            ! Modify the host version to check that it does
            ! not get copied over the device again
            IF (.NOT. isSharedEnv) THEN 
              array_1d(:) = 0
              array_2d(:,:) = 0
              array_3d(:,:,:) = 0
            END IF

            !$omp target map(from: helper_array_1d) &
            !$omp map(from: helper_array_2d, helper_array_3d)
              helper_array_1d = array_1d
              helper_array_2d = array_2d
              helper_array_3d = array_3d

              ! We modify the array
              array_1d(:) = 11
              array_2d(:,:) = 11
              array_3d(:,:,:) = 11
            !$omp end target

            OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(helper_array_1d /= 10))
            OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(helper_array_2d /= 10))
            OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(helper_array_3d /= 10))

            ! check that the data has not be copied back (this should
            ! happen with the exit data and not hte target region)
            IF (.not. isSharedEnv) THEN
              OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(array_1d /= 0))
              OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(array_2d /= 0))
              OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(array_3d /= 0))
            ELSE
              OMPVV_WARNING("Part of test ommited: shared data env")
            END IF

            !$omp target exit data map(from: array_1d, array_2d) &
            !$omp map(from: array_3d)

            OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(array_1d /= 11))
            OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(array_2d /= 11))
            OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(array_3d /= 11))

            test_module_map_to_from_array = errs

          END FUNCTION test_module_map_to_from_array
          ! Function to tests the map (alloc:/delete:...)
          INTEGER FUNCTION  test_module_map_alloc_delete_array()
            OMPVV_INFOMSG("testing map(alloc/delete) of array in module")

            errs = 0

            ! Mapping arrays from module
            !$omp target enter data map(alloc: array_1d) &
            !$omp map(alloc: array_2d, array_3d) 

            ! Initialize the target copy 
            !$omp target
              array_1d(:) = 10
              array_2d(:,:) = 10
              array_3d(:,:,:) = 10
            !$omp end target

            ! Initialize the host version, it should not affect the
            ! device version
            IF (.NOT. isSharedEnv) THEN 
              array_1d(:) = 0
              array_2d(:,:) = 0
              array_3d(:,:,:) = 0
            END IF

            !$omp target map(from: helper_array_1d) &
            !$omp map(from: helper_array_2d, helper_array_3d)
              helper_array_1d = array_1d
              helper_array_2d = array_2d
              helper_array_3d = array_3d
            !$omp end target

            OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(helper_array_1d /= 10))
            OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(helper_array_2d /= 10))
            OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(helper_array_3d /= 10))

            ! check that it did not copy back
            IF (.not. isSharedEnv) THEN
              OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(array_1d /= 0))
              OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(array_2d /= 0))
              OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(array_3d /= 0))
            ELSE
              OMPVV_WARNING("Part of test ommited: shared data env")
            END IF

            ! Deleting the target version should allow the host data to
            ! be copied over the device with another target region
            !$omp target exit data map(delete: array_1d, array_2d) &
            !$omp map(delete: array_3d)

            ! Setting a known host value 
            array_1d(:) = 99
            array_2d(:,:) = 99
            array_3d(:,:,:) = 99

            !$omp target map(from: helper_array_1d) &
            !$omp map(from: helper_array_2d, helper_array_3d) &
            !$omp map(tofrom: array_1d, array_2d, array_3d)
              helper_array_1d = array_1d
              helper_array_2d = array_2d
              helper_array_3d = array_3d
              ! Change the arrays values
              array_1d(:) = 0
              array_2d(:,:) = 0
              array_3d(:,:,:) = 0
            !$omp end target

            OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(helper_array_1d /= 99))
            OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(helper_array_2d /= 99))
            OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(helper_array_3d /= 99))

            ! check that it did not copy back
            OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(array_1d /= 0))
            OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(array_2d /= 0))
            OMPVV_TEST_AND_SET_VERBOSE(errs, ANY(array_3d /= 0))
            test_module_map_alloc_delete_array = errs

          END FUNCTION test_module_map_alloc_delete_array
      END PROGRAM test_target_map_module_array

