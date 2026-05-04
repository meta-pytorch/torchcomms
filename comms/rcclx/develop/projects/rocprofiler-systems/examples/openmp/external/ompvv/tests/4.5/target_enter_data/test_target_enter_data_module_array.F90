!===--test_target_enter_data_module_array.F90 -------mapping of arrays inmodules--===!
! 
! OpenMP API Version 4.5 Nov 2015
! 
! Testing the mapping of arrays that are instantiated in modules. This
! tests covers mapping the array in the main program as well as mapping
! the array in the module program       
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

       CONTAINS 
        ! Testing a function for mapping to the device inside the module
        INTEGER FUNCTION map_arrays_to_device(isSharedEnv)
          LOGICAL :: isSharedEnv
            OMPVV_INFOMSG("testing map(to...) inside module function")

            mod_errs = 0

            array_1d(:) = 10
            array_2d(:,:) = 10
            array_3d(:,:,:) = 10

            ! Mapping arrays from module
            !$omp target enter data map(to: array_1d) &
            !$omp map(to: array_2d, array_3d) 

            ! Modify the host version to check that it does
            ! not get pass again
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

              ! This value should not be copied back
              array_1d(:) = 11
              array_2d(:,:) = 11
              array_3d(:,:,:) = 11
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

            ! This is not part of the test but it is necessary to avoid
            ! memory leaks or side effects 
            !$omp target exit data map(delete: array_1d, array_2d) &
            !$omp map(delete: array_3d)
             map_arrays_to_device = mod_errs
        END FUNCTION map_arrays_to_device

        ! Testing a function for maping inside of the module 
        INTEGER FUNCTION  map_arrays_alloc_device(isSharedEnv)
          LOGICAL :: isSharedEnv
            OMPVV_INFOMSG("testing map(alloc...) inside module function")

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

            ! This is not part of the test but it is necessary to avoid
            ! memory leaks or side effects 
            !$omp target exit data map(delete: array_1d, array_2d) &
            !$omp map(delete: array_3d)
             map_arrays_alloc_device = mod_errs
        END FUNCTION map_arrays_alloc_device
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
        
        OMPVV_TEST_OFFLOADING
        OMPVV_TEST_AND_SET_SHARED_ENVIRONMENT(isSharedEnv)

        OMPVV_TEST_VERBOSE(test_module_map_to_array() .ne. 0)
        OMPVV_TEST_VERBOSE(test_module_map_alloc_array() .ne. 0)
        OMPVV_TEST_VERBOSE(map_arrays_to_device(isSharedEnv) .ne. 0)
        OMPVV_TEST_VERBOSE(map_arrays_alloc_device(isSharedEnv) .ne. 0)

        OMPVV_REPORT_AND_RETURN()


        CONTAINS 
          ! Function to tests the map (to:...)
          INTEGER FUNCTION  test_module_map_to_array()
            OMPVV_INFOMSG("testing map(to...) of array in module")

            errs = 0

            array_1d(:) = 10
            array_2d(:,:) = 10
            array_3d(:,:,:) = 10

            ! Mapping arrays from module
            !$omp target enter data map(to: array_1d) &
            !$omp map(to: array_2d, array_3d) 

            ! Modify the host version to check that it does
            ! not get pass again
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

              ! This value should not be copied back
              array_1d(:) = 11
              array_2d(:,:) = 11
              array_3d(:,:,:) = 11
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

            ! This is not part of the test but it is necessary to avoid
            ! memory leaks or side effects 
            !$omp target exit data map(delete: array_1d, array_2d) &
            !$omp map(delete: array_3d)
            test_module_map_to_array = errs

          END FUNCTION test_module_map_to_array
          ! Function to tests the map (alloc:...)
          INTEGER FUNCTION  test_module_map_alloc_array()
            OMPVV_INFOMSG("testing map(alloc...) of array in module")

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

            ! This is not part of the test but it is necessary to avoid
            ! memory leaks or side effects 
            !$omp target exit data map(delete: array_1d, array_2d) &
            !$omp map(delete: array_3d)
            test_module_map_alloc_array = errs

          END FUNCTION test_module_map_alloc_array
      END PROGRAM test_target_map_module_array

