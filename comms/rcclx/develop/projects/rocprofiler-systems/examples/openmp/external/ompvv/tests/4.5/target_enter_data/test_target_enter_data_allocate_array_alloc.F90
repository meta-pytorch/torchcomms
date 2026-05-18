!===--test_target_enter_data_allocate_array_alloc.F90 - allocate array map alloc--===!
! 
! OpenMP API Version 4.5 Nov 2015
!
! Testing the mapping of arrays that are allocated dynamically. This tests
! covers multiple array dimmensions and uses target enter data map(to) 
!
!!===----------------------------------------------------------------------===!
#include "ompvv.F90"

#define N 20

      PROGRAM tests_target_enter_data_allocate_array_alloc
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none
        INTEGER, ALLOCATABLE, DIMENSION(:) :: my1DPtr
        INTEGER, ALLOCATABLE, DIMENSION(:,:) :: my2DPtr
        INTEGER, ALLOCATABLE, DIMENSION(:,:,:) :: my3DPtr
        ! Helper arrays
        INTEGER, DIMENSION(N) :: my1DArr
        INTEGER, DIMENSION(N,N) :: my2DArr
        INTEGER, DIMENSION(N,N,N) :: my3DArr
        ! Helper functions
        LOGICAL :: isSharedEnv
        CHARACTER (len = 400) :: helperMsg
        INTEGER :: errors, i
      
        OMPVV_TEST_OFFLOADING
        OMPVV_TEST_AND_SET_SHARED_ENVIRONMENT(isSharedEnv)

        WRITE(helperMsg, *) "Omitting part of the test due to &
          &shared data environment"
        OMPVV_WARNING_IF(isSharedEnv, helperMsg)

        OMPVV_TEST_VERBOSE(test_allocate_array1D_map_alloc() .ne. 0)
        OMPVV_TEST_VERBOSE(test_allocate_array2D_map_alloc() .ne. 0)
        OMPVV_TEST_VERBOSE(test_allocate_array3D_map_alloc() .ne. 0)

        OMPVV_REPORT_AND_RETURN()


        CONTAINS 
          ! 1D Array test
          INTEGER FUNCTION test_allocate_array1D_map_alloc()

            OMPVV_INFOMSG("Testing map alloc of allocate 1D array")
            errors = 0
            ! Allocate the arrays
            allocate(my1DPtr(N))

            ! initialize 
            my1DPtr(:) = 0

            ! Mapping the array
            !$omp target enter data map(alloc: my1DPtr(:))

            ! Assign a value to the allocated space
            !$omp target
              my1DPtr(:) = (/ (i , i = 1,N) /)
            !$omp end target

            ! Confirm mapping with target region
            !$omp target map(from: my1DArr) 
              my1DArr = my1DPtr
            !$omp end target

            IF (.NOT. isSharedEnv) THEN
              OMPVV_TEST_AND_SET_VERBOSE(errors, ANY(my1DPtr /= 0))
            END IF
            OMPVV_TEST_AND_SET_VERBOSE(errors, SUM(my1DArr) /= ((N*(N+1)/2)))

            ! This is not part of the test but it is necessary to avoid
            ! having memory leaks
            !$omp target exit data map(delete: my1DPtr)
            deallocate(my1DPtr)

            test_allocate_array1D_map_alloc = errors

          END FUNCTION test_allocate_array1D_map_alloc
          ! 2D Array test
          INTEGER FUNCTION test_allocate_array2D_map_alloc()

            OMPVV_INFOMSG("Testing map alloc of allocate 2D array")
            errors = 0
            ! Allocate the arrays
            allocate(my2DPtr(N,N))

            ! initialize 
            my2DPtr(:,:) = 0

            ! Mapping the array
            !$omp target enter data map(alloc: my2DPtr(:,:))

            ! Assign a value to the allocated space
            !$omp target
              my2DPtr(:,:) = RESHAPE((/ (i , i = 1,N**2) /), (/ N,N /))
            !$omp end target

            ! Confirm mapping with target region
            !$omp target map(from: my2DArr) 
              my2DArr = my2DPtr
            !$omp end target

            IF (.NOT. isSharedEnv) THEN
              OMPVV_TEST_AND_SET_VERBOSE(errors, ANY(my2DPtr /= 0))
            END IF
            OMPVV_TEST_AND_SET_VERBOSE(errors, SUM(my2DArr) /= ((N**2*(N**2+1)/2)))

            ! This is not part of the test but it is necessary to avoid
            ! having memory leaks
            !$omp target exit data map(delete: my2DPtr)
            deallocate(my2DPtr)

            test_allocate_array2D_map_alloc = errors

          END FUNCTION test_allocate_array2D_map_alloc
          ! 3D Array test
          INTEGER FUNCTION test_allocate_array3D_map_alloc()

            OMPVV_INFOMSG("Testing map alloc of allocate 3D array")
            errors = 0
            ! Allocate the arrays
            allocate(my3DPtr(N,N,N))

            ! initialize 
            my3DPtr(:,:,:) = 0

            ! Mapping the array
            !$omp target enter data map(alloc: my3DPtr(:,:,:))

            ! Assign a value to the allocated space
            !$omp target
              my3DPtr(:,:,:) = RESHAPE((/ (i , i = 1,N**3) /), (/ N,N,N /))
            !$omp end target

            ! Confirm mapping with target region
            !$omp target map(from: my3DArr) 
              my3DArr = my3DPtr
            !$omp end target


            IF (.NOT. isSharedEnv) THEN
              OMPVV_TEST_AND_SET_VERBOSE(errors, ANY(my3DPtr /= 0))
            END IF
            OMPVV_TEST_AND_SET_VERBOSE(errors, SUM(my3DArr) /= ((N**3*(N**3+1)/2)))

            ! This is not part of the test but it is necessary to avoid
            ! having memory leaks
            !$omp target exit data map(delete: my3DPtr)
            deallocate(my3DPtr)

            test_allocate_array3D_map_alloc = errors

          END FUNCTION test_allocate_array3D_map_alloc
      END PROGRAM tests_target_enter_data_allocate_array_alloc

