!===---- test_target_data_map.F90 - test for map type modifiers ------------===//
!
! OpenMP API Version 4.5 Nov 2015
!
! This test check all the possible map-type-modifiers for the target data map
! clauses. These are: from, to, fromto, alloc, release and delete. There
! is a function for each test.
!
!===----------------------------------------------------------------------===//
#include "ompvv.F90"

#define N 1000

      PROGRAM test_target_data_map
        USE iso_fortran_env
        USE ompvv_lib
        USE omp_lib
        implicit none
        CHARACTER (len=400) :: msgHelper

        OMPVV_TEST_OFFLOADING
        OMPVV_TEST_VERBOSE(test_target_data_map_from() .ne. 0)
        OMPVV_TEST_VERBOSE(test_target_data_map_to() .ne. 0)
        OMPVV_TEST_VERBOSE(test_target_data_map_tofrom() .ne. 0)
        OMPVV_TEST_VERBOSE(test_target_data_map_alloc() .ne. 0)

        OMPVV_REPORT_AND_RETURN()


        CONTAINS
          ! Testing from:
          INTEGER FUNCTION test_target_data_map_from()
            ! heap and stack
            INTEGER, DIMENSION(N) :: h_array_s, aux_array
            INTEGER, POINTER, DIMENSION(:) :: h_array_h
            INTEGER :: err_bf, err_af, i

            OMPVV_INFOMSG("Testing from:...")
            OMPVV_GET_ERRORS(err_bf)
            allocate(h_array_h(N))

            ! init arrays
            aux_array(:) = 0
            h_array_s(:) = (/ (i, i=1,N) /)
            h_array_h(:) = (/ (i, i=1,N) /)

            !$omp target data map(from: h_array_h(1:N), h_array_s(1:N))
              !$omp target map(tofrom: aux_array)
                ! Since it is from, this should not be a known value
                aux_array(:) = h_array_s(:) + h_array_h(:)
                ! Modify the value to read it from the host
                h_array_h(:) = 10
                h_array_s(:) = 20
              !$omp end target
            !$omp end target data

            IF (SUM(aux_array) == SUM((/ (2*i, i=1,N)/))) THEN
              WRITE(msgHelper, *) "Possible data moved to the device &
                &when using the from modifier"
              OMPVV_WARNING(msgHelper)
            END IF
            OMPVV_TEST_VERBOSE(ANY(h_array_h /= 10))
            OMPVV_TEST_VERBOSE(ANY(h_array_s /= 20))

            deallocate(h_array_h)
            OMPVV_GET_ERRORS(err_af)
            test_target_data_map_from = err_af - err_bf
          END FUNCTION test_target_data_map_from

          ! Testing tofrom
          INTEGER FUNCTION test_target_data_map_tofrom()
            ! heap and stacka
            INTEGER, DIMENSION(N) :: h_array_s, aux_array
            INTEGER, POINTER, DIMENSION(:) :: h_array_h
            INTEGER :: err_bf, err_af, i

            OMPVV_INFOMSG("Testing tofrom:...")

            OMPVV_GET_ERRORS(err_bf)
            allocate(h_array_h(N))

            ! init arrays
            aux_array(:) = 0
            h_array_s(:) = (/ (i, i=1,N) /)
            h_array_h(:) = (/ (i, i=1,N) /)

            !$omp target data map(tofrom: h_array_h(1:N), h_array_s(1:N))
              !$omp target map(tofrom: aux_array)
                ! Since it is tofrom, this should have the original values added
                ! together
                aux_array(:) = h_array_s(:) + h_array_h(:)

                ! Modify the value to read it from the host
                h_array_h(:) = 10
                h_array_s(:) = 20
              !$omp end target
            !$omp end target data

            OMPVV_TEST_VERBOSE(SUM(aux_array) /= SUM((/ (2*i, i=1,N)/)))
            OMPVV_TEST_VERBOSE(ANY(h_array_h /= 10))
            OMPVV_TEST_VERBOSE(ANY(h_array_s /= 20))

            deallocate(h_array_h)
            OMPVV_GET_ERRORS(err_af)
            test_target_data_map_tofrom = err_af - err_bf
          END FUNCTION test_target_data_map_tofrom
          ! Testing to
          INTEGER FUNCTION test_target_data_map_to()
            ! heap and stacka
            INTEGER, DIMENSION(N) :: h_array_s, aux_array
            INTEGER, POINTER, DIMENSION(:) :: h_array_h
            INTEGER :: err_bf, err_af, i

            OMPVV_INFOMSG("Testing to:...")

            OMPVV_GET_ERRORS(err_bf)
            allocate(h_array_h(N))

            ! init arrays
            aux_array(:) = 0
            h_array_s(:) = (/ (i, i=1,N) /)
            h_array_h(:) = (/ (i, i=1,N) /)

            !$omp target data map(to: h_array_h(1:N), h_array_s(1:N))
              !$omp target map(tofrom: aux_array)
                ! Since it is to, this should have the original values added
                ! together
                aux_array(:) = h_array_s(:) + h_array_h(:)

                ! Modify the value should not change in the host unless shared
                ! memory
                h_array_h(:) = 10
                h_array_s(:) = 20
              !$omp end target
            !$omp end target data

            OMPVV_TEST_VERBOSE(SUM(aux_array) /= SUM((/ (2*i, i=1,N)/)))
            OMPVV_TEST_VERBOSE(ALL(h_array_h == 10))
            OMPVV_TEST_VERBOSE(ALL(h_array_s == 20))

            deallocate(h_array_h)
            OMPVV_GET_ERRORS(err_af)
            test_target_data_map_to = err_af - err_bf
          END FUNCTION test_target_data_map_to
          ! Testing alloc
          INTEGER FUNCTION test_target_data_map_alloc()
            ! heap and stacka
            INTEGER, DIMENSION(N) :: h_array_s, aux_array, aux_array2
            INTEGER, POINTER, DIMENSION(:) :: h_array_h
            INTEGER :: err_bf, err_af, i

            OMPVV_INFOMSG("Testing alloc:...")

            OMPVV_GET_ERRORS(err_bf)
            allocate(h_array_h(N))

            ! init arrays
            aux_array(:) = 0
            aux_array2(:) = 0
            h_array_s(:) = (/ (i, i=1,N) /)
            h_array_h(:) = (/ (i, i=1,N) /)

            !$omp target data map(alloc: h_array_h(1:N), h_array_s(1:N))
              !$omp target map(tofrom: aux_array, aux_array2)
                ! Since it is alloc, this should not be a known value
                aux_array(:) = h_array_s(:) + h_array_h(:)

                ! Modify the value, but the host should not change
                h_array_h(:) = 10
                h_array_s(:) = 20

                ! allocation should allow this to happen, but variables
                ! in the host should not change
                aux_array2(:) = h_array_h(:) + h_array_s(:)
              !$omp end target
            !$omp end target data

            OMPVV_TEST_VERBOSE(SUM(aux_array2) /= 30 * N )
            IF (SUM(aux_array) == SUM((/ (2*i, i=1,N)/))) THEN
               WRITE(msgHelper, *) "Possible data moved to the device &
                    &when using the alloc modifier"
               OMPVV_WARNING(msgHelper)
            END IF
            OMPVV_TEST_VERBOSE(ALL(h_array_h == 10))
            OMPVV_TEST_VERBOSE(ALL(h_array_s == 20))

            deallocate(h_array_h)
            OMPVV_GET_ERRORS(err_af)
            test_target_data_map_alloc = err_af - err_bf
          END FUNCTION test_target_data_map_alloc
      END PROGRAM test_target_data_map
