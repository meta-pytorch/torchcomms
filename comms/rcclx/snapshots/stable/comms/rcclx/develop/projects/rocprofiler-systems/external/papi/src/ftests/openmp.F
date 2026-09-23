#include "fpapi_test.h"

      program openmp
      
      use omp_lib

      integer*8 values(10)
      integer es
      integer retval

      Integer last_char
      External last_char
      integer tests_quiet, get_quiet
      external get_quiet

      integer nthreads, tid

      tests_quiet = get_quiet()
      es = PAPI_NULL

      call PAPIF_thread_init(omp_get_thread_num, retval)


      retval = PAPI_VER_CURRENT
      call PAPIf_library_init(retval)
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     .       'PAPI_library_init', retval)
      end if

      call PAPIf_query_event(PAPI_TOT_CYC, retval)
      if (retval .NE. PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .       'PAPIf_query_event', 
     *       retval)
      end if

      call PAPIf_create_eventset(es, retval)
      if ( retval.NE.PAPI_OK) then
        call ftest_fail(__FILE__, __LINE__,
     .       'PAPIf_create_eventset', 
     *       retval)
      end if
      
      call PAPIf_add_event( es, PAPI_TOT_CYC, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .       'PAPIf_add_event', retval)
      end if

      call PAPIf_start(es, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .       'PAPIf_start', retval)
      end if

!$OMP PARALLEL PRIVATE(NTHREADS, TID)

      tid  = OMP_GET_THREAD_NUM()

      if (tests_quiet .EQ. 0) then
         PRINT *, 'Hello World from thread = ', TID
      end if

!$OMP END PARALLEL


      call PAPIf_stop(es, values(1), retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .       'PAPIf_stop', retval)
      end if

      if (tests_quiet .EQ. 0) then
        write (*,*) "PAPI_TOT_CYC", values(1)
      end if


      
      call ftests_pass(__FILE__)
      end

