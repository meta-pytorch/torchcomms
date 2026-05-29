C From Dave McNamara at PSRV. Thanks! 
C Ported to Fortran by Kevin London
C If you try to add an event that doesn't exist, you get the correct error
C message, yet you get subsequent Seg. Faults when you try to do PAPI_start 
C and PAPI_stop. I would expect some bizarre behavior if I had no events 
C added to the event set and then tried to PAPI_start but if I had 
C successfully added one event, then the 2nd one get an error when I 
C tried to add it, is it possible for PAPI_start to work but just 
C count the first event?

#include  "fpapi_test.h"

      program case1
      IMPLICIT integer (p)

      INTEGER EventSet
      INTEGER retval
      INTEGER i,j
      INTEGER*8 gl(2)
      INTEGER n
      REAL c,a,b

      INTEGER last_char
      EXTERNAL last_char
      integer tests_quiet, get_quiet
      external get_quiet

      tests_quiet = get_quiet()

      n = 1000
      a = 0.999
      b = 1.001
      j = 0
      i = 0
      EventSet = PAPI_NULL

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init( retval )
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__, 
     . 'PAPI_library_init', retval)
      end if

      call PAPIf_create_eventset( EventSet, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     .  'PAPIf_create_eventset', 
     *retval)
      end if

      call PAPIf_query_event(PAPI_L2_TCM, retval)
      if (retval .EQ. PAPI_OK) then
        j = j + 1
      end if
      if (j .NE. 0) then
        call PAPIf_add_event( EventSet, PAPI_L2_TCM, retval )
        if (retval .NE. PAPI_OK) then
            if (retval .NE. PAPI_ECNFLCT) then
               call ftest_fail(__FILE__, __LINE__,
     .         'PAPIf_add_event', 
     *retval)
            else
              j = j - 1
            end if
        end if
      end if
      i = j

      call PAPIf_query_event(PAPI_L2_DCM, retval)
      if (retval .EQ. PAPI_OK) then
        j = j + 1
      end if
      if (j .EQ. i+1) then
        call PAPIf_add_event( EventSet, PAPI_L2_DCM, retval )
        if (retval .NE. PAPI_OK) then
            if (retval .NE. PAPI_ECNFLCT) then
               call ftest_fail(__FILE__, __LINE__,
     .              'PAPIf_add_event', 
     *retval)
            else
              j = j - 1
            end if
        end if
      end if

      if (J .GT. 0) then
        call PAPIf_start( EventSet, retval )
        if ( retval .NE. PAPI_OK ) then
           call ftest_fail(__FILE__, __LINE__,
     .        'PAPIf_start', retval)
        end if
      end if


      do i=1, n
       c = a * b
      end do

      if (j .GT. 0) then
        call PAPIf_stop( EventSet, gl, retval ) 
        if ( retval .NE. PAPI_OK ) then
           call ftest_fail(__FILE__, __LINE__,
     .         'PAPIf_stop', retval)
        end if
      end if
      
      call ftests_pass(__FILE__)
      end
