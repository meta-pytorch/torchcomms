C From Dave McNamara at PSRV. Thanks! 
C Ported to fortran by Kevin London
C If an event is countable but you've exhausted the counter resources
C and you try to add an event, it seems subsequent PAPI_start and/or
C PAPI_stop will causes a Seg. Violation.

C  I got around this by calling PAPI to get the # of countable events,
C then making sure that I didn't try to add more than these number of
C events. I still have a problem if someone adds Level 2 cache misses
C and then adds FLOPS 'cause I didn't count FLOPS as actually requiring
C 2 counters. 

#include "fpapi_test.h"

      program case2
      IMPLICIT integer (p)

      REAL c,a,b
      INTEGER n
      INTEGER EventSet
      INTEGER retval
      INTEGER I,j 
      INTEGER*8 gl(3)

      INTEGER last_char
      EXTERNAL last_char

      integer tests_quiet, get_quiet
      external get_quiet

      tests_quiet = get_quiet()

      a=0.999
      b=1.001
      n=1000
      i=0
      j=0
      EventSet = PAPI_NULL

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init( retval )
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     .  'PAPI_library_init', retval)
      end if

      call PAPIf_create_eventset( EventSet, retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_create_eventset', 
     *retval)
      end if

      call PAPIf_query_event(PAPI_BR_CN, retval)
      if (retval .EQ. PAPI_OK) then
        j = j + 1
      end if
      if (j .NE. 0) then
        call PAPIf_add_event( EventSet, PAPI_BR_CN, retval )
        if ( retval .NE. PAPI_OK ) then
           if (tests_quiet .EQ. 0) then
            call PAPIf_perror( 'PAPIf_add_event' )
           endif
        end if
      end if

      i = j

      call PAPIf_query_event(PAPI_TOT_CYC, retval)
      if (retval .EQ. PAPI_OK) then
        j = j + 1
      end if
      if (j .EQ. i+1) then
         call PAPIf_add_event( EventSet, PAPI_TOT_CYC, retval )
         if ( retval .NE. PAPI_OK )then
            if (tests_quiet .EQ. 0) then
                call PAPIf_perror( 'PAPIf_add_event' )
            end if 
         end if
      end if
 
      i = j
      call PAPIf_query_event(PAPI_FP_INS, retval)
      if (retval .EQ. PAPI_OK) then
        j = j + 1
      end if
      if (j .EQ. i+1) then
         call PAPIf_add_event(EventSet,PAPI_TOT_INS,retval)
         if ( retval .NE. PAPI_OK )then
           if ( retval .NE. PAPI_ECNFLCT ) then
            if (tests_quiet .EQ. 0) then
                call PAPIf_perror( 'PAPIf_add_event' )
            end if
          end if
         end if
      end if

      if (J .GT. 0) then
         call PAPIf_start(EventSet, retval )
         if ( retval .NE. PAPI_OK ) then
            call ftest_fail(__FILE__, __LINE__,
     .      'PAPIf_start', retval)
         end if
      end if

      do i=1,n
       c = a * b
      end do

      if (J .GT. 0) then
         call PAPIf_stop( EventSet, gl, retval)
         if ( retval .NE. PAPI_OK ) then
            call ftest_fail(__FILE__, __LINE__,
     .        'PAPIf_stop', retval)
         end if
      end if
      
      call ftests_pass(__FILE__)
      end
