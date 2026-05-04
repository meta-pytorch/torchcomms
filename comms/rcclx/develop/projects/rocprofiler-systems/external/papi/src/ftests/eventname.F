#include "fpapi_test.h"

      program eventname
      IMPLICIT integer (p)

      INTEGER retval, handle
      INTEGER preset
      integer tests_quiet, get_quiet
      external get_quiet

      tests_quiet = get_quiet()

      handle = 0
      retval = PAPI_VER_CURRENT
      call PAPIf_library_init( retval )
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPI_library_init', retval)
      end if

      call PAPIf_event_name_to_code( 'PAPI_FP_INS',preset,retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_event_name_to_code', 
     *retval)
      end if

      if (tests_quiet .EQ. 0) then
      write (*, 100) preset
 100  format ('PAPI_FP_INS code is', Z10)
      end if
      call ftests_pass(__FILE__)
      end

