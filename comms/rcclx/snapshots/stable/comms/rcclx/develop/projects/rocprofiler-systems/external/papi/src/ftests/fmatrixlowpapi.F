C ****************************************************************************
C
C matrixpapi.f
C An example of matrix-matrix multiplication and using PAPI low level to 
C look at the performance. written by Kevin London
C March 2000
C ****************************************************************************

#include "fpapi_test.h"

      program fmatrixlowpapi
      implicit integer (p)

      INTEGER ncols1,nrows1,ncols2,nrows2
      PARAMETER(nrows1=175,ncols1=225,nrows2=ncols1,ncols2=150)
      INTEGER i,j,k,retval,nchr,numevents,EventSet
      CHARACTER*(PAPI_MAX_STR_LEN)  vstring,mstring
C   PAPI values of the counters
      INTEGER event
      INTEGER*8 values(2)
      INTEGER*8 starttime,stoptime
      REAL*8 finaltime
      INTEGER ncpu,nnodes,totalcpus,vendor,model
      REAL revision, mhz
      real*8, allocatable, dimension(:,:) :: p, q, r
      integer tests_quiet, get_quiet
      external get_quiet

      tests_quiet = get_quiet()
      EventSet = PAPI_NULL

C   Setup default values
      numevents=0
      starttime=0
      stoptime=0
      allocate(p(nrows1, ncols1))
      allocate(q(nrows2, ncols2))
      allocate(r(nrows1, ncols2))

      retval = PAPI_VER_CURRENT
      call PAPIf_library_init( retval )
      if ( retval.NE.PAPI_VER_CURRENT) then
        call ftest_fail(__FILE__, __LINE__, 
     *'PAPI_library_init', retval)
      end if

C     Create the eventset
      call PAPIf_create_eventset(EventSet,retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 
     *'PAPIf_create_eventset', retval)
      end if

C   Total cycles
      call PAPIf_add_event(EventSet,PAPI_TOT_CYC,retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 
     *'PAPIf_add_event PAPI_TOT_CYC', retval)
      end if

C   Total [floating point] instructions
        call PAPIf_query_event(PAPI_FP_INS, retval)
        if (retval .NE. PAPI_OK) then
        event = PAPI_TOT_INS 
        else
        event = PAPI_FP_INS 
        end if

      call PAPIf_add_event(EventSet,event,retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__, 
     *'PAPIf_add_event PAPI_TOT_INS', retval)
      end if

C   Grab the hardware info
      call PAPIf_get_hardware_info( ncpu, nnodes, totalcpus, vendor,
     .   vstring, model, mstring, revision, mhz )
      do i=len(mstring),1,-1
        if(mstring(i:i).NE.' ') goto 10
      end do
 10   if(i.LT.1)then 
        nchr=1
      else
        nchr=i
      end if

      if (tests_quiet .EQ. 0) then
      print *
      print 100, totalcpus,mstring(1:nchr), mhz
      print *
      print 101,'ncpu',ncpu, 'nnodes',nnodes, 'totalcpus',totalcpus
      print 102,'mhz',mhz,'revision',revision
      print 103,'vendor',vendor,'vstring',vstring
      print 104,'model',model,'mstring',mstring
      print *
      end if
 100  format(i5,' CPU(s) ',a,' at ',f7.2,' MHz')
 101  format(a9,' =',i6,7x,a9,' =',i5,5x,a9,'=',i5)
 102  format(a9,' =',f7.2,6x,a9,' =',f15.5)
 103  format(a9,' =',i6,7x,a9,' =',a40)
 104  format(a9,' =',i6,7x,a9,' =',a40)
C   Open matrix file number 1 for reading
C      OPEN(UNIT=1,FILE='fmt1',STATUS='OLD')
C   Open matrix file number 2 for reading
C      OPEN(UNIT=2,FILE='fmt2',STATUS='OLD')

C   matrix 1: read in the matrix values
      do i=1, nrows1
         do j=1,ncols1
            p(i,j) = i*j*1.0
         end do
      end do

C   matrix 2: read in the matrix values
      do i=1, nrows2
         do j=1,ncols2
            q(i,j) = i*j*1.0
         end do
      end do

C  Initialize the result matrix 
      do i=1,nrows1
         do j=1, ncols2
            r(i,j) = i*j*1.0
         end do
      end do
      
C  Grab the beginning time
      call PAPIf_get_real_usec( starttime )
 
C  Start the event counters
      call PAPIf_start( EventSet, retval )
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_start', retval)
      end if

C  Compute the matrix-matrix multiplication
      do i=1,nrows1 
       do j=1,ncols2 
         do k=1,ncols1 
          r(i,j)=r(i,j) + p(i,k)*q(k,j)
         end do
       end do
      end do

C  Stop the counters and put the results in the array values 
      call PAPIf_stop(EventSet,values,retval)
      if ( retval .NE. PAPI_OK ) then
        call ftest_fail(__FILE__, __LINE__,
     . 'PAPIf_stop', retval)
      end if

      call PAPIf_get_real_usec( stoptime )
      finaltime=(REAL(stoptime)/1000000.0)-(REAL(starttime)/1000000.0)

C  Make sure the compiler does not optimize away the multiplication
      call dummy(r)

      if (tests_quiet .EQ. 0) then
        print *, 'Time: ', finaltime, 'seconds'
        print *, 'Cycles: ', values(1)

          if (event .EQ. PAPI_TOT_INS) then
          print *, 'Total Instructions: ', values(2)
          else
          print *, 'FP Instructions: ', values(2)
          write(*,'(a,f9.6)') ' Efficiency (fp/cycle):',
     &                         real(values(2))/real(values(1))
          end if
      end if

      deallocate(p, q, r)
      call ftests_pass(__FILE__)
      end

