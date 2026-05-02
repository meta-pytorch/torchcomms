/*
 * File:    linux-timer.c
 *
 * @author:  Vince Weaver
 *           vincent.weaver @ maine.edu
 * Mods:     Philip Mucci
 *           mucci @ icl.utk.edu
 */

#include <time.h>
#include <sys/syscall.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"

#include <fcntl.h>
#include <errno.h>
#include <string.h>

#include <sys/time.h>

#include <fcntl.h>
#include "linux-common.h"

#include <sys/time.h>
#include <sys/resource.h>

#include <sys/times.h>
#include <stdint.h>

#ifdef __ia64__
#include "perfmon/pfmlib_itanium2.h"
#include "perfmon/pfmlib_montecito.h"
#endif

#ifdef __powerpc__
#include <sys/platform/ppc.h>
#endif

#if defined(HAVE_MMTIMER)
#include <sys/mman.h>
#include <linux/mmtimer.h>
#include <sys/ioctl.h>
#ifndef MMTIMER_FULLNAME
#define MMTIMER_FULLNAME "/dev/mmtimer"
#endif

static int mmdev_fd;
static unsigned long mmdev_mask;
static unsigned long mmdev_ratio;
static volatile unsigned long *mmdev_timer_addr;

        /* setup mmtimer */
int mmtimer_setup(void) {

	  unsigned long femtosecs_per_tick = 0;
	  unsigned long freq = 0;
	  int result;
	  int offset;

	  SUBDBG( "MMTIMER Opening %s\n", MMTIMER_FULLNAME );
	  if ( ( mmdev_fd = open( MMTIMER_FULLNAME, O_RDONLY ) ) == -1 ) {
	    PAPIERROR( "Failed to open MM timer %s", MMTIMER_FULLNAME );
	    return PAPI_ESYS;
	  }
	  SUBDBG( "MMTIMER checking if we can mmap" );
	  if ( ioctl( mmdev_fd, MMTIMER_MMAPAVAIL, 0 ) != 1 ) {
	    PAPIERROR( "mmap of MM timer unavailable" );
	    return PAPI_ESYS;
	  }
	  SUBDBG( "MMTIMER setting close on EXEC flag\n" );
	  if ( fcntl( mmdev_fd, F_SETFD, FD_CLOEXEC ) == -1 ) {
	    PAPIERROR( "Failed to fcntl(FD_CLOEXEC) on MM timer FD %d: %s",
		       mmdev_fd, strerror( errno ) );
	    return PAPI_ESYS;
	  }
	  SUBDBG( "MMTIMER is on FD %d, getting offset\n", mmdev_fd );
	  if ( ( offset = ioctl( mmdev_fd, MMTIMER_GETOFFSET, 0 ) ) < 0 ) {
	    PAPIERROR( "Failed to get offset of MM timer" );
	    return PAPI_ESYS;
	  }
	  SUBDBG( "MMTIMER has offset of %d, getting frequency\n", offset );
	  if ( ioctl( mmdev_fd, MMTIMER_GETFREQ, &freq ) == -1 ) {
	    PAPIERROR( "Failed to get frequency of MM timer" );
	    return PAPI_ESYS;
	  }
	  SUBDBG( "MMTIMER has frequency %lu Mhz\n", freq / 1000000 );
	  // don't know for sure, but I think this ratio is inverted
	  //     mmdev_ratio = (freq/1000000) / (unsigned long)_papi_hwi_system_info.hw_info.mhz;
          mmdev_ratio =
		  ( unsigned long ) _papi_hwi_system_info.hw_info.cpu_max_mhz / 
	                      ( freq / 1000000 );
          SUBDBG( "MMTIMER has a ratio of %ld to the CPU's clock, getting resolution\n",
		    mmdev_ratio );
          if ( ioctl( mmdev_fd, MMTIMER_GETRES, &femtosecs_per_tick ) == -1 ) {
		  PAPIERROR( "Failed to get femtoseconds per tick" );
		  return PAPI_ESYS;
          }
          SUBDBG( "MMTIMER res is %lu femtosecs/tick (10^-15s) or %f Mhz, getting valid bits\n",
	  femtosecs_per_tick, 1.0e9 / ( double ) femtosecs_per_tick );
          if ( ( result = ioctl( mmdev_fd, MMTIMER_GETBITS, 0 ) ) == -ENOSYS ) {
	     PAPIERROR( "Failed to get number of bits in MMTIMER" );
	     return PAPI_ESYS;
          }
          mmdev_mask = ~( 0xffffffffffffffff << result );
          SUBDBG( "MMTIMER has %d valid bits, mask %#16lx, getting mmaped page\n",
		    result, mmdev_mask );
          if ( ( mmdev_timer_addr =
		       ( unsigned long * ) mmap( 0, getpagesize(  ), PROT_READ,
						 MAP_PRIVATE, mmdev_fd,
						 0 ) ) == NULL ) {
	     PAPIERROR( "Failed to mmap MM timer" );
	     return PAPI_ESYS;
          }
          SUBDBG( "MMTIMER page is at %p, actual address is %p\n",
			mmdev_timer_addr, mmdev_timer_addr + offset );
          mmdev_timer_addr += offset;
          /* mmdev_fd should be closed and page should be unmapped in a global shutdown routine */
	  return PAPI_OK;

}

#else

#if defined(__powerpc__)
static uint64_t multiplier = 1;
#endif

int mmtimer_setup(void) {

#if defined(__powerpc__)
multiplier = ((uint64_t)_papi_hwi_system_info.hw_info.cpu_max_mhz * 1000000ULL) / (__ppc_get_timebase_freq()/(uint64_t)1000);
#endif
return PAPI_OK; }
#endif





/* Hardware clock functions */

/* All architectures should set HAVE_CYCLES in configure if they have these.
   Not all do so for now, we have to guard at the end of the statement,
   instead of the top. When all archs set this, this region will be guarded
   with:
     #if defined(HAVE_CYCLE)
   which is equivalent to
     #if !defined(HAVE_GETTIMEOFDAY) && !defined(HAVE_CLOCK_GETTIME)
*/

/************************/
/* MMTIMER get_cycles() */
/************************/

#if defined(HAVE_MMTIMER)

static inline long long
get_cycles( void )
{
	long long tmp = 0;

        tmp = *mmdev_timer_addr & mmdev_mask;
	SUBDBG("MMTIMER is %llu, scaled %llu\n",tmp,tmp*mmdev_ratio);
        tmp *= mmdev_ratio;

	return tmp;
}

/************************/
/* ia64 get_cycles()    */
/************************/

#elif defined(__ia64__)
extern int _perfmon2_pfm_pmu_type;

static inline long long
get_cycles( void )
{
	long long tmp = 0;
#if defined(__INTEL_COMPILER)
	tmp = __getReg( _IA64_REG_AR_ITC );
#else
	__asm__ __volatile__( "mov %0=ar.itc":"=r"( tmp )::"memory" );
#endif
	switch ( _perfmon2_pfm_pmu_type ) {
	case PFMLIB_MONTECITO_PMU:
		tmp = tmp * 4;
		break;
	}
	return tmp;
}

/************************/
/* x86 get_cycles()     */
/************************/

#elif (defined(__i386__)||defined(__x86_64__))
static inline long long
get_cycles( void )
{
	long long ret = 0;
#ifdef __x86_64__
	do {
		unsigned int a, d;
		asm volatile ( "rdtsc":"=a" ( a ), "=d"( d ) );
		( ret ) = ( ( long long ) a ) | ( ( ( long long ) d ) << 32 );
	}
	while ( 0 );
#else
	__asm__ __volatile__( "rdtsc":"=A"( ret ): );
#endif
	return ret;
}

/************************/
/* SPARC get_cycles()   */
/************************/

/* #define get_cycles _rtc ?? */
#elif defined(__sparc__)
static inline long long
get_cycles( void )
{
	register unsigned long ret asm( "g1" );

	__asm__ __volatile__( ".word 0x83410000"	/* rd %tick, %g1 */
						  :"=r"( ret ) );
	return ret;
}

/************************/
/* aarch64 get_cycles() */
/************************/

#elif defined(__aarch64__)
static inline long long
get_cycles( void )
{
	register unsigned long ret;

	__asm__ __volatile__ ("isb; mrs %0, cntvct_el0" : "=r" (ret));

	return ret;
}

/************************/
/* POWER get_cycles()   */
/************************/

#elif defined(__powerpc__)

static inline long long get_cycles()
{
    uint64_t result;
	int64_t retval;
#ifdef _ARCH_PPC64
    /*
        This reads timebase in one 64bit go.  Does *not* include a workaround for the cell (see 
        http://ozlabs.org/pipermail/linuxppc-dev/2006-October/027052.html)
    */
    __asm__ volatile(
        "mftb    %0"
        : "=r" (result));
#else
    /*
        Read the high 32bits of the timer, then the lower, and repeat if high order has changed in the meantime.  See
        http://ozlabs.org/pipermail/linuxppc-dev/1999-October/003889.html
    */
    unsigned long dummy;
    __asm__ volatile(
        "mfspr   %1,269\n\t"  /* mftbu */
        "mfspr   %L0,268\n\t" /* mftb */
        "mfspr   %0,269\n\t"  /* mftbu */
        "cmpw    %0,%1\n\t"   /* check if the high order word has chanegd */
        "bne     $-16"
        : "=r" (result), "=r" (dummy));
#endif
    retval = (result*multiplier)/1000ULL;
    return retval;
}

#elif (defined(__arm__) || defined(__mips__) || defined(__hppa__)) || defined(__riscv)
static inline long long
get_cycles( void )
{
  return 0;
}

/************************/
/* NEC get_cycles()     */
/************************/

#elif defined(__NEC__)
static inline long long
get_cycles( void )
{
	long long ret = 0;
	return ret;
}



#elif !defined(HAVE_GETTIMEOFDAY) && !defined(HAVE_CLOCK_GETTIME)
#error "No get_cycles support for this architecture. "
#endif



long long
_linux_get_real_cycles( void )
{
	long long retval;
#if defined(HAVE_GETTIMEOFDAY)||defined(__arm__)||defined(__mips__)

	/* Crude estimate, not accurate in prescence of DVFS */

	retval = _papi_os_vector.get_real_usec(  ) *
		( long long ) _papi_hwi_system_info.hw_info.cpu_max_mhz;
#else
	retval = get_cycles(  );
#endif
	return retval;
}




/********************************************************************
 * microsecond timers                                               *
 ********************************************************************/


/*******************************
 * HAVE_CLOCK_GETTIME          *
 *******************************/

long long
_linux_get_real_usec_gettime( void )
{

   long long retval;

   struct timespec foo;
#ifdef HAVE_CLOCK_GETTIME_REALTIME_HR
   syscall( __NR_clock_gettime, CLOCK_REALTIME_HR, &foo );
#else
   syscall( __NR_clock_gettime, CLOCK_REALTIME, &foo );
#endif
   retval = ( long long ) foo.tv_sec * ( long long ) 1000000;
   retval += ( long long ) ( foo.tv_nsec / 1000 );

   return retval;
}

/**********************
 * HAVE_GETTIMEOFDAY  *
 **********************/

long long
_linux_get_real_usec_gettimeofday( void )
{

   long long retval;

   struct timeval buffer;
   gettimeofday( &buffer, NULL );
   retval = ( long long ) buffer.tv_sec * ( long long ) 1000000;
   retval += ( long long ) ( buffer.tv_usec );

   return retval;
}


long long
_linux_get_real_usec_cycles( void )
{

   long long retval;

   /* Not accurate in the prescence of DVFS */

   retval = get_cycles(  ) /
            ( long long ) _papi_hwi_system_info.hw_info.cpu_max_mhz;

   return retval;
}



/*******************************
 * HAVE_PER_THREAD_GETRUSAGE   *
 *******************************/

long long
_linux_get_virt_usec_rusage( void )
{

    long long retval;

    struct rusage buffer;

    getrusage( RUSAGE_SELF, &buffer );
    SUBDBG( "user %d system %d\n", ( int ) buffer.ru_utime.tv_sec,
				( int ) buffer.ru_stime.tv_sec );
    retval = ( long long ) ( buffer.ru_utime.tv_sec + buffer.ru_stime.tv_sec )
             * ( long long ) 1000000;
    retval += (long long) ( buffer.ru_utime.tv_usec + buffer.ru_stime.tv_usec );

    return retval;
}

/**************************
 * HAVE_PER_THREAD_TIMES  *
 **************************/

long long
_linux_get_virt_usec_times( void )
{

   long long retval;

   struct tms buffer;

   times( &buffer );

   SUBDBG( "user %d system %d\n", ( int ) buffer.tms_utime,
				( int ) buffer.tms_stime );
   retval = ( long long ) ( ( buffer.tms_utime + buffer.tms_stime ) *
			    1000000 / sysconf( _SC_CLK_TCK ));

   /* NOT CLOCKS_PER_SEC as in the headers! */

   return retval;
}

/******************************/
/* HAVE_CLOCK_GETTIME_THREAD  */
/******************************/

long long
_linux_get_virt_usec_gettime( void )
{

    long long retval;

    struct timespec foo;

    syscall( __NR_clock_gettime, CLOCK_THREAD_CPUTIME_ID, &foo );
    retval = ( long long ) foo.tv_sec * ( long long ) 1000000;
    retval += ( long long ) foo.tv_nsec / 1000;

    return retval;
}

/********************/
/* USE_PROC_PTTIMER */
/********************/

long long
_linux_get_virt_usec_pttimer( void )
{

   long long retval;
   char buf[LINE_MAX];
   long long utime, stime;
   int rv, cnt = 0, i = 0;
   int stat_fd;


again:
   sprintf( buf, "/proc/%d/task/%d/stat", getpid(  ), mygettid(  ) );
   stat_fd = open( buf, O_RDONLY );
   if ( stat_fd == -1 ) {
      PAPIERROR( "open(%s)", buf );
      return PAPI_ESYS;
   }

   rv = read( stat_fd, buf, LINE_MAX * sizeof ( char ) );
   if ( rv == -1 ) {
      if ( errno == EBADF ) {
	 close(stat_fd);
	 goto again;
      }
      PAPIERROR( "read()" );
      close(stat_fd);
      return PAPI_ESYS;
   }
   lseek( stat_fd, 0, SEEK_SET );

   if (rv == LINE_MAX) rv--;
   buf[rv] = '\0';
   SUBDBG( "Thread stat file is:%s\n", buf );
   while ( ( cnt != 13 ) && ( i < rv ) ) {
      if ( buf[i] == ' ' ) {
	 cnt++;
      }
      i++;
   }

   if ( cnt != 13 ) {
      PAPIERROR( "utime and stime not in thread stat file?" );
      close(stat_fd);
      return PAPI_ESYS;
   }

   if ( sscanf( buf + i, "%llu %llu", &utime, &stime ) != 2 ) {
      close(stat_fd);
      PAPIERROR("Unable to scan two items from thread stat file at 13th space?");
      return PAPI_ESYS;
   }

   retval = ( utime + stime ) * ( long long ) 1000000 /_papi_os_info.clock_ticks;

   close(stat_fd);

   return retval;
}


/********************************************************************
 * nanosecond timers                                                *
 ********************************************************************/



/*******************************
 * HAVE_CLOCK_GETTIME          *
 *******************************/

long long
_linux_get_real_nsec_gettime( void )
{

   long long retval;

   struct timespec foo;
#ifdef HAVE_CLOCK_GETTIME_REALTIME_HR
   syscall( __NR_clock_gettime, CLOCK_REALTIME_HR, &foo );
#else
   syscall( __NR_clock_gettime, CLOCK_REALTIME, &foo );
#endif
   retval = ( long long ) foo.tv_sec * ( long long ) 1000000000;
   retval += ( long long ) ( foo.tv_nsec );

   return retval;
}


/******************************/
/* HAVE_CLOCK_GETTIME_THREAD  */
/******************************/

long long
_linux_get_virt_nsec_gettime( void )
{

    long long retval;

    struct timespec foo;

    syscall( __NR_clock_gettime, CLOCK_THREAD_CPUTIME_ID, &foo );
    retval = ( long long ) foo.tv_sec * ( long long ) 1000000000;
    retval += ( long long ) foo.tv_nsec ;

    return retval;
}
