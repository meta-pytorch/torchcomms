#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <err.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <sys/utsname.h>
#include <sys/sysctl.h>

#include <time.h>
#include <sys/time.h>
#include <sys/times.h>


#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"

#include "darwin-memory.h"
#include "darwin-common.h"

#include "x86_cpuid_info.h"

PAPI_os_info_t _papi_os_info;

/* The locks used by Darwin */

#if defined(USE_PTHREAD_MUTEXES)
pthread_mutex_t _papi_hwd_lock_data[PAPI_MAX_LOCK];
#else
volatile unsigned int _papi_hwd_lock_data[PAPI_MAX_LOCK];
#endif


static int _darwin_init_locks(void) {

   int i;

   for ( i = 0; i < PAPI_MAX_LOCK; i++ ) {
#if defined(USE_PTHREAD_MUTEXES)
       pthread_mutex_init(&_papi_hwd_lock_data[i],NULL);
#else
       _papi_hwd_lock_data[i] = MUTEX_OPEN;
#endif
   }

   return PAPI_OK;
}

	
int
_darwin_detect_hypervisor(char *virtual_vendor_name) {

	int retval=0;

#if defined(__i386__)||defined(__x86_64__)
	retval=_x86_detect_hypervisor(virtual_vendor_name);
#else
	(void) virtual_vendor_name;
#endif
	
	return retval;
}


#define _PATH_SYS_SYSTEM "/sys/devices/system"
#define _PATH_SYS_CPU0	 _PATH_SYS_SYSTEM "/cpu/cpu0"

static char pathbuf[PATH_MAX] = "/";


static char *
search_cpu_info( FILE * f, char *search_str, char *line )
{
	/* This function courtesy of Rudolph Berrendorf! */
	/* See the home page for the German version of PAPI. */
	char *s;

	while ( fgets( line, 256, f ) != NULL ) {
		if ( strstr( line, search_str ) != NULL ) {
			/* ignore all characters in line up to : */
			for ( s = line; *s && ( *s != ':' ); ++s );
			if ( *s )
				return s;
		}
	}
	return NULL;
}



int
_darwin_get_cpu_info( PAPI_hw_info_t *hwinfo, int *cpuinfo_mhz )
{

  int mib[4];
  size_t len;
  char buffer[BUFSIZ];
  long long ll;

  /* "sysctl -a" shows lots of info we can get on OSX */

  /**********/
  /* Vendor */
  /**********/
  len = 3;
  sysctlnametomib("machdep.cpu.vendor", mib, &len);

  len = BUFSIZ;
  if (sysctl(mib, 3, &buffer, &len, NULL, 0) == -1) {
     return PAPI_ESYS;
  }
  strncpy( hwinfo->vendor_string,buffer,len);
   
  hwinfo->vendor = PAPI_VENDOR_INTEL;


  /**************/
  /* Model Name */
  /**************/
  len = 3;
  sysctlnametomib("machdep.cpu.brand_string", mib, &len);

  len = BUFSIZ;
  if (sysctl(mib, 3, &buffer, &len, NULL, 0) == -1) {
     return PAPI_ESYS;
  }
  strncpy( hwinfo->model_string,buffer,len);

  /************/
  /* Revision */
  /************/
  len = 3;
  sysctlnametomib("machdep.cpu.stepping", mib, &len);

  len = BUFSIZ;
  if (sysctl(mib, 3, &buffer, &len, NULL, 0) == -1) {
     return PAPI_ESYS;
  }

  hwinfo->cpuid_stepping=buffer[0];
  hwinfo->revision=(float)(hwinfo->cpuid_stepping);

  /**********/
  /* Family */
  /**********/
  len = 3;
  sysctlnametomib("machdep.cpu.family", mib, &len);

  len = BUFSIZ;
  if (sysctl(mib, 3, &buffer, &len, NULL, 0) == -1) {
     return PAPI_ESYS;
  }

  hwinfo->cpuid_family=buffer[0];

  /**********/
  /* Model  */
  /**********/
  len = 3;
  sysctlnametomib("machdep.cpu.model", mib, &len);

  len = BUFSIZ;
  if (sysctl(mib, 3, &buffer, &len, NULL, 0) == -1) {
     return PAPI_ESYS;
  }

  hwinfo->cpuid_model=buffer[0];
  hwinfo->model=hwinfo->cpuid_model;
   
  /*************/
  /* Frequency */
  /*************/
  len = 2;
  sysctlnametomib("hw.cpufrequency_max", mib, &len);

  len = 8;
  if (sysctl(mib, 2, &ll, &len, NULL, 0) == -1) {
     return PAPI_ESYS;
  }

  hwinfo->cpu_max_mhz=(int)(ll/(1000*1000));

  len = 2;
  sysctlnametomib("hw.cpufrequency_min", mib, &len);

  len = 8;
  if (sysctl(mib, 2, &ll, &len, NULL, 0) == -1) {
     return PAPI_ESYS;
  }

  hwinfo->cpu_min_mhz=(int)(ll/(1000*1000));

  /**********/
  /* ncpu   */
  /**********/
  len = 2;
  sysctlnametomib("hw.ncpu", mib, &len);

  len = BUFSIZ;
  if (sysctl(mib, 2, &buffer, &len, NULL, 0) == -1) {
     return PAPI_ESYS;
  }

  hwinfo->totalcpus=buffer[0];


  return PAPI_OK;
}


int
_darwin_get_system_info( papi_mdi_t *mdi ) {

	int retval;

	char maxargs[PAPI_HUGE_STR_LEN];
	pid_t pid;

	int cpuinfo_mhz,sys_min_khz,sys_max_khz;

	/* Software info */

	/* Path and args */

	pid = getpid(  );
	if ( pid < 0 ) {
		PAPIERROR( "getpid() returned < 0" );
		return PAPI_ESYS;
	}
	mdi->pid = pid;

#if 0
	sprintf( maxargs, "/proc/%d/exe", ( int ) pid );
	if ( readlink( maxargs, mdi->exe_info.fullname, PAPI_HUGE_STR_LEN ) < 0 ) {
		PAPIERROR( "readlink(%s) returned < 0", maxargs );
		return PAPI_ESYS;
	}

	/* Careful, basename can modify it's argument */

	strcpy( maxargs, mdi->exe_info.fullname );
	strcpy( mdi->exe_info.address_info.name, basename( maxargs ) );

	SUBDBG( "Executable is %s\n", mdi->exe_info.address_info.name );
	SUBDBG( "Full Executable is %s\n", mdi->exe_info.fullname );

	/* Executable regions, may require reading /proc/pid/maps file */

	retval = _darwin_update_shlib_info( mdi );
	SUBDBG( "Text: Start %p, End %p, length %d\n",
			mdi->exe_info.address_info.text_start,
			mdi->exe_info.address_info.text_end,
			( int ) ( mdi->exe_info.address_info.text_end -
					  mdi->exe_info.address_info.text_start ) );
	SUBDBG( "Data: Start %p, End %p, length %d\n",
			mdi->exe_info.address_info.data_start,
			mdi->exe_info.address_info.data_end,
			( int ) ( mdi->exe_info.address_info.data_end -
					  mdi->exe_info.address_info.data_start ) );
	SUBDBG( "Bss: Start %p, End %p, length %d\n",
			mdi->exe_info.address_info.bss_start,
			mdi->exe_info.address_info.bss_end,
			( int ) ( mdi->exe_info.address_info.bss_end -
					  mdi->exe_info.address_info.bss_start ) );
#endif
	/* PAPI_preload_option information */

	strcpy( mdi->preload_info.lib_preload_env, "LD_PRELOAD" );
	mdi->preload_info.lib_preload_sep = ' ';
	strcpy( mdi->preload_info.lib_dir_env, "LD_LIBRARY_PATH" );
	mdi->preload_info.lib_dir_sep = ':';

	/* Hardware info */

	retval = _darwin_get_cpu_info( &mdi->hw_info, &cpuinfo_mhz );
	if ( retval ) {
		return retval;
	}

	/* Set Up Memory */

	retval = _darwin_get_memory_info( &mdi->hw_info, mdi->hw_info.model );
	if ( retval )
		return retval;

	SUBDBG( "Found %d %s(%d) %s(%d) CPUs at %d Mhz.\n",
			mdi->hw_info.totalcpus,
			mdi->hw_info.vendor_string,
			mdi->hw_info.vendor, 
		        mdi->hw_info.model_string, 
		        mdi->hw_info.model,
		        mdi->hw_info.cpu_max_mhz);

	/* Get virtualization info */
	mdi->hw_info.virtualized=_darwin_detect_hypervisor(mdi->hw_info.virtual_vendor_string);

	return PAPI_OK;
}

int 
_papi_hwi_init_os(void) {

    int major=0,minor=0,sub=0;
    char *ptr;
    struct utsname uname_buffer;

    /* Initialize the locks */
    _darwin_init_locks();

    /* Get the kernel info */
    uname(&uname_buffer);

    SUBDBG("Native kernel version %s\n",uname_buffer.release);

    strncpy(_papi_os_info.name,uname_buffer.sysname,PAPI_MAX_STR_LEN);

    strncpy(_papi_os_info.version,uname_buffer.release,PAPI_MAX_STR_LEN);

    ptr=strtok(_papi_os_info.version,".");
    if (ptr!=NULL) major=atoi(ptr);

    ptr=strtok(NULL,".");
    if (ptr!=NULL) minor=atoi(ptr);

    ptr=strtok(NULL,".");
    if (ptr!=NULL) sub=atoi(ptr);

    //   _papi_os_info.os_version=LINUX_VERSION(major,minor,sub);

   _papi_os_info.itimer_sig = PAPI_INT_MPX_SIGNAL;
   _papi_os_info.itimer_num = PAPI_INT_ITIMER;
   _papi_os_info.itimer_ns = PAPI_INT_MPX_DEF_US * 1000;
   _papi_os_info.itimer_res_ns = 1;
   _papi_os_info.clock_ticks = sysconf( _SC_CLK_TCK );

   /* Get Darwin-specific system info */
   _darwin_get_system_info( &_papi_hwi_system_info );

   return PAPI_OK;
}


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

long long
_darwin_get_real_cycles( void )
{
	long long retval;

	retval = get_cycles(  );

	return retval;
}


long long
_darwin_get_real_usec_gettimeofday( void )
{
	
   long long retval;

   struct timeval buffer;
   gettimeofday( &buffer, NULL );
   retval = ( long long ) buffer.tv_sec * ( long long ) 1000000;
   retval += ( long long ) ( buffer.tv_usec );
	
   return retval;
}


long long
_darwin_get_virt_usec_times( void )
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





papi_os_vector_t _papi_os_vector = {
  .get_memory_info =   _darwin_get_memory_info,
  .get_dmem_info =     _darwin_get_dmem_info,
  .get_real_cycles =   _darwin_get_real_cycles,
  .update_shlib_info = _darwin_update_shlib_info,
  .get_system_info =   _darwin_get_system_info,

  .get_real_usec =  _darwin_get_real_usec_gettimeofday,
  .get_virt_usec =  _darwin_get_virt_usec_times,

};
