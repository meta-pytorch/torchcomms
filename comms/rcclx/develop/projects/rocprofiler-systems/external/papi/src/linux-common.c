/*
* File:    linux-common.c
*/

#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <err.h>
#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <syscall.h>
#include <sys/utsname.h>
#include <sys/time.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"

#include "linux-memory.h"
#include "linux-common.h"
#include "linux-timer.h"

#include "x86_cpuid_info.h"

PAPI_os_info_t _papi_os_info;

/* The locks used by Linux */

#if defined(USE_PTHREAD_MUTEXES)
pthread_mutex_t _papi_hwd_lock_data[PAPI_MAX_LOCK];
#elif defined(USE_LIBAO_ATOMICS)
AO_TS_t _papi_hwd_lock_data[PAPI_MAX_LOCK];
#else
volatile unsigned int _papi_hwd_lock_data[PAPI_MAX_LOCK];
#endif


static int _linux_init_locks(void) {

   int i;

   for ( i = 0; i < PAPI_MAX_LOCK; i++ ) {
#if defined(USE_PTHREAD_MUTEXES)
       pthread_mutex_init(&_papi_hwd_lock_data[i],NULL);
#elif defined(USE_LIBAO_ATOMICS)
       _papi_hwd_lock_data[i] = AO_TS_INITIALIZER;
#else
       _papi_hwd_lock_data[i] = MUTEX_OPEN;
#endif
   }

   return PAPI_OK;
}


int
_linux_detect_hypervisor(char *virtual_vendor_name) {

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
search_cpu_info( FILE * f, char *search_str)
{
	static char line[PAPI_HUGE_STR_LEN] = "";
	char *s, *start = NULL;

	rewind(f);

	while (fgets(line,PAPI_HUGE_STR_LEN,f)!=NULL) {
		s=strstr(line,search_str);
		if (s!=NULL) {
			/* skip all characters in line up to the colon */
			/* and then spaces */
			s=strchr(s,':');
			if (s==NULL) break;
			s++;
			while (isspace(*s)) {
				s++;
			}
			start = s;
			/* Find and clear newline */
			s=strrchr(start,'\n');
			if (s!=NULL) *s = 0;
			break;
		}
	}
	return start;
}

static void
decode_vendor_string( char *s, int *vendor )
{
	if ( strcasecmp( s, "GenuineIntel" ) == 0 )
		*vendor = PAPI_VENDOR_INTEL;
	else if ( ( strcasecmp( s, "AMD" ) == 0 ) ||
			  ( strcasecmp( s, "AuthenticAMD" ) == 0 ) )
		*vendor = PAPI_VENDOR_AMD;
	else if ( strcasecmp( s, "IBM" ) == 0 )
		*vendor = PAPI_VENDOR_IBM;
	else if ( strcasecmp( s, "Cray" ) == 0 )
		*vendor = PAPI_VENDOR_CRAY;
	else if ( strcasecmp( s, "ARM_ARM" ) == 0 )
		*vendor = PAPI_VENDOR_ARM_ARM;
	else if ( strcasecmp( s, "ARM_BROADCOM" ) == 0 )
		*vendor = PAPI_VENDOR_ARM_BROADCOM;
	else if ( strcasecmp( s, "ARM_CAVIUM" ) == 0 )
		*vendor = PAPI_VENDOR_ARM_CAVIUM;
	else if ( strcasecmp( s, "ARM_FUJITSU" ) == 0 )
		*vendor = PAPI_VENDOR_ARM_FUJITSU;
	else if ( strcasecmp( s, "ARM_HISILICON") == 0 )
		*vendor = PAPI_VENDOR_ARM_HISILICON;
	else if ( strcasecmp( s, "ARM_APM" ) == 0 )
		*vendor = PAPI_VENDOR_ARM_APM;
	else if ( strcasecmp( s, "ARM_QUALCOMM" ) == 0 )
		*vendor = PAPI_VENDOR_ARM_QUALCOMM;
	else if ( strcasecmp( s, "MIPS" ) == 0 )
		*vendor = PAPI_VENDOR_MIPS;
	else if ( strcasecmp( s, "SiCortex" ) == 0 )
		*vendor = PAPI_VENDOR_MIPS;
	else
		*vendor = PAPI_VENDOR_UNKNOWN;
}

static FILE *
xfopen( const char *path, const char *mode )
{
	FILE *fd = fopen( path, mode );
	if ( !fd )
		err( EXIT_FAILURE, "error: %s", path );
	return fd;
}

static FILE *
path_vfopen( const char *mode, const char *path, va_list ap )
{
	vsnprintf( pathbuf, sizeof ( pathbuf ), path, ap );
	return xfopen( pathbuf, mode );
}


static int
path_sibling( const char *path, ... )
{
	int c;
	long n;
	int result = 0;
	char s[2];
	FILE *fp;
	va_list ap;
	va_start( ap, path );
	fp = path_vfopen( "r", path, ap );
	va_end( ap );

	while ( ( c = fgetc( fp ) ) != EOF ) {
		if ( isxdigit( c ) ) {
			s[0] = ( char ) c;
			s[1] = '\0';
			for ( n = strtol( s, NULL, 16 ); n > 0; n /= 2 ) {
				if ( n % 2 )
					result++;
			}
		}
	}

	fclose( fp );
	return result;
}

static int
path_exist( const char *path, ... )
{
	va_list ap;
	va_start( ap, path );
	vsnprintf( pathbuf, sizeof ( pathbuf ), path, ap );
	va_end( ap );
	return access( pathbuf, F_OK ) == 0;
}

static int
decode_cpuinfo_x86( FILE *f, PAPI_hw_info_t *hwinfo )
{
	int tmp;
	unsigned int strSize;
	char *s;

	/* Stepping */
	s = search_cpu_info( f, "stepping");
	if ( s ) {
		if (sscanf( s, "%d", &tmp ) ==1 ) {
			hwinfo->revision = ( float ) tmp;
			hwinfo->cpuid_stepping = tmp;
		}
	}

	/* Model Name */
	s = search_cpu_info( f, "model name");
	strSize = sizeof(hwinfo->model_string);
	if ( s ) {
		strncpy( hwinfo->model_string, s, strSize - 1);
	}

	/* Family */
	s = search_cpu_info( f, "cpu family");
	if ( s ) {
		sscanf( s, "%d", &tmp );
		hwinfo->cpuid_family = tmp;
	}


	/* CPU Model */
	s = search_cpu_info( f, "model");
	if ( s ) {
		sscanf( s , "%d", &tmp );
		hwinfo->model = tmp;
		hwinfo->cpuid_model = tmp;
	}

	return PAPI_OK;
}

static int
decode_cpuinfo_power(FILE *f, PAPI_hw_info_t *hwinfo )
{

	int tmp;
	unsigned int strSize;
	char *s;

	/* Revision */
	s = search_cpu_info( f, "revision");
	if ( s ) {
		sscanf( s, "%d", &tmp );
		hwinfo->revision = ( float ) tmp;
		hwinfo->cpuid_stepping = tmp;
	}

       /* Model Name */
	s = search_cpu_info( f, "model");
	strSize = sizeof(hwinfo->model_string);
	if ( s ) {
		strncpy( hwinfo->model_string, s, strSize - 1);
	}

	return PAPI_OK;
}



static int
decode_cpuinfo_arm(FILE *f, PAPI_hw_info_t *hwinfo )
{

	int tmp;
	unsigned int strSize;
	char *s, *t;

	/* revision */
	s = search_cpu_info( f, "CPU revision");
	if ( s ) {
		sscanf( s, "%d", &tmp );
		hwinfo->revision = ( float ) tmp;
		/* For compatability with old PAPI */
		hwinfo->model = tmp;
	}

	/* Model Name */
	s = search_cpu_info( f, "model name");
	strSize = sizeof(hwinfo->model_string);
	if ( s ) {
		strncpy( hwinfo->model_string, s, strSize - 1);
	}

	/* Architecture (ARMv6, ARMv7, ARMv8, etc.) */

	/* Parsing this is a bit fragile. */
	/* On ARM64 the "CPU architecture field" */
	/*     Prior to Linux 3.19: always "AArch64" */
	/*     Since Linux 3.19: always "8" */
	/* On ARM32 the "CPU architecture field" is a value and not */
	/*	necessarily an integer, so it might be 7 or 7M */
	/*	also, unknown architectures are assigned a value */
	/*	such as (10) where 10 does not mean version 10, just */
	/*	the 10th element in an array */
	/* Note the original Raspberry Pi lies in the CPU architecture line */
	/* (it's ARMv6 not ARMv7)                                  */
	/* So we should actually get the value from the            */
	/*	Processor/ model name line                         */


	s = search_cpu_info( f, "CPU architecture");
	if ( s ) {

		/* Handle old (prior to Linux 3.19) ARM64 */
		if (strstr(s,"AArch64")) {
			hwinfo->cpuid_family = 8;
		}
		else {
			hwinfo->cpuid_family=strtol(s, NULL, 10);
		}

		/* Old Fallbacks if the above didn't work */
		if (hwinfo->cpuid_family<0) {

			/* Try the processor field and look inside of parens */
			s = search_cpu_info( f, "Processor" );
			if (s) {
				t=strchr(s,'(');
				tmp=*(t+2)-'0';
				hwinfo->cpuid_family = tmp;
			}
			/* Try the model name and look inside of parens */
			else {
				s = search_cpu_info( f, "model name" );
				if (s) {
					t=strchr(s,'(');
					tmp=*(t+2)-'0';
					hwinfo->cpuid_family = tmp;
				}
			}
		}
	}

	/* CPU Model */
	s = search_cpu_info( f, "CPU part" );
	if ( s ) {
		sscanf( s, "%x", &tmp );
		hwinfo->cpuid_model = tmp;
	}

	/* CPU Variant */
	s = search_cpu_info( f, "CPU variant" );
	if ( s ) {
		sscanf( s, "%x", &tmp );
		hwinfo->cpuid_stepping = tmp;
	}

	return PAPI_OK;

}


int
_linux_get_cpu_info( PAPI_hw_info_t *hwinfo, int *cpuinfo_mhz )
{
	int retval = PAPI_OK;
	char *s;
	float mhz = 0.0;
	FILE *f;
	char cpuinfo_filename[]="/proc/cpuinfo";

	if ( ( f = fopen( cpuinfo_filename, "r" ) ) == NULL ) {
		PAPIERROR( "fopen(/proc/cpuinfo) errno %d", errno );
		return PAPI_ESYS;
	}

	/* All of this information may be overwritten by the component */

	/***********************/
	/* Attempt to find MHz */
	/***********************/
	s = search_cpu_info( f, "cpu MHz" );
	if ( !s ) {
		s = search_cpu_info( f, "clock" );
	}
	if ( s ) {
		sscanf( s, "%f", &mhz );
		*cpuinfo_mhz = mhz;
	}
	else {
      *cpuinfo_mhz = -1;   // Could not find it.
	//	PAPIWARN("Failed to find a clock speed in /proc/cpuinfo");
	}

	/*******************************/
	/* Vendor Name and Vendor Code */
	/*******************************/

	/* First try to read "vendor_id" field */
	/* Which is the most common field      */
	s = search_cpu_info( f, "vendor_id");
	if ( s ) {
		strncpy( hwinfo->vendor_string, s, PAPI_MAX_STR_LEN );
      hwinfo->vendor_string[PAPI_MAX_STR_LEN-1]=0;
	}
	else {
		/* If not found, try "vendor" which seems to be Itanium specific */
		s = search_cpu_info( f, "vendor" );
		if ( s ) {
		   strncpy( hwinfo->vendor_string, s, PAPI_MAX_STR_LEN );
         hwinfo->vendor_string[PAPI_MAX_STR_LEN-1]=0;
		}
		else {
			/* "system type" seems to be MIPS and Alpha */
			s = search_cpu_info( f, "system type");
			if ( s ) {
		      strncpy( hwinfo->vendor_string, s, PAPI_MAX_STR_LEN );
            hwinfo->vendor_string[PAPI_MAX_STR_LEN-1]=0;
			}
			else {
				/* "platform" indicates Power */
				s = search_cpu_info( f, "platform");
				if ( s ) {
					if ( ( strcasecmp( s, "pSeries" ) == 0 ) ||
						( strcasecmp( s, "PowerNV" ) == 0 ) ||
						( strcasecmp( s, "PowerMac" ) == 0 ) ) {
						strcpy( hwinfo->vendor_string, "IBM" );
					}
				}
				else {
					/* "CPU implementer" indicates ARM */
					/* For ARM processors, hwinfo->vendor >= PAPI_VENDOR_ARM_ARM(0x41). */
					/* If implementer is ARM Limited., hwinfo->vendor == PAPI_VENDOR_ARM_ARM. */
					/* If implementer is Cavium Inc., hwinfo->vendor == PAPI_VENDOR_ARM_CAVIUM(0x43). */
					s = search_cpu_info( f, "CPU implementer");
					if ( s ) {
						int tmp;
						sscanf( s, "%x", &tmp );
						switch( tmp ) {
						case PAPI_VENDOR_ARM_ARM:
							strcpy( hwinfo->vendor_string, "ARM_ARM" );
							break;
						case PAPI_VENDOR_ARM_BROADCOM:
							strcpy( hwinfo->vendor_string, "ARM_BROADCOM" );
							break;
						case PAPI_VENDOR_ARM_CAVIUM:
							strcpy( hwinfo->vendor_string, "ARM_CAVIUM" );
							break;
						case PAPI_VENDOR_ARM_FUJITSU:
							strcpy( hwinfo->vendor_string, "ARM_FUJITSU" );
							break;
						case PAPI_VENDOR_ARM_HISILICON:
							strcpy( hwinfo->vendor_string, "ARM_HISILICON" );
							break;
						case PAPI_VENDOR_ARM_APM:
							strcpy( hwinfo->vendor_string, "ARM_APM" );
							break;
						case PAPI_VENDOR_ARM_QUALCOMM:
							strcpy( hwinfo->vendor_string, "ARM_QUALCOMM" );
							break;
						default:
							strcpy( hwinfo->vendor_string, "ARM_UNKNOWN" );
						}
					}
				}
			}
		}
	}

	/* Decode the string to a PAPI specific implementer value */
	if ( strlen( hwinfo->vendor_string ) ) {
		decode_vendor_string( hwinfo->vendor_string, &hwinfo->vendor );
	}

	/**********************************************/
	/* Provide more stepping/model/family numbers */
	/**********************************************/

	if ((hwinfo->vendor==PAPI_VENDOR_INTEL) ||
		(hwinfo->vendor==PAPI_VENDOR_AMD)) {

		decode_cpuinfo_x86(f,hwinfo);
	}

	if (hwinfo->vendor==PAPI_VENDOR_IBM) {

		decode_cpuinfo_power(f,hwinfo);
	}

	if (hwinfo->vendor>=PAPI_VENDOR_ARM_ARM) {

		decode_cpuinfo_arm(f,hwinfo);
	}




	/* The following members are set using the same methodology */
	/* used in lscpu.                                           */

	/* Total number of CPUs */
	/* The following line assumes totalcpus was initialized to zero! */
	while ( path_exist( _PATH_SYS_SYSTEM "/cpu/cpu%d", hwinfo->totalcpus ) )
		hwinfo->totalcpus++;

	/* Number of threads per core */
	if ( path_exist( _PATH_SYS_CPU0 "/topology/thread_siblings" ) )
		hwinfo->threads =
			path_sibling( _PATH_SYS_CPU0 "/topology/thread_siblings" );

	/* Number of cores per socket */
	if ( path_exist( _PATH_SYS_CPU0 "/topology/core_siblings" ) &&
		 hwinfo->threads > 0 )
		hwinfo->cores =
			path_sibling( _PATH_SYS_CPU0 "/topology/core_siblings" ) /
			hwinfo->threads;

	/* Number of NUMA nodes */
	/* The following line assumes nnodes was initialized to zero! */
	while ( path_exist( _PATH_SYS_SYSTEM "/node/node%d", hwinfo->nnodes ) ) {
		hwinfo->nnodes++;
	}

	/* Number of CPUs per node */
	hwinfo->ncpu = hwinfo->nnodes > 1 ?
			hwinfo->totalcpus / hwinfo->nnodes : hwinfo->totalcpus;

	/* Number of sockets */
	if ( hwinfo->threads > 0 && hwinfo->cores > 0 ) {
		hwinfo->sockets = hwinfo->totalcpus / hwinfo->cores / hwinfo->threads;
	}

#if 0
	int *nodecpu;
	/* cpumap data is not currently part of the _papi_hw_info struct */
        nodecpu = malloc( (unsigned int) hwinfo->nnodes * sizeof(int) );
	if ( nodecpu ) {
	   int i;
	   for ( i = 0; i < hwinfo->nnodes; ++i ) {
	       nodecpu[i] = path_sibling(
                             _PATH_SYS_SYSTEM "/node/node%d/cpumap", i );
	   }
	} else {
		PAPIERROR( "malloc failed for variable not currently used" );
	}
#endif


	/* Fixup missing Megahertz Value */
	/* This is missing from cpuinfo on ARM and MIPS */
	if (*cpuinfo_mhz < 1.0) {
		s = search_cpu_info( f, "BogoMIPS" );
		if ((!s) || (sscanf( s, "%f", &mhz ) != 1)) {
			INTDBG("MHz detection failed. "
				"Please edit file %s at line %d.\n",
				__FILE__,__LINE__);
		}

		if (hwinfo->vendor == PAPI_VENDOR_MIPS) {
			/* MIPS has 2x clock multiplier */
			*cpuinfo_mhz = 2*(((int)mhz)+1);

			/* Also update version info on MIPS */
			s = search_cpu_info( f, "cpu model");
			s = strstr(s," V")+2;
			strtok(s," ");
			sscanf(s, "%f ", &hwinfo->revision );
		}
		else {
			/* In general bogomips is proportional to number of CPUs */
			if (hwinfo->totalcpus) {
				if (mhz!=0) *cpuinfo_mhz = mhz / hwinfo->totalcpus;
			}
		}
	}

	fclose( f );

	return retval;
}

int
_linux_get_mhz( int *sys_min_mhz, int *sys_max_mhz ) {

  FILE *fff;
  int result;

  /* Try checking for min MHz */
  /* Assume cpu0 exists */
  fff=fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_min_freq","r");
  if (fff==NULL) return PAPI_EINVAL;
  result=fscanf(fff,"%d",sys_min_mhz);
  fclose(fff);
  if (result!=1) return PAPI_EINVAL;

  fff=fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq","r");
  if (fff==NULL) return PAPI_EINVAL;
  result=fscanf(fff,"%d",sys_max_mhz);
  fclose(fff);
  if (result!=1) return PAPI_EINVAL;

  return PAPI_OK;

}

int
_linux_get_system_info( papi_mdi_t *mdi ) {

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

	sprintf( maxargs, "/proc/%d/exe", ( int ) pid );
	retval = readlink( maxargs, mdi->exe_info.fullname,
			PAPI_HUGE_STR_LEN-1 );
	if ( retval < 0 ) {
		PAPIERROR( "readlink(%s) returned < 0", maxargs );
		return PAPI_ESYS;
	}

	if (retval > PAPI_HUGE_STR_LEN-1) {
		retval=PAPI_HUGE_STR_LEN-1;
	}
	mdi->exe_info.fullname[retval] = '\0';

	/* Careful, basename can modify its argument */
	strcpy( maxargs, mdi->exe_info.fullname );

	strncpy( mdi->exe_info.address_info.name, basename( maxargs ),
		PAPI_HUGE_STR_LEN-1);
	mdi->exe_info.address_info.name[PAPI_HUGE_STR_LEN-1] = '\0';

	SUBDBG( "Executable is %s\n", mdi->exe_info.address_info.name );
	SUBDBG( "Full Executable is %s\n", mdi->exe_info.fullname );

	/* Executable regions, may require reading /proc/pid/maps file */

	retval = _linux_update_shlib_info( mdi );
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

	/* PAPI_preload_option information */

	strcpy( mdi->preload_info.lib_preload_env, "LD_PRELOAD" );
	mdi->preload_info.lib_preload_sep = ' ';
	strcpy( mdi->preload_info.lib_dir_env, "LD_LIBRARY_PATH" );
	mdi->preload_info.lib_dir_sep = ':';

	/* Hardware info */

	retval = _linux_get_cpu_info( &mdi->hw_info, &cpuinfo_mhz );
	if ( retval )
		return retval;

	/* Handle MHz */

	retval = _linux_get_mhz( &sys_min_khz, &sys_max_khz );
	if ( retval ) {

		mdi->hw_info.cpu_max_mhz=cpuinfo_mhz;
		mdi->hw_info.cpu_min_mhz=cpuinfo_mhz;

	   /*
	   mdi->hw_info.mhz=cpuinfo_mhz;
	   mdi->hw_info.clock_mhz=cpuinfo_mhz;
	   */
	}
	else {
		mdi->hw_info.cpu_max_mhz=sys_max_khz/1000;
		mdi->hw_info.cpu_min_mhz=sys_min_khz/1000;

	   /*
	   mdi->hw_info.mhz=sys_max_khz/1000;
	   mdi->hw_info.clock_mhz=sys_max_khz/1000;
	   */
	}

	/* Set Up Memory */

	retval = _linux_get_memory_info( &mdi->hw_info, mdi->hw_info.model );
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
	mdi->hw_info.virtualized=_linux_detect_hypervisor(mdi->hw_info.virtual_vendor_string);

	return PAPI_OK;
}

int
_papi_hwi_init_os(void) {

    int major=0,minor=0,sub=0;
    char *ptr;
    struct utsname uname_buffer;

    /* Initialize the locks */
    _linux_init_locks();

    /* Get the kernel info */
    uname(&uname_buffer);

    SUBDBG("Native kernel version %s\n",uname_buffer.release);

    strncpy(_papi_os_info.name,uname_buffer.sysname,PAPI_MAX_STR_LEN);

#ifdef ASSUME_KERNEL
    strncpy(_papi_os_info.version,ASSUME_KERNEL,PAPI_MAX_STR_LEN);
    SUBDBG("Assuming kernel version %s\n",_papi_os_info.name);
#else
    strncpy(_papi_os_info.version,uname_buffer.release,PAPI_MAX_STR_LEN);
#endif

    ptr=strtok(_papi_os_info.version,".");
    if (ptr!=NULL) major=atoi(ptr);

    ptr=strtok(NULL,".");
    if (ptr!=NULL) minor=atoi(ptr);

    ptr=strtok(NULL,".");
    if (ptr!=NULL) sub=atoi(ptr);

   _papi_os_info.os_version=LINUX_VERSION(major,minor,sub);

   _papi_os_info.itimer_sig = PAPI_INT_MPX_SIGNAL;
   _papi_os_info.itimer_num = PAPI_INT_ITIMER;
   _papi_os_info.itimer_ns = PAPI_INT_MPX_DEF_US * 1000;
   _papi_os_info.itimer_res_ns = 1;
   _papi_os_info.clock_ticks = sysconf( _SC_CLK_TCK );

   /* Get Linux-specific system info */
   _linux_get_system_info( &_papi_hwi_system_info );

   return PAPI_OK;
}



int _linux_detect_nmi_watchdog() {

  int watchdog_detected=0,watchdog_value=0;
  FILE *fff;

  fff=fopen("/proc/sys/kernel/nmi_watchdog","r");
  if (fff!=NULL) {
     if (fscanf(fff,"%d",&watchdog_value)==1) {
        if (watchdog_value>0) watchdog_detected=1;
     }
     fclose(fff);
  }

  return watchdog_detected;
}

papi_os_vector_t _papi_os_vector = {
  .get_memory_info =   _linux_get_memory_info,
  .get_dmem_info =     _linux_get_dmem_info,
  .get_real_cycles =   _linux_get_real_cycles,
  .update_shlib_info = _linux_update_shlib_info,
  .get_system_info =   _linux_get_system_info,


#if defined(HAVE_CLOCK_GETTIME)
  .get_real_usec =  _linux_get_real_usec_gettime,
#elif defined(HAVE_GETTIMEOFDAY)
  .get_real_usec =  _linux_get_real_usec_gettimeofday,
#else
  .get_real_usec =  _linux_get_real_usec_cycles,
#endif


#if defined(USE_PROC_PTTIMER)
  .get_virt_usec =   _linux_get_virt_usec_pttimer,
#elif defined(HAVE_CLOCK_GETTIME_THREAD)
  .get_virt_usec =   _linux_get_virt_usec_gettime,
#elif defined(HAVE_PER_THREAD_TIMES)
  .get_virt_usec =   _linux_get_virt_usec_times,
#elif defined(HAVE_PER_THREAD_GETRUSAGE)
  .get_virt_usec =   _linux_get_virt_usec_rusage,
#endif


#if defined(HAVE_CLOCK_GETTIME)
  .get_real_nsec =  _linux_get_real_nsec_gettime,
#endif

#if defined(HAVE_CLOCK_GETTIME_THREAD)
  .get_virt_nsec =   _linux_get_virt_nsec_gettime,
#endif


};
