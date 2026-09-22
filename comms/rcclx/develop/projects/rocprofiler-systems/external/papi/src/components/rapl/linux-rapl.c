/**
 * @file    linux-rapl.c
 * @author  Vince Weaver
 *
 * @ingroup papi_components
 *
 * @brief rapl component
 *
 *  This component enables RAPL (Running Average Power Level)
 *  energy measurements on Intel SandyBridge/IvyBridge/Haswell
 *
 *  To work, either msr_safe kernel module from LLNL
 *  (https://github.com/scalability-llnl/msr-safe), or
 *  the x86 generic MSR driver must be installed
 *    (CONFIG_X86_MSR) and the /dev/cpu/?/<msr_safe | msr> files must have read permissions
 */

#include <stdio.h>
#include <unistd.h>
#include <dirent.h>
#include <fcntl.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

// The following macro follows if a string function has an error. It should 
// never happen; but it is necessary to prevent compiler warnings. We print 
// something just in case there is programmer error in invoking the function.
#define HANDLE_STRING_ERROR {fprintf(stderr,"%s:%i unexpected string function error.\n",__FILE__,__LINE__); exit(-1);}


/***************/
/* AMD Support */
/***************/
#define MSR_AMD_RAPL_POWER_UNIT                 0xc0010299

#define MSR_AMD_PKG_ENERGY_STATUS               0xc001029B
#define MSR_AMD_PP0_ENERGY_STATUS               0xc001029A


/*****************/
/* Intel support */
/*****************/

/*
 * Platform specific RAPL Domains.
 * Note that PP1 RAPL Domain is supported on 062A only
 * And DRAM RAPL Domain is supported on 062D only
 */

/* RAPL defines */
#define MSR_INTEL_RAPL_POWER_UNIT	0x606

/* Package */
#define MSR_PKG_RAPL_POWER_LIMIT        0x610
#define MSR_INTEL_PKG_ENERGY_STATUS     0x611
#define MSR_PKG_PERF_STATUS             0x613
#define MSR_PKG_POWER_INFO              0x614

/* PP0 */
#define MSR_PP0_POWER_LIMIT             0x638
#define MSR_INTEL_PP0_ENERGY_STATUS     0x639
#define MSR_PP0_POLICY                  0x63A
#define MSR_PP0_PERF_STATUS             0x63B

/* PP1 */
#define MSR_PP1_POWER_LIMIT             0x640
#define MSR_PP1_ENERGY_STATUS           0x641
#define MSR_PP1_POLICY                  0x642

/* DRAM */
#define MSR_DRAM_POWER_LIMIT            0x618
#define MSR_DRAM_ENERGY_STATUS          0x619
#define MSR_DRAM_PERF_STATUS            0x61B
#define MSR_DRAM_POWER_INFO             0x61C

/* PSYS RAPL Domain */
#define MSR_PLATFORM_ENERGY_STATUS      0x64d

/* RAPL bitsmasks */
#define POWER_UNIT_OFFSET          0
#define POWER_UNIT_MASK         0x0f

#define ENERGY_UNIT_OFFSET      0x08
#define ENERGY_UNIT_MASK        0x1f

#define TIME_UNIT_OFFSET        0x10
#define TIME_UNIT_MASK          0x0f

/* RAPL POWER UNIT MASKS */
#define POWER_INFO_UNIT_MASK     0x7fff
#define THERMAL_SHIFT                 0
#define MINIMUM_POWER_SHIFT          16
#define MAXIMUM_POWER_SHIFT          32
#define MAXIMUM_TIME_WINDOW_SHIFT    48


typedef struct _rapl_register
{
	unsigned int selector;
} _rapl_register_t;

typedef struct _rapl_native_event_entry
{
  char name[PAPI_MAX_STR_LEN];
  char units[PAPI_MIN_STR_LEN];
  char description[PAPI_MAX_STR_LEN];
  int fd_offset;
  int msr;
  int type;
  int return_type;
  _rapl_register_t resources;
} _rapl_native_event_entry_t;

typedef struct _rapl_reg_alloc
{
	_rapl_register_t ra_bits;
} _rapl_reg_alloc_t;

/* actually 32?  But setting this to be safe? */
#define RAPL_MAX_COUNTERS 64

typedef struct _rapl_control_state
{
  int being_measured[RAPL_MAX_COUNTERS];
  long long count[RAPL_MAX_COUNTERS];
  int need_difference[RAPL_MAX_COUNTERS];
  long long lastupdate;
} _rapl_control_state_t;

// The _ENERGY_ counters should return a monotonically increasing
// value from the _start point, but the hardware only returns a
// uint32_t that may wrap. We keep a start_value which is reset at
// _start and every read, handle overflows of the uint32_t, and
// accumulate a uint64_t which we return.

typedef struct _rapl_context
{
  long long start_value[RAPL_MAX_COUNTERS];
  long long accumulated_value[RAPL_MAX_COUNTERS];
  _rapl_control_state_t state;
} _rapl_context_t;


papi_vector_t _rapl_vector;

struct fd_array_t {
  int fd;
  int open;
};

static _rapl_native_event_entry_t * rapl_native_events=NULL;
static int num_events		= 0;
struct fd_array_t *fd_array=NULL;
static int num_packages=0,num_cpus=0;

int power_divisor,time_divisor;
int cpu_energy_divisor,dram_energy_divisor;
unsigned int msr_rapl_power_unit;

#define PACKAGE_ENERGY      	0
#define PACKAGE_THERMAL     	1
#define PACKAGE_MINIMUM     	2
#define PACKAGE_MAXIMUM     	3
#define PACKAGE_TIME_WINDOW 	4
#define PACKAGE_ENERGY_CNT      5
#define PACKAGE_THERMAL_CNT     6
#define PACKAGE_MINIMUM_CNT     7
#define PACKAGE_MAXIMUM_CNT     8
#define PACKAGE_TIME_WINDOW_CNT 9
#define DRAM_ENERGY		10
#define PLATFORM_ENERGY		11

/***************************************************************************/
/******  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT *******/
/***************************************************************************/


static long long read_msr(int fd, unsigned int which) {

	uint64_t data;

	if ( fd<0 || pread(fd, &data, sizeof data, which) != sizeof data ) {
		perror("rdmsr:pread");
		fprintf(stderr,"rdmsr error, msr %x\n",which);
		exit(127);
	}

	return (long long)data;
}

static int open_fd(int offset) {
  
  int fd=-1;
  char filename[BUFSIZ];

  if (fd_array[offset].open==0) {
	  sprintf(filename,"/dev/cpu/%d/msr_safe",offset);
      fd = open(filename, O_RDONLY);
	  if (fd<0) {
		  sprintf(filename,"/dev/cpu/%d/msr",offset);
          fd = open(filename, O_RDONLY);
	  }
	  if (fd>=0) {
		  fd_array[offset].fd=fd;
	      fd_array[offset].open=1;
      } 
  }
  else {
    fd=fd_array[offset].fd;
  }

  return fd;
}

static long long read_rapl_value(int index) {

   int fd;

   fd=open_fd(rapl_native_events[index].fd_offset);
   return read_msr(fd,rapl_native_events[index].msr);

}

static long long convert_rapl_energy(int index, long long value) {

   union {
      long long ll;
      double fp;
   } return_val;

   return_val.ll = value; /* default case: return raw input value */

   if (rapl_native_events[index].type==PACKAGE_ENERGY) {
      return_val.ll = (long long)(((double)value/cpu_energy_divisor)*1e9);
   }

   if (rapl_native_events[index].type==DRAM_ENERGY) {
      return_val.ll = (long long)(((double)value/dram_energy_divisor)*1e9);
   }

   if (rapl_native_events[index].type==PLATFORM_ENERGY) {
      return_val.ll = (long long)(((double)value/cpu_energy_divisor)*1e9);
   }

   if (rapl_native_events[index].type==PACKAGE_THERMAL) {
      return_val.fp = (double)
                      ((value>>THERMAL_SHIFT)&POWER_INFO_UNIT_MASK) /
                       (double)power_divisor;
   }

   if (rapl_native_events[index].type==PACKAGE_MINIMUM) {
       return_val.fp = (double)
                       ((value>>MINIMUM_POWER_SHIFT)&POWER_INFO_UNIT_MASK)/
                        (double)power_divisor;
   }

   if (rapl_native_events[index].type==PACKAGE_MAXIMUM) {
      return_val.fp = (double)
                      ((value>>MAXIMUM_POWER_SHIFT)&POWER_INFO_UNIT_MASK)/
                       (double)power_divisor;
   }

   if (rapl_native_events[index].type==PACKAGE_TIME_WINDOW) {
      return_val.fp =  (double)
                    ((value>>MAXIMUM_TIME_WINDOW_SHIFT)&POWER_INFO_UNIT_MASK)/
                     (double)time_divisor;
   }

   if (rapl_native_events[index].type==PACKAGE_THERMAL_CNT) {
      return_val.ll = ((value>>THERMAL_SHIFT)&POWER_INFO_UNIT_MASK);
   }

   if (rapl_native_events[index].type==PACKAGE_MINIMUM_CNT) {
       return_val.ll = ((value>>MINIMUM_POWER_SHIFT)&POWER_INFO_UNIT_MASK);
   }

   if (rapl_native_events[index].type==PACKAGE_MAXIMUM_CNT) {
      return_val.ll = ((value>>MAXIMUM_POWER_SHIFT)&POWER_INFO_UNIT_MASK);
   }

   if (rapl_native_events[index].type==PACKAGE_TIME_WINDOW_CNT) {
      return_val.ll = ((value>>MAXIMUM_TIME_WINDOW_SHIFT)&POWER_INFO_UNIT_MASK);
   }

   return return_val.ll;
}

static int
get_kernel_nr_cpus(void)
{
  FILE *fff;
  int num_read, nr_cpus = 1;
  fff=fopen("/sys/devices/system/cpu/kernel_max","r");
  if (fff==NULL) return nr_cpus;
  num_read=fscanf(fff,"%d",&nr_cpus);
  fclose(fff);
  if (num_read==1) {
    nr_cpus++;
  } else {
    nr_cpus = 1;
  }
  return nr_cpus;
}

/************************* PAPI Functions **********************************/


/*
 * This is called whenever a thread is initialized
 */
static int
_rapl_init_thread( hwd_context_t *ctx )
{
  ( void ) ctx;

  return PAPI_OK;
}



/*
 * Called when PAPI process is initialized (i.e. PAPI_library_init)
 */
static int
_rapl_init_component( int cidx )
{
    int retval = PAPI_OK;
     int i,j,k,fd;
     FILE *fff;
     char filename[BUFSIZ];
     long unsigned int strErr;
     char *strCpy;

	int package_avail, dram_avail, pp0_avail, pp1_avail, psys_avail;
	int different_units;

     long long result;
     int package;

     const PAPI_hw_info_t *hw_info;

     int nr_cpus = get_kernel_nr_cpus();
     int packages[nr_cpus];
     int cpu_to_use[nr_cpus];

	unsigned int msr_pkg_energy_status,msr_pp0_energy_status;


     /* Fill with sentinel values */
     for (i=0; i<nr_cpus; ++i) {
       packages[i] = -1;
       cpu_to_use[i] = -1;
     }


	/* check if supported processor */
	hw_info=&(_papi_hwi_system_info.hw_info);

	/* Ugh can't use PAPI_get_hardware_info() if
		PAPI library not done initializing yet */

	switch(hw_info->vendor) {
		case PAPI_VENDOR_INTEL:
		case PAPI_VENDOR_AMD:
			break;
		default:
			strCpy=strncpy(_rapl_vector.cmp_info.disabled_reason,
			"Not a supported processor",PAPI_MAX_STR_LEN);
			_rapl_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
         if (strCpy == NULL) HANDLE_STRING_ERROR;
         retval = PAPI_ENOSUPP;
         goto fn_fail;
	}


	/* Make sure it is a family 6 Intel Chip */

	if (hw_info->vendor==PAPI_VENDOR_INTEL) {

		msr_rapl_power_unit=MSR_INTEL_RAPL_POWER_UNIT;
		msr_pkg_energy_status=MSR_INTEL_PKG_ENERGY_STATUS;
		msr_pp0_energy_status=MSR_INTEL_PP0_ENERGY_STATUS;

		if (hw_info->cpuid_family!=6) {
			/* Not a family 6 machine */
			strCpy=strncpy(_rapl_vector.cmp_info.disabled_reason,
				"CPU family not supported",PAPI_MAX_STR_LEN);
			_rapl_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
         if (strCpy == NULL) HANDLE_STRING_ERROR;
         retval = PAPI_ENOIMPL;
         goto fn_fail;
		}

		/* Detect RAPL support */
		switch(hw_info->cpuid_model) {

		/* Desktop / Laptops */

		case 42:	/* SandyBridge */
		case 58:	/* IvyBridge */
		case 183:	/* RaptorLake-S/HX */
		case 186:	/* RaptorLake */
		case 191:	/* RaptorLake */
			package_avail=1;
			pp0_avail=1;
			pp1_avail=1;
			dram_avail=0;
			psys_avail=0;
			different_units=0;
			break;

		case 60:	/* Haswell */
		case 69:	/* Haswell ULT */
		case 70:	/* Haswell GT3E */
		case 92:	/* Atom Goldmont */
		case 122:	/* Atom Gemini Lake */
		case 95:	/* Atom Denverton */
			package_avail=1;
			pp0_avail=1;
			pp1_avail=1;
			dram_avail=1;
			psys_avail=0;
			different_units=0;
			break;

		case 61:	/* Broadwell */
		case 71:	/* Broadwell-H (GT3E) */
		case 86:	/* Broadwell XEON_D */
			package_avail=1;
			pp0_avail=1;
			pp1_avail=0;
			dram_avail=1;
			psys_avail=0;
			different_units=0;
			break;

		case 78:	/* Skylake Mobile */
		case 94:	/* Skylake Desktop (H/S) */
		case 142:	/* Kabylake Mobile */
		case 158:	/* Kabylake Desktop */
		case 165:	/* Comet Lake S/H */
			package_avail=1;
			pp0_avail=1;
			pp1_avail=0;
			dram_avail=1;
			psys_avail=1;
			different_units=0;
			break;

		/* Server Class Machines */

		case 45:	/* SandyBridge-EP */
		case 62:	/* IvyBridge-EP */
			package_avail=1;
			pp0_avail=1;
			pp1_avail=0;
			dram_avail=1;
			psys_avail=0;
			different_units=0;
			break;

		case 63:	/* Haswell-EP */
		case 79:	/* Broadwell-EP */
		case 85:	/* Skylake-X */
		case 106:	/* Icelake-SP */
			package_avail=1;
			pp0_avail=1;
			pp1_avail=0;
			dram_avail=1;
			psys_avail=0;
			different_units=1;
			break;
			
		case 143:      /* Sapphire Rapids-SP */
		case 207:      /* Emerald Rapids */
		        package_avail=1;
			pp0_avail=0;
			pp1_avail=0;
			dram_avail=1;
			psys_avail=0;
			different_units=0;
			break;

		case 87:	/* Knights Landing (KNL) */
		case 133:	/* Knights Mill (KNM) */
			package_avail=1;
			pp0_avail=0;
			pp1_avail=0;
			dram_avail=1;
			psys_avail=0;
			different_units=1;
			break;

		default:	/* not a supported model */
			strCpy=strncpy(_rapl_vector.cmp_info.disabled_reason,
				"CPU model not supported",
				PAPI_MAX_STR_LEN);
			_rapl_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
         if (strCpy == NULL) HANDLE_STRING_ERROR;
         retval = PAPI_ENOIMPL;
         goto fn_fail;
		}
	}

	if (hw_info->vendor==PAPI_VENDOR_AMD) {

		msr_rapl_power_unit=MSR_AMD_RAPL_POWER_UNIT;
		msr_pkg_energy_status=MSR_AMD_PKG_ENERGY_STATUS;
		msr_pp0_energy_status=MSR_AMD_PP0_ENERGY_STATUS;

		if ((hw_info->cpuid_family!=0x17) && (hw_info->cpuid_family!=0x19)) {
			/* Not a family 17h or 19h machine */
			strCpy=strncpy(_rapl_vector.cmp_info.disabled_reason,
				"CPU family not supported",PAPI_MAX_STR_LEN);
			_rapl_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
         if (strCpy == NULL) HANDLE_STRING_ERROR;
         retval = PAPI_ENOIMPL;
         goto fn_fail;
		}

		package_avail=1;
		pp0_avail=1;		/* Doesn't work on EPYC? */
		pp1_avail=0;
		dram_avail=0;
		psys_avail=0;
		different_units=0;
	}


     /* Detect how many packages */
     // Some code below may be flagged by Coverity due to uninitialized array
     // entries of cpu_to_use[]. This is not a bug; the 'filename' listed below
     // will have 'cpu0', 'cpu1', sequentially on up to the maximum.  Coverity
     // cannot know that, so its code analysis allows the possibility that the
     // cpu_to_use[] array is only partially filled in. [Tony C. 11-27-19].

     j=0;
     while(1) {
       int num_read;

       strErr=snprintf(filename, BUFSIZ, 
	       "/sys/devices/system/cpu/cpu%d/topology/physical_package_id",j);
       filename[BUFSIZ-1]=0;
       if (strErr > BUFSIZ) HANDLE_STRING_ERROR;
       fff=fopen(filename,"r");
       if (fff==NULL) break;
       num_read=fscanf(fff,"%d",&package);
       fclose(fff);
       if (num_read!=1) {
    		 strCpy=strcpy(_rapl_vector.cmp_info.disabled_reason, "Error reading file: ");
          if (strCpy == NULL) HANDLE_STRING_ERROR;
    		 strCpy=strncat(_rapl_vector.cmp_info.disabled_reason, filename, PAPI_MAX_STR_LEN - strlen(_rapl_vector.cmp_info.disabled_reason) - 1);
    		 _rapl_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1] = '\0';
          if (strCpy == NULL) HANDLE_STRING_ERROR;
          retval = PAPI_ESYS;
          goto fn_fail;
       }

       /* Check if a new package */
       if ((package >= 0) && (package < nr_cpus)) {
         if (packages[package] == -1) {
           SUBDBG("Found package %d out of total %d\n",package,num_packages);
	   packages[package]=package;
	   cpu_to_use[package]=j;
	   num_packages++;
         }
       } else {
	 SUBDBG("Package outside of allowed range\n");
	 strCpy=strncpy(_rapl_vector.cmp_info.disabled_reason,
		"Package outside of allowed range",PAPI_MAX_STR_LEN);
	 _rapl_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
    if (strCpy == NULL) HANDLE_STRING_ERROR;
    retval = PAPI_ESYS;
    goto fn_fail;
       }

       j++;
     }
     num_cpus=j;

     if (num_packages==0) {
        SUBDBG("Can't access /dev/cpu/*/<msr_safe | msr>\n");
    strCpy=strncpy(_rapl_vector.cmp_info.disabled_reason,
		"Can't access /dev/cpu/*/<msr_safe | msr>",PAPI_MAX_STR_LEN);
    _rapl_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
    if (strCpy == NULL) HANDLE_STRING_ERROR;
    retval = PAPI_ESYS;
    goto fn_fail;
     }

     SUBDBG("Found %d packages with %d cpus\n",num_packages,num_cpus);

     /* Init fd_array */

     fd_array=papi_calloc(num_cpus, sizeof(struct fd_array_t));
     if (fd_array==NULL) {
         retval = PAPI_ENOMEM;
         goto fn_fail;
     }

     fd=open_fd(cpu_to_use[0]);
     if (fd<0) {
        strErr=snprintf(_rapl_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
        "Can't open fd for cpu0: %s",strerror(errno));
        _rapl_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        retval = PAPI_ESYS;
        goto fn_fail;
     }

     /* Verify needed MSR is readable. In a guest VM it may not be readable*/
     if (pread(fd, &result, sizeof result, msr_rapl_power_unit) != sizeof result ) {
        strCpy=strncpy(_rapl_vector.cmp_info.disabled_reason,
               "Unable to access RAPL registers",PAPI_MAX_STR_LEN);
        _rapl_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strCpy == NULL) HANDLE_STRING_ERROR;
        retval = PAPI_ESYS;
        goto fn_fail;
     }

     /* Calculate the units used */
     result=read_msr(fd,msr_rapl_power_unit);

     /* units are 0.5^UNIT_VALUE */
     /* which is the same as 1/(2^UNIT_VALUE) */

     power_divisor=1<<((result>>POWER_UNIT_OFFSET)&POWER_UNIT_MASK);
     cpu_energy_divisor=1<<((result>>ENERGY_UNIT_OFFSET)&ENERGY_UNIT_MASK);
     time_divisor=1<<((result>>TIME_UNIT_OFFSET)&TIME_UNIT_MASK);

	/* Note! On Haswell-EP DRAM energy is fixed at 15.3uJ	*/
	/* see https://lkml.org/lkml/2015/3/20/582		*/
	/* Knights Landing is the same */
	/* so is Broadwell-EP */
	if ( different_units ) {
		dram_energy_divisor=1<<16;
	}
	else {
		dram_energy_divisor=cpu_energy_divisor;
	}

     SUBDBG("Power units = %.3fW\n",1.0/power_divisor);
     SUBDBG("CPU Energy units = %.8fJ\n",1.0/cpu_energy_divisor);
     SUBDBG("DRAM Energy units = %.8fJ\n",1.0/dram_energy_divisor);
     SUBDBG("Time units = %.8fs\n",1.0/time_divisor);

     /* Allocate space for events */
     /* Include room for both counts and scaled values */

     num_events= ((package_avail*num_packages) +
                 (pp0_avail*num_packages) +
                 (pp1_avail*num_packages) +
                 (dram_avail*num_packages) +
		(psys_avail*num_packages)) * 2;

	if (hw_info->vendor==PAPI_VENDOR_INTEL) {
		num_events+=(4*num_packages) * 2;
	}

     rapl_native_events = (_rapl_native_event_entry_t*)
          papi_calloc(num_events, sizeof(_rapl_native_event_entry_t));
     if (rapl_native_events == NULL) {
        strErr=snprintf(_rapl_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
        "%s:%i rapl_native_events papi_calloc for %lu bytes failed.", __FILE__, __LINE__, num_events*sizeof(_rapl_native_event_entry_t));
        _rapl_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        retval = PAPI_ENOMEM;
        goto fn_fail;
     }

     i = 0;
     k = num_events/2;

     /* Create events for package power info */

	if (hw_info->vendor==PAPI_VENDOR_INTEL)
     for(j=0;j<num_packages;j++) {
        strErr=snprintf(rapl_native_events[i].name, PAPI_MAX_STR_LEN, 
			"THERMAL_SPEC_CNT:PACKAGE%d",j);
        rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;

        strErr=snprintf(rapl_native_events[i].description, PAPI_MAX_STR_LEN,
		   "Thermal specification in counts; package %d",j);
        rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
		rapl_native_events[i].fd_offset=cpu_to_use[j];
		rapl_native_events[i].msr=MSR_PKG_POWER_INFO;
		rapl_native_events[i].resources.selector = i + 1;
		rapl_native_events[i].type=PACKAGE_THERMAL_CNT;
		rapl_native_events[i].return_type=PAPI_DATATYPE_UINT64;

        strErr=snprintf(rapl_native_events[k].name, PAPI_MAX_STR_LEN,
			"THERMAL_SPEC:PACKAGE%d",j);
        rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;

        strCpy=strncpy(rapl_native_events[k].units,"W",PAPI_MIN_STR_LEN);
        rapl_native_events[k].units[PAPI_MIN_STR_LEN-1]=0;
        if (strCpy == NULL) HANDLE_STRING_ERROR;

        strErr=snprintf(rapl_native_events[k].description, PAPI_MAX_STR_LEN,
		   "Thermal specification for package %d",j);
        rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
		rapl_native_events[k].fd_offset=cpu_to_use[j];
		rapl_native_events[k].msr=MSR_PKG_POWER_INFO;
		rapl_native_events[k].resources.selector = k + 1;
		rapl_native_events[k].type=PACKAGE_THERMAL;
		rapl_native_events[k].return_type=PAPI_DATATYPE_FP64;

		i++;
		k++;
     }

	if (hw_info->vendor==PAPI_VENDOR_INTEL)
     for(j=0;j<num_packages;j++) {
        strErr=snprintf(rapl_native_events[i].name, PAPI_MAX_STR_LEN,
			"MINIMUM_POWER_CNT:PACKAGE%d",j);
        rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        strErr=snprintf(rapl_native_events[i].description, PAPI_MAX_STR_LEN,
		   "Minimum power in counts; package %d",j);
        rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
		rapl_native_events[i].fd_offset=cpu_to_use[j];
		rapl_native_events[i].msr=MSR_PKG_POWER_INFO;
		rapl_native_events[i].resources.selector = i + 1;
		rapl_native_events[i].type=PACKAGE_MINIMUM_CNT;
		rapl_native_events[i].return_type=PAPI_DATATYPE_UINT64;

        strErr=snprintf(rapl_native_events[k].name, PAPI_MAX_STR_LEN,
			"MINIMUM_POWER:PACKAGE%d",j);
        rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        strCpy=strncpy(rapl_native_events[k].units,"W",PAPI_MIN_STR_LEN);
        rapl_native_events[k].units[PAPI_MIN_STR_LEN-1]=0;
        if (strCpy == NULL) HANDLE_STRING_ERROR;
        strErr=snprintf(rapl_native_events[k].description, PAPI_MAX_STR_LEN,
		   "Minimum power for package %d",j);
        rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
		rapl_native_events[k].fd_offset=cpu_to_use[j];
		rapl_native_events[k].msr=MSR_PKG_POWER_INFO;
		rapl_native_events[k].resources.selector = k + 1;
		rapl_native_events[k].type=PACKAGE_MINIMUM;
		rapl_native_events[k].return_type=PAPI_DATATYPE_FP64;

		i++;
		k++;
     }

	if (hw_info->vendor==PAPI_VENDOR_INTEL)
     for(j=0;j<num_packages;j++) {
        strErr=snprintf(rapl_native_events[i].name, PAPI_MAX_STR_LEN,
			"MAXIMUM_POWER_CNT:PACKAGE%d",j);
        rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        strErr=snprintf(rapl_native_events[i].description, PAPI_MAX_STR_LEN,
		   "Maximum power in counts; package %d",j);
        rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
		rapl_native_events[i].fd_offset=cpu_to_use[j];
		rapl_native_events[i].msr=MSR_PKG_POWER_INFO;
		rapl_native_events[i].resources.selector = i + 1;
		rapl_native_events[i].type=PACKAGE_MAXIMUM_CNT;
		rapl_native_events[i].return_type=PAPI_DATATYPE_UINT64;

        strErr=snprintf(rapl_native_events[k].name, PAPI_MAX_STR_LEN,
			"MAXIMUM_POWER:PACKAGE%d",j);
        rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        strCpy=strncpy(rapl_native_events[k].units,"W",PAPI_MIN_STR_LEN);
        rapl_native_events[k].units[PAPI_MIN_STR_LEN-1]=0;
        if (strCpy == NULL) HANDLE_STRING_ERROR;
        strErr=snprintf(rapl_native_events[k].description, PAPI_MAX_STR_LEN,
		   "Maximum power for package %d",j);
        rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
        if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
		rapl_native_events[k].fd_offset=cpu_to_use[j];
		rapl_native_events[k].msr=MSR_PKG_POWER_INFO;
		rapl_native_events[k].resources.selector = k + 1;
		rapl_native_events[k].type=PACKAGE_MAXIMUM;
		rapl_native_events[k].return_type=PAPI_DATATYPE_FP64;

		i++;
		k++;
     }

	if (hw_info->vendor==PAPI_VENDOR_INTEL)
     for(j=0;j<num_packages;j++) {
         strErr=snprintf(rapl_native_events[i].name, PAPI_MAX_STR_LEN,
			"MAXIMUM_TIME_WINDOW_CNT:PACKAGE%d",j);
         rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
         if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
         strErr=snprintf(rapl_native_events[i].description, PAPI_MAX_STR_LEN,
		   "Maximum time window in counts; package %d",j);
         rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
         if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
		rapl_native_events[i].fd_offset=cpu_to_use[j];
		rapl_native_events[i].msr=MSR_PKG_POWER_INFO;
		rapl_native_events[i].resources.selector = i + 1;
		rapl_native_events[i].type=PACKAGE_TIME_WINDOW_CNT;
		rapl_native_events[i].return_type=PAPI_DATATYPE_UINT64;

         strErr=snprintf(rapl_native_events[k].name, PAPI_MAX_STR_LEN,
			"MAXIMUM_TIME_WINDOW:PACKAGE%d",j);
         rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
         if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
         strCpy=strncpy(rapl_native_events[k].units,"s",PAPI_MIN_STR_LEN);
         rapl_native_events[k].units[PAPI_MIN_STR_LEN-1]=0;
         if (strCpy == NULL) HANDLE_STRING_ERROR;
         strErr=snprintf(rapl_native_events[k].description, PAPI_MAX_STR_LEN,
		   "Maximum time window for package %d",j);
         rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
         if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
		rapl_native_events[k].fd_offset=cpu_to_use[j];
		rapl_native_events[k].msr=MSR_PKG_POWER_INFO;
		rapl_native_events[k].resources.selector = k + 1;
		rapl_native_events[k].type=PACKAGE_TIME_WINDOW;
		rapl_native_events[k].return_type=PAPI_DATATYPE_FP64;

		i++;
		k++;
     }

     /* Create Events for energy measurements */

     if (package_avail) {
        for(j=0;j<num_packages;j++) {
            strErr=snprintf(rapl_native_events[i].name, PAPI_MAX_STR_LEN,
		   		"PACKAGE_ENERGY_CNT:PACKAGE%d",j);
            rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            strErr=snprintf(rapl_native_events[i].description, PAPI_MAX_STR_LEN,
		   		"Energy used in counts by chip package %d",j);
            rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
	   		rapl_native_events[i].fd_offset=cpu_to_use[j];
	   		rapl_native_events[i].msr=msr_pkg_energy_status;
	   		rapl_native_events[i].resources.selector = i + 1;
	   		rapl_native_events[i].type=PACKAGE_ENERGY_CNT;
	   		rapl_native_events[i].return_type=PAPI_DATATYPE_UINT64;

            strErr=snprintf(rapl_native_events[k].name, PAPI_MAX_STR_LEN,
		   		"PACKAGE_ENERGY:PACKAGE%d",j);
            rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            strCpy=strncpy(rapl_native_events[k].units,"nJ",PAPI_MIN_STR_LEN);
            rapl_native_events[k].units[PAPI_MIN_STR_LEN-1]=0;
            if (strCpy == NULL) HANDLE_STRING_ERROR;
            strErr=snprintf(rapl_native_events[k].description, PAPI_MAX_STR_LEN,
		   		"Energy used by chip package %d",j);
            rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
	   		rapl_native_events[k].fd_offset=cpu_to_use[j];
	   		rapl_native_events[k].msr=msr_pkg_energy_status;
	   		rapl_native_events[k].resources.selector = k + 1;
	   		rapl_native_events[k].type=PACKAGE_ENERGY;
	   		rapl_native_events[k].return_type=PAPI_DATATYPE_UINT64;

	   		i++;
			k++;
		}
     }

     if (pp1_avail) {
        for(j=0;j<num_packages;j++) {
            strErr=snprintf(rapl_native_events[i].name, PAPI_MAX_STR_LEN,
		   		"PP1_ENERGY_CNT:PACKAGE%d",j);
            rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            strErr=snprintf(rapl_native_events[i].description, PAPI_MAX_STR_LEN,
		   	"Energy used in counts by Power Plane 1 (Often GPU) on package %d",j);
            rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
           	rapl_native_events[i].fd_offset=cpu_to_use[j];
	   		rapl_native_events[i].msr=MSR_PP1_ENERGY_STATUS;
	   		rapl_native_events[i].resources.selector = i + 1;
	   		rapl_native_events[i].type=PACKAGE_ENERGY_CNT;
	   		rapl_native_events[i].return_type=PAPI_DATATYPE_UINT64;

            strErr=snprintf(rapl_native_events[k].name, PAPI_MAX_STR_LEN,
		   		"PP1_ENERGY:PACKAGE%d",j);
            rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            strCpy=strncpy(rapl_native_events[k].units,"nJ",PAPI_MIN_STR_LEN);
            rapl_native_events[k].units[PAPI_MIN_STR_LEN-1]=0;
            if (strCpy == NULL) HANDLE_STRING_ERROR;
            strErr=snprintf(rapl_native_events[k].description, PAPI_MAX_STR_LEN,
		   		"Energy used by Power Plane 1 (Often GPU) on package %d",j);
            rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
           	rapl_native_events[k].fd_offset=cpu_to_use[j];
	   		rapl_native_events[k].msr=MSR_PP1_ENERGY_STATUS;
	   		rapl_native_events[k].resources.selector = k + 1;
	   		rapl_native_events[k].type=PACKAGE_ENERGY;
	   		rapl_native_events[k].return_type=PAPI_DATATYPE_UINT64;

	   		i++;
			k++;
		}
     }

     if (dram_avail) {
        for(j=0;j<num_packages;j++) {
            strErr=snprintf(rapl_native_events[i].name, PAPI_MAX_STR_LEN,
		   		"DRAM_ENERGY_CNT:PACKAGE%d",j);
            rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            strErr=snprintf(rapl_native_events[i].description, PAPI_MAX_STR_LEN,
		   		"Energy used in counts by DRAM on package %d",j);
            rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
	   		rapl_native_events[i].fd_offset=cpu_to_use[j];
	   		rapl_native_events[i].msr=MSR_DRAM_ENERGY_STATUS;
	   		rapl_native_events[i].resources.selector = i + 1;
	   		rapl_native_events[i].type=PACKAGE_ENERGY_CNT;
	   		rapl_native_events[i].return_type=PAPI_DATATYPE_UINT64;

            strErr=snprintf(rapl_native_events[k].name,PAPI_MAX_STR_LEN,
		   		"DRAM_ENERGY:PACKAGE%d",j);
            rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            strCpy=strncpy(rapl_native_events[k].units,"nJ",PAPI_MIN_STR_LEN);
            rapl_native_events[k].units[PAPI_MIN_STR_LEN-1]=0;
            if (strCpy == NULL) HANDLE_STRING_ERROR;
            strErr=snprintf(rapl_native_events[k].description, PAPI_MAX_STR_LEN,
		   		"Energy used by DRAM on package %d",j);
            rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
	   		rapl_native_events[k].fd_offset=cpu_to_use[j];
	   		rapl_native_events[k].msr=MSR_DRAM_ENERGY_STATUS;
	   		rapl_native_events[k].resources.selector = k + 1;
	   		rapl_native_events[k].type=DRAM_ENERGY;
	   		rapl_native_events[k].return_type=PAPI_DATATYPE_UINT64;

	   		i++;
			k++;
		}
     }

     if (psys_avail) {
        for(j=0;j<num_packages;j++) {

         strErr=snprintf(rapl_native_events[i].name, PAPI_MAX_STR_LEN,
		   		"PSYS_ENERGY_CNT:PACKAGE%d",j);
         rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
         if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            strErr=snprintf(rapl_native_events[i].description, PAPI_MAX_STR_LEN,
		   		"Energy used in counts by SoC on package %d",j);
            rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
	   		rapl_native_events[i].fd_offset=cpu_to_use[j];
	   		rapl_native_events[i].msr=MSR_PLATFORM_ENERGY_STATUS;
	   		rapl_native_events[i].resources.selector = i + 1;
	   		rapl_native_events[i].type=PACKAGE_ENERGY_CNT;
	   		rapl_native_events[i].return_type=PAPI_DATATYPE_UINT64;

            strErr=snprintf(rapl_native_events[k].name, PAPI_MAX_STR_LEN,
		   		"PSYS_ENERGY:PACKAGE%d",j);
            rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            strCpy=strncpy(rapl_native_events[k].units,"nJ",PAPI_MIN_STR_LEN);
            rapl_native_events[k].units[PAPI_MIN_STR_LEN-1]=0;
            if (strCpy == NULL) HANDLE_STRING_ERROR;
            strErr=snprintf(rapl_native_events[k].description, PAPI_MAX_STR_LEN,
		   		"Energy used by SoC on package %d",j);
            rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
	   		rapl_native_events[k].fd_offset=cpu_to_use[j];
	   		rapl_native_events[k].msr=MSR_PLATFORM_ENERGY_STATUS;
	   		rapl_native_events[k].resources.selector = k + 1;
	   		rapl_native_events[k].type=PLATFORM_ENERGY;
	   		rapl_native_events[k].return_type=PAPI_DATATYPE_UINT64;

			i++;
			k++;
		}
     }

     if (pp0_avail) {
        for(j=0;j<num_packages;j++) {
            strErr=snprintf(rapl_native_events[i].name, PAPI_MAX_STR_LEN,
		   		"PP0_ENERGY_CNT:PACKAGE%d",j);
            rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            strErr=snprintf(rapl_native_events[i].description, PAPI_MAX_STR_LEN,
		   		"Energy used in counts by all cores in package %d",j);
            rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
	   		rapl_native_events[i].fd_offset=cpu_to_use[j];
	   		rapl_native_events[i].msr=msr_pp0_energy_status;
	   		rapl_native_events[i].resources.selector = i + 1;
	   		rapl_native_events[i].type=PACKAGE_ENERGY_CNT;
	   		rapl_native_events[i].return_type=PAPI_DATATYPE_UINT64;

            strErr=snprintf(rapl_native_events[k].name, PAPI_MAX_STR_LEN,
		   		"PP0_ENERGY:PACKAGE%d",j);
            rapl_native_events[i].name[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
            strCpy=strncpy(rapl_native_events[k].units,"nJ",PAPI_MIN_STR_LEN);
            rapl_native_events[k].units[PAPI_MIN_STR_LEN-1]=0;
            if (strCpy == NULL) HANDLE_STRING_ERROR;
            strErr=snprintf(rapl_native_events[k].description, PAPI_MAX_STR_LEN,
		   		"Energy used by all cores in package %d",j);
            rapl_native_events[i].description[PAPI_MAX_STR_LEN-1]=0;
            if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
	   		rapl_native_events[k].fd_offset=cpu_to_use[j];
	   		rapl_native_events[k].msr=msr_pp0_energy_status;
	   		rapl_native_events[k].resources.selector = k + 1;
	   		rapl_native_events[k].type=PACKAGE_ENERGY;
	   		rapl_native_events[k].return_type=PAPI_DATATYPE_UINT64;

	   		i++;
			k++;
		}
     }

     /* Export the total number of events available */
     _rapl_vector.cmp_info.num_native_events = num_events;

     _rapl_vector.cmp_info.num_cntrs = num_events;
     _rapl_vector.cmp_info.num_mpx_cntrs = num_events;


     /* Export the component id */
     _rapl_vector.cmp_info.CmpIdx = cidx;

  fn_exit:
    _papi_hwd[cidx]->cmp_info.disabled = retval;
     return retval;
  fn_fail:
     goto fn_exit;
}


/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
static int
_rapl_init_control_state( hwd_control_state_t *ctl)
{

  _rapl_control_state_t* control = (_rapl_control_state_t*) ctl;
  int i;

  for(i=0;i<RAPL_MAX_COUNTERS;i++) {
     control->being_measured[i]=0;
  }

  return PAPI_OK;
}

static int
_rapl_start( hwd_context_t *ctx, hwd_control_state_t *ctl)
{
  _rapl_context_t* context = (_rapl_context_t*) ctx;
  _rapl_control_state_t* control = (_rapl_control_state_t*) ctl;
  long long now = PAPI_get_real_usec();
  int i;

  
  for( i = 0; i < RAPL_MAX_COUNTERS; i++ ) {
     if ((control->being_measured[i]) && (control->need_difference[i])) {
        context->start_value[i]=(read_rapl_value(i) & 0xFFFFFFFF);
        context->accumulated_value[i]=0;
     }
  }

  control->lastupdate = now;

  return PAPI_OK;
}

static int
_rapl_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
   /* read values */
   _rapl_context_t* context = (_rapl_context_t*) ctx;
   _rapl_control_state_t* control = (_rapl_control_state_t*) ctl;
   long long now = PAPI_get_real_usec();
   int i;
   long long temp, newstart;

   for ( i = 0; i < RAPL_MAX_COUNTERS; i++ ) {
      if (control->being_measured[i]) {
         temp = read_rapl_value(i);
         if (control->need_difference[i]) {
            temp &= 0xFFFFFFFF;
            newstart = temp;
            /* test for wrap around */
            if (temp < context->start_value[i] ) {
               SUBDBG("Wraparound!\nstart:\t%#016x\ttemp:\t%#016x",
                  (unsigned)context->start_value[i], (unsigned)temp);
               temp += (0x100000000 - context->start_value[i]);
               SUBDBG("\tresult:\t%#016x\n", (unsigned)temp);
            } else {
               temp -= context->start_value[i];
            }
            // reset the start value, add to accum, set temp for convert call.
            context->start_value[i]=newstart;
            context->accumulated_value[i] += temp;
            temp = context->accumulated_value[i];
         }
         control->count[i] = convert_rapl_energy( i, temp );
      }
    }
    control->lastupdate = now;
    return PAPI_OK;
}

/* Shutdown a thread */
static int
_rapl_shutdown_thread( hwd_context_t *ctx )
{
  ( void ) ctx;
  return PAPI_OK;
}

int
_rapl_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
	    long long **events, int flags)
{
    (void) flags;

    _rapl_stop( ctx, ctl );

    /* Pass back a pointer to our results */
    *events = ((_rapl_control_state_t*) ctl)->count;

    return PAPI_OK;
}


/*
 * Clean up what was setup in  rapl_init_component().
 */
static int
_rapl_shutdown_component( void )
{
    int i;

    if (rapl_native_events) papi_free(rapl_native_events);
    if (fd_array) {
       for(i=0;i<num_cpus;i++) {
	  if (fd_array[i].open) close(fd_array[i].fd);
       }
       papi_free(fd_array);
    }

    return PAPI_OK;
}


/* This function sets various options in the component
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
static int
_rapl_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
    ( void ) ctx;
    ( void ) code;
    ( void ) option;

    return PAPI_OK;
}


static int
_rapl_update_control_state( hwd_control_state_t *ctl,
			    NativeInfo_t *native, int count,
			    hwd_context_t *ctx )
{
  int i, index;
    ( void ) ctx;

    _rapl_control_state_t* control = (_rapl_control_state_t*) ctl;

    /* Ugh, what is this native[] stuff all about ?*/
    /* Mostly remap stuff in papi_internal */

    for(i=0;i<RAPL_MAX_COUNTERS;i++) {
       control->being_measured[i]=0;
    }

    for( i = 0; i < count; i++ ) {
       index=native[i].ni_event&PAPI_NATIVE_AND_MASK;
       native[i].ni_position=rapl_native_events[index].resources.selector - 1;
       control->being_measured[index]=1;

       /* Only need to subtract if it's a PACKAGE_ENERGY or ENERGY_CNT type */
       control->need_difference[index]=
	 	(rapl_native_events[index].type==PACKAGE_ENERGY ||
		rapl_native_events[index].type==DRAM_ENERGY ||
		rapl_native_events[index].type==PLATFORM_ENERGY ||
	 	rapl_native_events[index].type==PACKAGE_ENERGY_CNT);
    }

    return PAPI_OK;
}


/*
 * This function has to set the bits needed to count different domains
 * In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 * By default return PAPI_EINVAL if none of those are specified
 * and PAPI_OK with success
 * PAPI_DOM_USER is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL   is all of the domains
 */
static int
_rapl_set_domain( hwd_control_state_t *ctl, int domain )
{
    ( void ) ctl;

    /* In theory we only support system-wide mode */
    /* How to best handle that? */
    if ( PAPI_DOM_ALL != domain )
	return PAPI_EINVAL;

    return PAPI_OK;
}


static int
_rapl_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    ( void ) ctx;
    ( void ) ctl;

    return PAPI_OK;
}


/*
 * Native Event functions
 */
static int
_rapl_ntv_enum_events( unsigned int *EventCode, int modifier )
{

     int index;

     switch ( modifier ) {

	case PAPI_ENUM_FIRST:

	   if (num_events==0) {
	      return PAPI_ENOEVNT;
	   }
	   *EventCode = 0;

	   return PAPI_OK;


	case PAPI_ENUM_EVENTS:

	   index = *EventCode & PAPI_NATIVE_AND_MASK;

	   if ( index < num_events - 1 ) {
	      *EventCode = *EventCode + 1;
	      return PAPI_OK;
	   } else {
	      return PAPI_ENOEVNT;
	   }
	   break;

	default:
		return PAPI_EINVAL;
	}

	return PAPI_EINVAL;
}

/*
 *
 */
static int
_rapl_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{

     int index = EventCode & PAPI_NATIVE_AND_MASK;

     if ( index >= 0 && index < num_events ) {
	strncpy( name, rapl_native_events[index].name, len );
	return PAPI_OK;
     }

     return PAPI_ENOEVNT;
}

/*
 *
 */
static int
_rapl_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
     int index = EventCode;

     if ( index >= 0 && index < num_events ) {
	strncpy( name, rapl_native_events[index].description, len );
	return PAPI_OK;
     }
     return PAPI_ENOEVNT;
}

static int
_rapl_ntv_code_to_info(unsigned int EventCode, PAPI_event_info_t *info) 
{

  int index = EventCode;

  if ( ( index < 0) || (index >= num_events )) return PAPI_ENOEVNT;

  strncpy( info->symbol, rapl_native_events[index].name, sizeof(info->symbol)-1);
  info->symbol[sizeof(info->symbol)-1] = '\0';

  strncpy( info->long_descr, rapl_native_events[index].description, sizeof(info->long_descr)-1);
  info->long_descr[sizeof(info->long_descr)-1] = '\0';

  strncpy( info->units, rapl_native_events[index].units, sizeof(info->units)-1);
  info->units[sizeof(info->units)-1] = '\0';

  info->data_type = rapl_native_events[index].return_type;

  return PAPI_OK;
}



papi_vector_t _rapl_vector = {
    .cmp_info = { /* (unspecified values are initialized to 0) */
       .name = "rapl",
       .short_name = "rapl",
       .description = "Linux RAPL energy measurements",
       .version = "5.3.0",
       .default_domain = PAPI_DOM_ALL,
       .default_granularity = PAPI_GRN_SYS,
       .available_granularities = PAPI_GRN_SYS,
       .hardware_intr_sig = PAPI_INT_SIGNAL,
       .available_domains = PAPI_DOM_ALL,
    },

	/* sizes of framework-opaque component-private structures */
    .size = {
	.context = sizeof ( _rapl_context_t ),
	.control_state = sizeof ( _rapl_control_state_t ),
	.reg_value = sizeof ( _rapl_register_t ),
	.reg_alloc = sizeof ( _rapl_reg_alloc_t ),
    },
	/* function pointers in this component */
    .init_thread =          _rapl_init_thread,
    .init_component =       _rapl_init_component,
    .init_control_state =   _rapl_init_control_state,
    .start =                _rapl_start,
    .stop =                 _rapl_stop,
    .read =                 _rapl_read,
    .shutdown_thread =      _rapl_shutdown_thread,
    .shutdown_component =   _rapl_shutdown_component,
    .ctl =                  _rapl_ctl,

    .update_control_state = _rapl_update_control_state,
    .set_domain =           _rapl_set_domain,
    .reset =                _rapl_reset,

    .ntv_enum_events =      _rapl_ntv_enum_events,
    .ntv_code_to_name =     _rapl_ntv_code_to_name,
    .ntv_code_to_descr =    _rapl_ntv_code_to_descr,
    .ntv_code_to_info =     _rapl_ntv_code_to_info,
};
