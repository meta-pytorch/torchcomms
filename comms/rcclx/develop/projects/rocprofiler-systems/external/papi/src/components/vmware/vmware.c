/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    mware.c
 * @author  Matt Johnson
 *          mrj@eecs.utk.edu
 * @author  John Nelson
 *          jnelso37@eecs.utk.edu
 * @author  Vince Weaver
 *          vweaver1@eecs.utk.edu
 *
 * @ingroup papi_components
 *
 * VMware component
 *
 * @brief
 *	This is the VMware component for PAPI-V. It will allow user access to
 *	hardware information available from a VMware virtual machine.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#include <unistd.h>
#include <dlfcn.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#define VMWARE_MAX_COUNTERS 256

#define VMWARE_CPU_LIMIT_MHZ            0
#define VMWARE_CPU_RESERVATION_MHZ      1
#define VMWARE_CPU_SHARES               2
#define VMWARE_CPU_STOLEN_MS            3
#define VMWARE_CPU_USED_MS              4
#define VMWARE_ELAPSED_MS               5

#define VMWARE_MEM_ACTIVE_MB            6
#define VMWARE_MEM_BALLOONED_MB         7
#define VMWARE_MEM_LIMIT_MB             8
#define VMWARE_MEM_MAPPED_MB            9
#define VMWARE_MEM_OVERHEAD_MB          10
#define VMWARE_MEM_RESERVATION_MB       11
#define VMWARE_MEM_SHARED_MB            12
#define VMWARE_MEM_SHARES               13
#define VMWARE_MEM_SWAPPED_MB           14
#define VMWARE_MEM_TARGET_SIZE_MB       15
#define VMWARE_MEM_USED_MB              16

#define VMWARE_HOST_CPU_MHZ             17

/* The following 3 require VMWARE_PSEUDO_PERFORMANCE env_var to be set. */

#define VMWARE_HOST_TSC			18
#define VMWARE_ELAPSED_TIME             19
#define VMWARE_ELAPSED_APPARENT         20

/* Begin PAPI definitions */
papi_vector_t _vmware_vector;


void (*_dl_non_dynamic_init)(void) __attribute__((weak));

/** Structure that stores private information for each event */
struct _vmware_register {
    unsigned int selector;
    /**< Signifies which counter slot is being used */
    /**< Indexed from 1 as 0 has a special meaning  */
};

/** This structure is used to build the table of events */
struct _vmware_native_event_entry {
	char name[PAPI_MAX_STR_LEN];        /**< Name of the counter         */
	char description[PAPI_HUGE_STR_LEN]; /**< Description of counter     */
        char units[PAPI_MIN_STR_LEN];
        int which_counter;
        int report_difference;
};

struct _vmware_reg_alloc {
	struct _vmware_register ra_bits;
};


inline uint64_t rdpmc(int c)
{
  uint32_t low, high;
  __asm__ __volatile__("rdpmc" : "=a" (low), "=d" (high) : "c" (c));
  return (uint64_t)high << 32 | (uint64_t)low;
}



#ifdef VMGUESTLIB
/* Headers required by VMware */
#include "vmGuestLib.h"

/* Functions to dynamically load from the GuestLib library. */
char const * (*GuestLib_GetErrorText)(VMGuestLibError);
VMGuestLibError (*GuestLib_OpenHandle)(VMGuestLibHandle*);
VMGuestLibError (*GuestLib_CloseHandle)(VMGuestLibHandle);
VMGuestLibError (*GuestLib_UpdateInfo)(VMGuestLibHandle handle);
VMGuestLibError (*GuestLib_GetSessionId)(VMGuestLibHandle handle, VMSessionId *id);
VMGuestLibError (*GuestLib_GetCpuReservationMHz)(VMGuestLibHandle handle, uint32 *cpuReservationMHz);
VMGuestLibError (*GuestLib_GetCpuLimitMHz)(VMGuestLibHandle handle, uint32 *cpuLimitMHz);
VMGuestLibError (*GuestLib_GetCpuShares)(VMGuestLibHandle handle, uint32 *cpuShares);
VMGuestLibError (*GuestLib_GetCpuUsedMs)(VMGuestLibHandle handle, uint64 *cpuUsedMs);
VMGuestLibError (*GuestLib_GetHostProcessorSpeed)(VMGuestLibHandle handle, uint32 *mhz);
VMGuestLibError (*GuestLib_GetMemReservationMB)(VMGuestLibHandle handle, uint32 *memReservationMB);
VMGuestLibError (*GuestLib_GetMemLimitMB)(VMGuestLibHandle handle, uint32 *memLimitMB);
VMGuestLibError (*GuestLib_GetMemShares)(VMGuestLibHandle handle, uint32 *memShares);
VMGuestLibError (*GuestLib_GetMemMappedMB)(VMGuestLibHandle handle, uint32 *memMappedMB);
VMGuestLibError (*GuestLib_GetMemActiveMB)(VMGuestLibHandle handle, uint32 *memActiveMB);
VMGuestLibError (*GuestLib_GetMemOverheadMB)(VMGuestLibHandle handle, uint32 *memOverheadMB);
VMGuestLibError (*GuestLib_GetMemBalloonedMB)(VMGuestLibHandle handle, uint32 *memBalloonedMB);
VMGuestLibError (*GuestLib_GetMemSwappedMB)(VMGuestLibHandle handle, uint32 *memSwappedMB);
VMGuestLibError (*GuestLib_GetMemSharedMB)(VMGuestLibHandle handle, uint32 *memSharedMB);
VMGuestLibError (*GuestLib_GetMemSharedSavedMB)(VMGuestLibHandle handle, uint32 *memSharedSavedMB);
VMGuestLibError (*GuestLib_GetMemUsedMB)(VMGuestLibHandle handle, uint32 *memUsedMB);
VMGuestLibError (*GuestLib_GetElapsedMs)(VMGuestLibHandle handle, uint64 *elapsedMs);
VMGuestLibError (*GuestLib_GetResourcePoolPath)(VMGuestLibHandle handle, size_t *bufferSize, char *pathBuffer);
VMGuestLibError (*GuestLib_GetCpuStolenMs)(VMGuestLibHandle handle, uint64 *cpuStolenMs);
VMGuestLibError (*GuestLib_GetMemTargetSizeMB)(VMGuestLibHandle handle, uint64 *memTargetSizeMB);
VMGuestLibError (*GuestLib_GetHostNumCpuCores)(VMGuestLibHandle handle, uint32 *hostNumCpuCores);
VMGuestLibError (*GuestLib_GetHostCpuUsedMs)(VMGuestLibHandle handle, uint64 *hostCpuUsedMs);
VMGuestLibError (*GuestLib_GetHostMemSwappedMB)(VMGuestLibHandle handle, uint64 *hostMemSwappedMB);
VMGuestLibError (*GuestLib_GetHostMemSharedMB)(VMGuestLibHandle handle, uint64 *hostMemSharedMB);
VMGuestLibError (*GuestLib_GetHostMemUsedMB)(VMGuestLibHandle handle, uint64 *hostMemUsedMB);
VMGuestLibError (*GuestLib_GetHostMemPhysMB)(VMGuestLibHandle handle, uint64 *hostMemPhysMB);
VMGuestLibError (*GuestLib_GetHostMemPhysFreeMB)(VMGuestLibHandle handle, uint64 *hostMemPhysFreeMB);
VMGuestLibError (*GuestLib_GetHostMemKernOvhdMB)(VMGuestLibHandle handle, uint64 *hostMemKernOvhdMB);
VMGuestLibError (*GuestLib_GetHostMemMappedMB)(VMGuestLibHandle handle, uint64 *hostMemMappedMB);
VMGuestLibError (*GuestLib_GetHostMemUnmappedMB)(VMGuestLibHandle handle, uint64 *hostMemUnmappedMB);


static void *dlHandle = NULL;


/*
 * Macro to load a single GuestLib function from the shared library.
 */

#define LOAD_ONE_FUNC(funcname)                                 \
do {                                                            \
funcname = dlsym(dlHandle, "VM" #funcname);                     \
if ((dlErrStr = dlerror()) != NULL) {                           \
fprintf(stderr, "Failed to load \'%s\': \'%s\'\n",              \
#funcname, dlErrStr);                                           \
return FALSE;                                                   \
}                                                               \
} while (0)

#endif

/** Holds control flags, usually out-of band configuration of the hardware */
struct _vmware_control_state {
   long long value[VMWARE_MAX_COUNTERS];
   int which_counter[VMWARE_MAX_COUNTERS];
   int num_events;
};

/** Holds per-thread information */
struct _vmware_context {
  long long values[VMWARE_MAX_COUNTERS];
  long long start_values[VMWARE_MAX_COUNTERS];
#ifdef VMGUESTLIB
  VMGuestLibHandle glHandle;
#endif
};






/*
 *-----------------------------------------------------------------------------
 *
 * LoadFunctions --
 *
 *      Load the functions from the shared library.
 *
 * Results:
 *      TRUE on success
 *      FALSE on failure
 *
 * Side effects:
 *      None
 *
 * Credit: VMware
 *-----------------------------------------------------------------------------
 */

static int
LoadFunctions(void)
{

#ifdef VMGUESTLIB
	/*
	 * First, try to load the shared library.
	 */

	/* Attempt to guess if we were statically linked to libc, if so bail */
	if ( _dl_non_dynamic_init != NULL ) {
		strncpy(_vmware_vector.cmp_info.disabled_reason, "The VMware component does not support statically linking of libc.", PAPI_MAX_STR_LEN);
		return PAPI_ENOSUPP;
	}

	char const *dlErrStr;
	char filename[BUFSIZ];

	sprintf(filename,"%s","libvmGuestLib.so");
	dlHandle = dlopen(filename, RTLD_NOW);
	if (!dlHandle) {
	   dlErrStr = dlerror();
	   fprintf(stderr, "dlopen of %s failed: \'%s\'\n", filename, 
		   dlErrStr);

	   sprintf(filename,"%s/lib/lib64/libvmGuestLib.so",VMWARE_INCDIR);
	   dlHandle = dlopen(filename, RTLD_NOW);
	   if (!dlHandle) {
	      dlErrStr = dlerror();
	      fprintf(stderr, "dlopen of %s failed: \'%s\'\n", filename, 
		   dlErrStr);

	      sprintf(filename,"%s/lib/lib32/libvmGuestLib.so",VMWARE_INCDIR);
	      dlHandle = dlopen(filename, RTLD_NOW);
	      if (!dlHandle) {
	         dlErrStr = dlerror();
	         fprintf(stderr, "dlopen of %s failed: \'%s\'\n", filename, 
		      dlErrStr);
		 return PAPI_ECMP;
	      }
	   }
	}

	/* Load all the individual library functions. */
	LOAD_ONE_FUNC(GuestLib_GetErrorText);
	LOAD_ONE_FUNC(GuestLib_OpenHandle);
	LOAD_ONE_FUNC(GuestLib_CloseHandle);
	LOAD_ONE_FUNC(GuestLib_UpdateInfo);
	LOAD_ONE_FUNC(GuestLib_GetSessionId);
	LOAD_ONE_FUNC(GuestLib_GetCpuReservationMHz);
	LOAD_ONE_FUNC(GuestLib_GetCpuLimitMHz);
	LOAD_ONE_FUNC(GuestLib_GetCpuShares);
	LOAD_ONE_FUNC(GuestLib_GetCpuUsedMs);
	LOAD_ONE_FUNC(GuestLib_GetHostProcessorSpeed);
	LOAD_ONE_FUNC(GuestLib_GetMemReservationMB);
	LOAD_ONE_FUNC(GuestLib_GetMemLimitMB);
	LOAD_ONE_FUNC(GuestLib_GetMemShares);
	LOAD_ONE_FUNC(GuestLib_GetMemMappedMB);
	LOAD_ONE_FUNC(GuestLib_GetMemActiveMB);
	LOAD_ONE_FUNC(GuestLib_GetMemOverheadMB);
	LOAD_ONE_FUNC(GuestLib_GetMemBalloonedMB);
	LOAD_ONE_FUNC(GuestLib_GetMemSwappedMB);
	LOAD_ONE_FUNC(GuestLib_GetMemSharedMB);
	LOAD_ONE_FUNC(GuestLib_GetMemSharedSavedMB);
	LOAD_ONE_FUNC(GuestLib_GetMemUsedMB);
	LOAD_ONE_FUNC(GuestLib_GetElapsedMs);
	LOAD_ONE_FUNC(GuestLib_GetResourcePoolPath);
	LOAD_ONE_FUNC(GuestLib_GetCpuStolenMs);
	LOAD_ONE_FUNC(GuestLib_GetMemTargetSizeMB);
	LOAD_ONE_FUNC(GuestLib_GetHostNumCpuCores);
	LOAD_ONE_FUNC(GuestLib_GetHostCpuUsedMs);
	LOAD_ONE_FUNC(GuestLib_GetHostMemSwappedMB);
	LOAD_ONE_FUNC(GuestLib_GetHostMemSharedMB);
	LOAD_ONE_FUNC(GuestLib_GetHostMemUsedMB);
	LOAD_ONE_FUNC(GuestLib_GetHostMemPhysMB);
	LOAD_ONE_FUNC(GuestLib_GetHostMemPhysFreeMB);
	LOAD_ONE_FUNC(GuestLib_GetHostMemKernOvhdMB);
	LOAD_ONE_FUNC(GuestLib_GetHostMemMappedMB);
	LOAD_ONE_FUNC(GuestLib_GetHostMemUnmappedMB);
#endif
	return PAPI_OK;
}



/** This table contains the native events */
static struct _vmware_native_event_entry *_vmware_native_table;
/** number of events in the table*/
static int num_events = 0;
static int use_pseudo=0;
static int use_guestlib=0;

/************************************************************************/
/* Below is the actual "hardware implementation" of our VMWARE counters */
/************************************************************************/

/** Code that reads event values.
 You might replace this with code that accesses
 hardware or reads values from the operatings system. */
static long long
_vmware_hardware_read( struct _vmware_context *context, int starting)
{

  int i;

	if (use_pseudo) {
           context->values[VMWARE_HOST_TSC]=rdpmc(0x10000);
           context->values[VMWARE_ELAPSED_TIME]=rdpmc(0x10001);
           context->values[VMWARE_ELAPSED_APPARENT]=rdpmc(0x10002);
	}


#ifdef VMGUESTLIB
	static VMSessionId sessionId = 0;
	VMSessionId tmpSession;
	uint32_t temp32;
	uint64_t temp64;
	VMGuestLibError glError;

	if (use_guestlib) {

	glError = GuestLib_UpdateInfo(context->glHandle);
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr,"UpdateInfo failed: %s\n", 
		   GuestLib_GetErrorText(glError));
	   	   return PAPI_ECMP;
	}

	/* Retrieve and check the session ID */
	glError = GuestLib_GetSessionId(context->glHandle, &tmpSession);
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr, "Failed to get session ID: %s\n", 
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}

	if (tmpSession == 0) {
	   fprintf(stderr, "Error: Got zero sessionId from GuestLib\n");
	   return PAPI_ECMP;
	}

	if (sessionId == 0) {
	   sessionId = tmpSession;
	} else if (tmpSession != sessionId) {
	   sessionId = tmpSession;
	}

	glError = GuestLib_GetCpuLimitMHz(context->glHandle,&temp32);
	context->values[VMWARE_CPU_LIMIT_MHZ]=temp32;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr,"Failed to get CPU limit: %s\n", 
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}

	glError = GuestLib_GetCpuReservationMHz(context->glHandle,&temp32); 
	context->values[VMWARE_CPU_RESERVATION_MHZ]=temp32;
        if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr,"Failed to get CPU reservation: %s\n", 
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}
	
	glError = GuestLib_GetCpuShares(context->glHandle,&temp32);
	context->values[VMWARE_CPU_SHARES]=temp32;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr,"Failed to get cpu shares: %s\n", 
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}

	glError = GuestLib_GetCpuStolenMs(context->glHandle,&temp64);
	context->values[VMWARE_CPU_STOLEN_MS]=temp64;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   if (glError == VMGUESTLIB_ERROR_UNSUPPORTED_VERSION) {
	      context->values[VMWARE_CPU_STOLEN_MS]=0;
	      fprintf(stderr, "Skipping CPU stolen, not supported...\n");
	   } else {
	      fprintf(stderr, "Failed to get CPU stolen: %s\n", 
		      GuestLib_GetErrorText(glError));
	      return PAPI_ECMP;
	   }
	}

	glError = GuestLib_GetCpuUsedMs(context->glHandle,&temp64);
	context->values[VMWARE_CPU_USED_MS]=temp64;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr, "Failed to get used ms: %s\n", 
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}
	
	glError = GuestLib_GetElapsedMs(context->glHandle, &temp64);
	context->values[VMWARE_ELAPSED_MS]=temp64;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr, "Failed to get elapsed ms: %s\n",
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}

	glError = GuestLib_GetMemActiveMB(context->glHandle, &temp32);
	context->values[VMWARE_MEM_ACTIVE_MB]=temp32;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr, "Failed to get active mem: %s\n", 
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}
	
	glError = GuestLib_GetMemBalloonedMB(context->glHandle, &temp32);
	context->values[VMWARE_MEM_BALLOONED_MB]=temp32;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr, "Failed to get ballooned mem: %s\n", 
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}
	
	glError = GuestLib_GetMemLimitMB(context->glHandle, &temp32);
	context->values[VMWARE_MEM_LIMIT_MB]=temp32;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr,"Failed to get mem limit: %s\n", 
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}

        glError = GuestLib_GetMemMappedMB(context->glHandle, &temp32);
	context->values[VMWARE_MEM_MAPPED_MB]=temp32;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr, "Failed to get mapped mem: %s\n", 
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}

	glError = GuestLib_GetMemOverheadMB(context->glHandle, &temp32);
	context->values[VMWARE_MEM_OVERHEAD_MB]=temp32;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr, "Failed to get overhead mem: %s\n", 
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}

	glError = GuestLib_GetMemReservationMB(context->glHandle, &temp32);
	context->values[VMWARE_MEM_RESERVATION_MB]=temp32;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr, "Failed to get mem reservation: %s\n", 
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}

        glError = GuestLib_GetMemSharedMB(context->glHandle, &temp32);
	context->values[VMWARE_MEM_SHARED_MB]=temp32;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr, "Failed to get swapped mem: %s\n", 
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}

	glError = GuestLib_GetMemShares(context->glHandle, &temp32);
	context->values[VMWARE_MEM_SHARES]=temp32;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   if (glError == VMGUESTLIB_ERROR_NOT_AVAILABLE) {
	      context->values[VMWARE_MEM_SHARES]=0;
	      fprintf(stderr, "Skipping mem shares, not supported...\n");
	   } else {
	      fprintf(stderr, "Failed to get mem shares: %s\n", 
		      GuestLib_GetErrorText(glError));
	      return PAPI_ECMP;
	   }
	}

	glError = GuestLib_GetMemSwappedMB(context->glHandle, &temp32);
	context->values[VMWARE_MEM_SWAPPED_MB]=temp32;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr, "Failed to get swapped mem: %s\n",
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}
	
	glError = GuestLib_GetMemTargetSizeMB(context->glHandle, &temp64);
	context->values[VMWARE_MEM_TARGET_SIZE_MB]=temp64;
        if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   if (glError == VMGUESTLIB_ERROR_UNSUPPORTED_VERSION) {
	      context->values[VMWARE_MEM_TARGET_SIZE_MB]=0;
	      fprintf(stderr, "Skipping target mem size, not supported...\n");
	   } else {
	      fprintf(stderr, "Failed to get target mem size: %s\n", 
		      GuestLib_GetErrorText(glError));
	      return PAPI_ECMP;
	   }
	}

        glError = GuestLib_GetMemUsedMB(context->glHandle, &temp32);
	context->values[VMWARE_MEM_USED_MB]=temp32;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr, "Failed to get swapped mem: %s\n",
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}

        glError = GuestLib_GetHostProcessorSpeed(context->glHandle, &temp32); 
	context->values[VMWARE_HOST_CPU_MHZ]=temp32;
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr, "Failed to get host proc speed: %s\n", 
		   GuestLib_GetErrorText(glError));
	   return PAPI_ECMP;
	}
	}

#endif

	if (starting) {

	  for(i=0;i<VMWARE_MAX_COUNTERS;i++) {
	    context->start_values[i]=context->values[i];
	  }

	}

	return PAPI_OK;
}

/********************************************************************/
/* Below are the functions required by the PAPI component interface */
/********************************************************************/

/** This is called whenever a thread is initialized */
int
_vmware_init_thread( hwd_context_t *ctx )
{
	(void) ctx;


#ifdef VMGUESTLIB

	struct _vmware_context *context;
	VMGuestLibError glError;

	context=(struct _vmware_context *)ctx;

	if (use_guestlib) {
	   glError = GuestLib_OpenHandle(&(context->glHandle));
	   if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	      fprintf(stderr,"OpenHandle failed: %s\n", 
		   GuestLib_GetErrorText(glError));
	      return PAPI_ECMP;
	   }
	}

#endif

	return PAPI_OK;
}


/** Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
_vmware_init_component( int cidx )
{
    int retval = PAPI_OK;
  (void) cidx;

  int result;

	SUBDBG( "_vmware_init_component..." );

	/* Initialize and try to load the VMware library */
	/* Try to load the library. */
	result=LoadFunctions();

	if (result!=PAPI_OK) {
	   strncpy(_vmware_vector.cmp_info.disabled_reason,
		  "GuestLibTest: Failed to load shared library",
		   PAPI_MAX_STR_LEN);
       retval = PAPI_ECMP;
       goto fn_fail;
	}

	/* we know in advance how many events we want                       */
	/* for actual hardware this might have to be determined dynamically */

	/* Allocate memory for the our event table */
	_vmware_native_table = ( struct _vmware_native_event_entry * )
	  calloc( VMWARE_MAX_COUNTERS, sizeof ( struct _vmware_native_event_entry ));
	if ( _vmware_native_table == NULL ) {
        retval = PAPI_ENOMEM;
        goto fn_fail;
	}


#ifdef VMGUESTLIB

	/* Detect if GuestLib works */
	{

        VMGuestLibError glError;
        VMGuestLibHandle glHandle;

	use_guestlib=0;

	/* try to open */
	glError = GuestLib_OpenHandle(&glHandle);
	if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	   fprintf(stderr,"OpenHandle failed: %s\n", 
		   GuestLib_GetErrorText(glError));
	}
	else {
	   /* open worked, try to update */
	   glError = GuestLib_UpdateInfo(glHandle);
	   if (glError != VMGUESTLIB_ERROR_SUCCESS) {
	      fprintf(stderr,"UpdateInfo failed: %s\n", 
		      GuestLib_GetErrorText(glError));
	   }
	   else {
	      /* update worked, things work! */
	      use_guestlib=1;
	   }
	   /* shut things down */
	   glError = GuestLib_CloseHandle(glHandle);
	}

        }



	if (use_guestlib) {

	/* fill in the event table parameters */
	strcpy( _vmware_native_table[num_events].name,
		"CPU_LIMIT" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the upper limit of processor use in MHz "
		"available to the virtual machine.",
		PAPI_HUGE_STR_LEN);
	strcpy( _vmware_native_table[num_events].units,"MHz");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_CPU_LIMIT_MHZ;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"CPU_RESERVATION" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the minimum processing power in MHz "
		"reserved for the virtual machine.",
		PAPI_HUGE_STR_LEN);
	strcpy( _vmware_native_table[num_events].units,"MHz");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_CPU_RESERVATION_MHZ;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"CPU_SHARES" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the number of CPU shares allocated "
		"to the virtual machine.",
		PAPI_HUGE_STR_LEN);
	strcpy( _vmware_native_table[num_events].units,"shares");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_CPU_SHARES;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"CPU_STOLEN" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the number of milliseconds that the "
		"virtual machine was in a ready state (able to "
		"transition to a run state), but was not scheduled to run.",
		PAPI_HUGE_STR_LEN);
	strcpy( _vmware_native_table[num_events].units,"ms");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_CPU_STOLEN_MS;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"CPU_USED" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the number of milliseconds during which "
		"the virtual machine has used the CPU. This value "
		"includes the time used by the guest operating system "
		"and the time used by virtualization code for tasks for "
		"this virtual machine. You can combine this value with "
		"the elapsed time (VMWARE_ELAPSED) to estimate the "
		"effective virtual machine CPU speed. This value is a "
		"subset of elapsedMs.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"ms");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_CPU_USED_MS;
	_vmware_native_table[num_events].report_difference=1;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"ELAPSED" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the number of milliseconds that have passed "
		"in the virtual machine since it last started running on "
		"the server. The count of elapsed time restarts each time "
		"the virtual machine is powered on, resumed, or migrated "
		"using VMotion. This value counts milliseconds, regardless "
		"of whether the virtual machine is using processing power "
		"during that time. You can combine this value with the CPU "
		"time used by the virtual machine (VMWARE_CPU_USED) to "
		"estimate the effective virtual machine xCPU speed. "
		"cpuUsedMS is a subset of this value.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"ms");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_ELAPSED_MS;
	_vmware_native_table[num_events].report_difference=1;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"MEM_ACTIVE" );
	strncpy( _vmware_native_table[num_events].description,
		 "Retrieves the amount of memory the virtual machine is "
		 "actively using in MB - Its estimated working set size.",
		 PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
                 VMWARE_MEM_ACTIVE_MB;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"MEM_BALLOONED" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the amount of memory that has been reclaimed "
		"from this virtual machine by the vSphere memory balloon "
		"driver (also referred to as the 'vmemctl' driver) in MB.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_BALLOONED_MB;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"MEM_LIMIT" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the upper limit of memory that is available "
		"to the virtual machine in MB.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_LIMIT_MB;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"MEM_MAPPED" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the amount of memory that is allocated to "
		"the virtual machine in MB. Memory that is ballooned, "
		"swapped, or has never been accessed is excluded.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_MAPPED_MB;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"MEM_OVERHEAD" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the amount of 'overhead' memory associated "
		"with this virtual machine that is currently consumed "
		"on the host system in MB. Overhead memory is additional "
		"memory that is reserved for data structures required by "
		"the virtualization layer.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_OVERHEAD_MB;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"MEM_RESERVATION" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the minimum amount of memory that is "
		"reserved for the virtual machine in MB.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_RESERVATION_MB;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"MEM_SHARED" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the amount of physical memory associated "
		"with this virtual machine that is copy-on-write (COW) "
		"shared on the host in MB.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_SHARED_MB;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"MEM_SHARES" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the number of memory shares allocated to "
		"the virtual machine.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"shares");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_SHARES;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"MEM_SWAPPED" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the amount of memory that has been reclaimed "
		"from this virtual machine by transparently swapping "
		"guest memory to disk in MB.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_SWAPPED_MB;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"MEM_TARGET_SIZE" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the size of the target memory allocation "
		"for this virtual machine in MB.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_TARGET_SIZE_MB;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"MEM_USED" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the estimated amount of physical host memory "
		"currently consumed for this virtual machine's "
		"physical memory.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MB");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_MEM_USED_MB;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;

	strcpy( _vmware_native_table[num_events].name,
		"HOST_CPU" );
	strncpy( _vmware_native_table[num_events].description,
		"Retrieves the speed of the ESX system's physical "
		"CPU in MHz.",
		PAPI_HUGE_STR_LEN );
	strcpy( _vmware_native_table[num_events].units,"MHz");
	_vmware_native_table[num_events].which_counter=
	        VMWARE_HOST_CPU_MHZ;
	_vmware_native_table[num_events].report_difference=0;
	num_events++;
	}

#endif

	/* For VMWare Pseudo Performance Counters */
	if ( getenv( "PAPI_VMWARE_PSEUDOPERFORMANCE" ) ) {

	        use_pseudo=1;

		strcpy( _vmware_native_table[num_events].name,
			"HOST_TSC" );
		strncpy( _vmware_native_table[num_events].description,
			"Physical host TSC",
			PAPI_HUGE_STR_LEN );
		strcpy( _vmware_native_table[num_events].units,"cycles");
		_vmware_native_table[num_events].which_counter=
		        VMWARE_HOST_TSC;
	        _vmware_native_table[num_events].report_difference=1;
		num_events++;

		strcpy( _vmware_native_table[num_events].name,
			"ELAPSED_TIME" );
		strncpy( _vmware_native_table[num_events].description,
			"Elapsed real time in ns.",
			PAPI_HUGE_STR_LEN );
	        strcpy( _vmware_native_table[num_events].units,"ns");
		_vmware_native_table[num_events].which_counter=
		        VMWARE_ELAPSED_TIME;
	        _vmware_native_table[num_events].report_difference=1;
		num_events++;

		strcpy( _vmware_native_table[num_events].name,
			"ELAPSED_APPARENT" );
		strncpy( _vmware_native_table[num_events].description,
			"Elapsed apparent time in ns.",
			PAPI_HUGE_STR_LEN );
	        strcpy( _vmware_native_table[num_events].units,"ns");
		_vmware_native_table[num_events].which_counter=
		        VMWARE_ELAPSED_APPARENT;
	        _vmware_native_table[num_events].report_difference=1;
		num_events++;
	}

	if (num_events==0) {
	   strncpy(_vmware_vector.cmp_info.disabled_reason,
		  "VMware SDK not installed, and PAPI_VMWARE_PSEUDOPERFORMANCE not set",
		   PAPI_MAX_STR_LEN);
       retval = PAPI_ECMP;
       goto fn_fail;
	}

	_vmware_vector.cmp_info.num_native_events = num_events;

  fn_exit:
    _papi_hwd[cidx]->cmp_info.disabled = retval;
    return retval;
  fn_fail:
    goto fn_exit;
}

/** Setup the counter control structure */
int
_vmware_init_control_state( hwd_control_state_t *ctl )
{
  (void) ctl;

	return PAPI_OK;
}

/** Enumerate Native Events 
 @param EventCode is the event of interest
 @param modifier is one of PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS
 */
int
_vmware_ntv_enum_events( unsigned int *EventCode, int modifier )
{

	switch ( modifier ) {
			/* return EventCode of first event */
		case PAPI_ENUM_FIRST:
		     if (num_events==0) return PAPI_ENOEVNT;
		     *EventCode = 0;
		     return PAPI_OK;
		     break;
			/* return EventCode of passed-in Event */
		case PAPI_ENUM_EVENTS:{
		     int index = *EventCode;

		     if ( index < num_events - 1 ) {
			*EventCode = *EventCode + 1;
			return PAPI_OK;
		     } else {
			return PAPI_ENOEVNT;
		     }
		     break;
		}
		default:
		     return PAPI_EINVAL;
	}
	return PAPI_EINVAL;
}

int
_vmware_ntv_code_to_info(unsigned int EventCode, PAPI_event_info_t *info) 
{

  int index = EventCode;

  if ( ( index < 0) || (index >= num_events )) return PAPI_ENOEVNT;

  strncpy( info->symbol, _vmware_native_table[index].name, 
           sizeof(info->symbol));

  strncpy( info->long_descr, _vmware_native_table[index].description, 
           sizeof(info->symbol));

  strncpy( info->units, _vmware_native_table[index].units, 
           sizeof(info->units));

  return PAPI_OK;
}


/** Takes a native event code and passes back the name 
 @param EventCode is the native event code
 @param name is a pointer for the name to be copied to
 @param len is the size of the string
 */
int
_vmware_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
	int index = EventCode;

	if ( index >= 0 && index < num_events ) {
	   strncpy( name, _vmware_native_table[index].name, len );
	}
	return PAPI_OK;
}

/** Takes a native event code and passes back the event description
 @param EventCode is the native event code
 @param name is a pointer for the description to be copied to
 @param len is the size of the string
 */
int
_vmware_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
  int index = EventCode;

	if ( index >= 0 && index < num_events ) {
	   strncpy( name, _vmware_native_table[index].description, len );
	}
	return PAPI_OK;
}

/** Triggered by eventset operations like add or remove */
int
_vmware_update_control_state( hwd_control_state_t *ctl, 
			      NativeInfo_t *native, 
			      int count, 
			      hwd_context_t *ctx )
{
	(void) ctx;

	struct _vmware_control_state *control;

	int i, index;

	control=(struct _vmware_control_state *)ctl;

	for ( i = 0; i < count; i++ ) {
	    index = native[i].ni_event;
	    control->which_counter[i]=_vmware_native_table[index].which_counter;
	    native[i].ni_position = i;
	}
	control->num_events=count;

	return PAPI_OK;
}

/** Triggered by PAPI_start() */
int
_vmware_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
	struct _vmware_context *context;
	(void) ctl;

	context=(struct _vmware_context *)ctx;

	_vmware_hardware_read( context, 1 );

	return PAPI_OK;
}

/** Triggered by PAPI_stop() */
int
_vmware_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

	struct _vmware_context *context;
	(void) ctl;

	context=(struct _vmware_context *)ctx;

	_vmware_hardware_read( context, 0 );	

	return PAPI_OK;
}

/** Triggered by PAPI_read() */
int
_vmware_read( hwd_context_t *ctx, 
	      hwd_control_state_t *ctl,
	      long_long **events, int flags )
{

	struct _vmware_context *context;
	struct _vmware_control_state *control;

	(void) flags;
	int i;

	context=(struct _vmware_context *)ctx;
	control=(struct _vmware_control_state *)ctl;

	_vmware_hardware_read( context, 0 );

	for (i=0; i<control->num_events; i++) {
	  
	  if (_vmware_native_table[
              _vmware_native_table[control->which_counter[i]].which_counter].
             report_difference) {
	     control->value[i]=context->values[control->which_counter[i]]-
	                       context->start_values[control->which_counter[i]];
	  } else {
	     control->value[i]=context->values[control->which_counter[i]];
	  }
	  //	  printf("%d %d %lld-%lld=%lld\n",i,control->which_counter[i],
	  // context->values[control->which_counter[i]],
	  //	 context->start_values[control->which_counter[i]],
	  //	 control->value[i]);

	}

	*events = control->value;

	return PAPI_OK;
}

/** Triggered by PAPI_write(), but only if the counters are running */
/*    otherwise, the updated state is written to ESI->hw_start      */
int
_vmware_write( hwd_context_t * ctx, hwd_control_state_t * ctrl, long_long events[] )
{
	(void) ctx;
	(void) ctrl;
	(void) events;
	SUBDBG( "_vmware_write... %p %p", ctx, ctrl );
	/* FIXME... this should actually carry out the write, though     */
	/*  this is non-trivial as which counter being written has to be */
	/*  determined somehow.                                          */
	return PAPI_OK;
}

/** Triggered by PAPI_reset */
int
_vmware_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
	(void) ctx;
	(void) ctl;

	return PAPI_OK;
}

/** Shutting down a context */
int
_vmware_shutdown_thread( hwd_context_t *ctx )
{
	(void) ctx;

#ifdef VMGUESTLIB
        VMGuestLibError glError;
	struct _vmware_context *context;

	context=(struct _vmware_context *)ctx;

	if (use_guestlib) {
           glError = GuestLib_CloseHandle(context->glHandle);
           if (glError != VMGUESTLIB_ERROR_SUCCESS) {
               fprintf(stderr, "Failed to CloseHandle: %s\n", 
		       GuestLib_GetErrorText(glError));
               return PAPI_ECMP;
	   }
	}
#endif

	return PAPI_OK;
}

/** Triggered by PAPI_shutdown() */
int
_vmware_shutdown_component( void )
{

#ifdef VMGUESTLIB
	if (dlclose(dlHandle)) {
		fprintf(stderr, "dlclose failed\n");
		return EXIT_FAILURE;
	}
#endif

	return PAPI_OK;
}


/** This function sets various options in the component
 @param ctx
 @param code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
 @param option
 */
int
_vmware_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{

	(void) ctx;
	(void) code;
	(void) option;

	SUBDBG( "_vmware_ctl..." );

	return PAPI_OK;
}

/** This function has to set the bits needed to count different domains
 In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 By default return PAPI_EINVAL if none of those are specified
 and PAPI_OK with success
 PAPI_DOM_USER is only user context is counted
 PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 PAPI_DOM_ALL   is all of the domains
 */
int
_vmware_set_domain( hwd_control_state_t *ctl, int domain )
{
	(void) ctl;

	int found = 0;
	SUBDBG( "_vmware_set_domain..." );
	if ( PAPI_DOM_USER & domain ) {
		SUBDBG( " PAPI_DOM_USER " );
		found = 1;
	}
	if ( PAPI_DOM_KERNEL & domain ) {
		SUBDBG( " PAPI_DOM_KERNEL " );
		found = 1;
	}
	if ( PAPI_DOM_OTHER & domain ) {
		SUBDBG( " PAPI_DOM_OTHER " );
		found = 1;
	}
	if ( PAPI_DOM_ALL & domain ) {
		SUBDBG( " PAPI_DOM_ALL " );
		found = 1;
	}
	if ( !found ) {
		return ( PAPI_EINVAL );
	}
	return PAPI_OK;
}

/** Vector that points to entry points for our component */
papi_vector_t _vmware_vector = {
	.cmp_info = {
		/* default component information (unspecified values are initialized to 0) */
		.name = "vmware",
		.short_name = "vmware",
		.description = "Provide support for VMware vmguest and pseudo counters",
		.version = "5.0",
		.num_mpx_cntrs = VMWARE_MAX_COUNTERS,
		.num_cntrs = VMWARE_MAX_COUNTERS,
		.default_domain = PAPI_DOM_USER,
		.available_domains = PAPI_DOM_USER,
		.default_granularity = PAPI_GRN_THR,
		.available_granularities = PAPI_GRN_THR,
		.hardware_intr_sig = PAPI_INT_SIGNAL,

		/* component specific cmp_info initializations */
		.fast_real_timer = 0,
		.fast_virtual_timer = 0,
		.attach = 0,
		.attach_must_ptrace = 0,
	},
	/* sizes of framework-opaque component-private structures */
	.size = {
		.context = sizeof ( struct _vmware_context ),
		.control_state = sizeof ( struct _vmware_control_state ),
		.reg_value = sizeof ( struct _vmware_register ),
		.reg_alloc = sizeof ( struct _vmware_reg_alloc ),
	}
	,
	/* function pointers in this component */
	.init_thread =        _vmware_init_thread,
	.init_component =     _vmware_init_component,
	.init_control_state = _vmware_init_control_state,
	.start =              _vmware_start,
	.stop =               _vmware_stop,
	.read =               _vmware_read,
	.write =              _vmware_write,
	.shutdown_thread =    _vmware_shutdown_thread,
	.shutdown_component = _vmware_shutdown_component,
	.ctl =                _vmware_ctl,

	.update_control_state = _vmware_update_control_state,
	.set_domain = _vmware_set_domain,
	.reset = _vmware_reset,

	.ntv_enum_events = _vmware_ntv_enum_events,
	.ntv_code_to_name = _vmware_ntv_code_to_name,
	.ntv_code_to_descr = _vmware_ntv_code_to_descr,
	.ntv_code_to_info = _vmware_ntv_code_to_info,

};

