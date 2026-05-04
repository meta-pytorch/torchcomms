/** linux-host_micpower.c
 *	@author James Ralph
 *	ralph@icl.utk.edu
 *
 *	@ingroup papi_components
 *
 *	@brief
 *		This component wraps the MicAccessAPI to provide hostside 
 *		power information for attached Intel Xeon Phi (MIC) cards.
*/ 

/* From intel examples, see $(mic_dir)/sysmgt/sdk/Examples/Usage */
#define MAX_DEVICES (32)
#define EVENTS_PER_DEVICE 10
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <dlfcn.h> 

#include "MicAccessTypes.h"
#include "MicBasicTypes.h"
#include "MicAccessErrorTypes.h"
#include "MicAccessApi.h"
#include "MicPowerManagerAPI.h"

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

void (*_dl_non_dynamic_init)(void) __attribute__((weak));

/* This is a guess, refine this later */
#define UPDATEFREQ 500000

papi_vector_t _host_micpower_vector;

typedef struct host_micpower_register {
	/** Corresponds to counter slot, indexed from 1, 0 has a special meaning */
	unsigned int selector;
} host_micpower_register_t;

typedef struct host_micpower_reg_alloc {
	host_micpower_register_t ra_bits;
} host_micpower_reg_alloc_t;

/** Internal structure used to build the table of events */
typedef struct host_micpower_native_event_entry {
	host_micpower_register_t resources; 
	char name[PAPI_MAX_STR_LEN];
	char description[PAPI_MAX_STR_LEN];
	char units[3];
} host_micpower_native_event_entry_t;

/** Per-eventset structure used to hold control flags. */
typedef struct host_micpower_control_state {
	int num_events;
	int resident[MAX_DEVICES*EVENTS_PER_DEVICE];
	long long counts[MAX_DEVICES*EVENTS_PER_DEVICE];
	long long lastupdate[MAX_DEVICES];
} host_micpower_control_state_t;

/** Per-thread data */
typedef struct host_micpower_context {
	host_micpower_control_state_t state;
} host_micpower_context_t; 

/* Global state info */
static MicDeviceOnSystem adapters[MAX_DEVICES];
static HANDLE handles[MAX_DEVICES];
static long long lastupdate[MAX_DEVICES];
static HANDLE accessHandle = NULL;
static U32 nAdapters = MAX_DEVICES;

static void* mic_access 	= 	NULL;
static void* scif_access	=	NULL;

#undef MICACCESS_API
#define MICACCESS_API __attribute__((weak))
const char *MicGetErrorString(U32);
U32 MICACCESS_API MicCloseAdapter(HANDLE);
U32 MICACCESS_API MicInitAPI(HANDLE *, ETarget, MicDeviceOnSystem *, U32 *);
U32 MICACCESS_API MicCloseAPI(HANDLE *);
U32 MICACCESS_API MicInitAdapter(HANDLE *, MicDeviceOnSystem *);
U32 MICACCESS_API MicGetPowerUsage(HANDLE, MicPwrUsage *);

const char *(*MicGetErrorStringPtr)(U32);
U32 (*MicCloseAdapterPtr)(HANDLE);
U32 (*MicInitAPIPtr)(HANDLE *, ETarget, MicDeviceOnSystem *, U32 *);
U32 (*MicCloseAPIPtr)(HANDLE *);
U32 (*MicInitAdapterPtr)(HANDLE *, MicDeviceOnSystem *);
U32 (*MicGetPowerUsagePtr)(HANDLE, MicPwrUsage *);
static host_micpower_native_event_entry_t *native_events_table = NULL;

struct powers {
		int total0;
		int total1;
		int	inst;
		int imax;
		int pcie;
		int c2x3;
		int c2x4;
		int vccp;
		int vddg;
		int vddq;
};

typedef union {
		struct powers power;
		int array[EVENTS_PER_DEVICE]; 
} power_t;

static power_t cached_values[MAX_DEVICES];

static int 
loadFunctionPtrs()
{
	/* Attempt to guess if we were statically linked to libc, if so bail */
	if ( _dl_non_dynamic_init != NULL ) {
		strncpy(_host_micpower_vector.cmp_info.disabled_reason, "The host_micpower component does not support statically linking of libc.", PAPI_MAX_STR_LEN);
		return PAPI_ENOSUPP;
	}

	  /* Need to link in the cuda libraries, if not found disable the component */
	scif_access = dlopen("libscif.so", RTLD_NOW | RTLD_GLOBAL);
    if (NULL == scif_access)
    {
        snprintf(_host_micpower_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "Problem loading the SCIF library: %s\n", dlerror());
			_host_micpower_vector.cmp_info.disabled = 1;
        return ( PAPI_ENOSUPP );
    }

    mic_access = dlopen("libMicAccessSDK.so", RTLD_NOW | RTLD_GLOBAL);
    if (NULL == mic_access)
    {
        snprintf(_host_micpower_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "Problem loading libMicAccessSDK.so: %s\n", dlerror());
			_host_micpower_vector.cmp_info.disabled = 1;
        return ( PAPI_ENOSUPP );
    }

	MicGetErrorStringPtr = dlsym(mic_access, "MicGetErrorString");
	if (dlerror() != NULL)
	{
			strncpy(_host_micpower_vector.cmp_info.disabled_reason, "MicAccessSDK function MicGetErrorString not found.",PAPI_MAX_STR_LEN);
			_host_micpower_vector.cmp_info.disabled = 1;
			return ( PAPI_ENOSUPP );
	}
	MicCloseAdapterPtr = dlsym(mic_access, "MicCloseAdapter");
	if (dlerror() != NULL)
	{
			strncpy(_host_micpower_vector.cmp_info.disabled_reason, "MicAccessSDK function MicCloseAdapter not found.",PAPI_MAX_STR_LEN);
			_host_micpower_vector.cmp_info.disabled = 1;
			return ( PAPI_ENOSUPP );
	}
	MicInitAPIPtr = dlsym(mic_access, "MicInitAPI");
	if (dlerror() != NULL)
	{
			strncpy(_host_micpower_vector.cmp_info.disabled_reason, "MicAccessSDK function MicInitAPI not found.",PAPI_MAX_STR_LEN);
			_host_micpower_vector.cmp_info.disabled = 1;
			return ( PAPI_ENOSUPP );
	}
	MicCloseAPIPtr = dlsym(mic_access, "MicCloseAPI");
	if (dlerror() != NULL)
	{
			strncpy(_host_micpower_vector.cmp_info.disabled_reason, "MicAccessSDK function MicCloseAPI not found.",PAPI_MAX_STR_LEN);
			_host_micpower_vector.cmp_info.disabled = 1;
			return ( PAPI_ENOSUPP );
	}
	MicInitAdapterPtr = dlsym(mic_access, "MicInitAdapter");
	if (dlerror() != NULL)
	{
			strncpy(_host_micpower_vector.cmp_info.disabled_reason, "MicAccessSDK function MicInitAdapter not found.",PAPI_MAX_STR_LEN);
			_host_micpower_vector.cmp_info.disabled = 1;
			return ( PAPI_ENOSUPP );
	}

	MicGetPowerUsagePtr = dlsym(mic_access, "MicGetPowerUsage");
	if (dlerror() != NULL)
	{
			strncpy(_host_micpower_vector.cmp_info.disabled_reason, "MicAccessSDK function MicGetPowerUsage not found.",PAPI_MAX_STR_LEN);
			_host_micpower_vector.cmp_info.disabled = 1;
			return ( PAPI_ENOSUPP );
	}

	return 0;
}


/* ###############################################
 * 			Component Interface code 
 * ############################################### */


int 
_host_micpower_init_component( int cidx ) 
{
    int retval = PAPI_OK;
	U32 ret = MIC_ACCESS_API_ERROR_UNKNOWN;
	U32 adapterNum = 0;
	U32 throwaway = 1;

	_host_micpower_vector.cmp_info.CmpIdx = cidx;

	if ( loadFunctionPtrs() ) {
        retval = PAPI_ENOSUPP;
		goto fn_fail;
	}

	memset( lastupdate, 0x0, sizeof(lastupdate));
	memset( cached_values, 0x0, sizeof(struct powers)*MAX_DEVICES );
	ret = MicInitAPIPtr( &accessHandle, eTARGET_SCIF_DRIVER, adapters, &nAdapters );
	if ( MIC_ACCESS_API_SUCCESS != ret ) {
		snprintf( _host_micpower_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "Failed to init: %s", MicGetErrorStringPtr(ret));
		MicCloseAPIPtr(&accessHandle);
        retval = PAPI_ENOSUPP;
		goto fn_fail;
	}
	/* Sanity check on array size */
	if ( nAdapters >= MAX_DEVICES ) {
		snprintf(_host_micpower_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "Too many MIC cards [%d] found, bailing.", nAdapters);
		MicCloseAPIPtr(&accessHandle);
        retval = PAPI_ENOSUPP;
		goto fn_fail;
	}

/* XXX: This code initializes a token for each adapter, in testing this appeared to be required/
 *	One has to call MicInitAdapter() before calling into that adapter's entries */
	for (adapterNum=0; adapterNum < nAdapters; adapterNum++) {
			ret = MicInitAPIPtr( &handles[adapterNum], eTARGET_SCIF_DRIVER, adapters, &throwaway );
			throwaway = 1;
			if (MIC_ACCESS_API_SUCCESS != ret) {
					fprintf(stderr, "%d:MicInitAPI carps: %s\n", __LINE__, MicGetErrorStringPtr(ret));
					nAdapters = adapterNum;
					for (adapterNum=0; adapterNum < nAdapters; adapterNum++)
							MicCloseAdapterPtr( handles[adapterNum] );
					MicCloseAPIPtr( &accessHandle );
					snprintf(_host_micpower_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
						"Failed to initialize card %d's interface.", nAdapters);
                    retval = PAPI_ENOSUPP;
					goto fn_fail;
			}
			ret = MicInitAdapterPtr(&handles[adapterNum], &adapters[adapterNum]);
			if (MIC_ACCESS_API_SUCCESS != ret) {
					fprintf(stderr, "%d:MicInitAdapter carps: %s\n", __LINE__, MicGetErrorStringPtr(ret));
					nAdapters = adapterNum;
					for (adapterNum=0; adapterNum < nAdapters; adapterNum++)
							MicCloseAdapterPtr( handles[adapterNum] );
					MicCloseAPIPtr( &accessHandle );
					snprintf(_host_micpower_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
						"Failed to initialize card %d's interface.", nAdapters);
                    retval = PAPI_ENOSUPP;
					goto fn_fail;
			}
	}

	native_events_table = ( host_micpower_native_event_entry_t*)papi_malloc( nAdapters * EVENTS_PER_DEVICE * sizeof(host_micpower_native_event_entry_t));
	if ( NULL == native_events_table ) {
        retval = PAPI_ENOMEM;
        goto fn_fail;
	}
	for (adapterNum=0; adapterNum < nAdapters; adapterNum++) {
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE].name, PAPI_MAX_STR_LEN, "mic%d:tot0", adapterNum);
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE].description, PAPI_MAX_STR_LEN, "Total power utilization, Averaged over Time Window 0 (uWatts)");
		native_events_table[adapterNum*EVENTS_PER_DEVICE].resources.selector = adapterNum*EVENTS_PER_DEVICE + 1;
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE].units, PAPI_MIN_STR_LEN, "uW");

        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 1].name, PAPI_MAX_STR_LEN, "mic%d:tot1", adapterNum);
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 1].description, PAPI_MAX_STR_LEN, "Total power utilization, Averaged over Time Window 1 (uWatts)");
		native_events_table[adapterNum*EVENTS_PER_DEVICE + 1].resources.selector = adapterNum*EVENTS_PER_DEVICE + 2;
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 1].units, PAPI_MIN_STR_LEN, "uW");

        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 2].name, PAPI_MAX_STR_LEN, "mic%d:pcie", adapterNum);
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 2].description, PAPI_MAX_STR_LEN, "PCI-E connector power (uWatts)");
		native_events_table[adapterNum*EVENTS_PER_DEVICE + 2].resources.selector = adapterNum*EVENTS_PER_DEVICE + 3;
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 2].units, PAPI_MIN_STR_LEN, "uW");

        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 3].name, PAPI_MAX_STR_LEN, "mic%d:inst", adapterNum);
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 3].description, PAPI_MAX_STR_LEN, "Instantaneous power (uWatts)");
		native_events_table[adapterNum*EVENTS_PER_DEVICE + 3].resources.selector = adapterNum*EVENTS_PER_DEVICE + 4;
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 3].units, PAPI_MIN_STR_LEN, "uW");

        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 4].name, PAPI_MAX_STR_LEN, "mic%d:imax", adapterNum);
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 4].description, PAPI_MAX_STR_LEN, "Max instantaneous power (uWatts)");
		native_events_table[adapterNum*EVENTS_PER_DEVICE + 4].resources.selector = adapterNum*EVENTS_PER_DEVICE + 5;
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 4].units, PAPI_MIN_STR_LEN, "uW");

        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 5].name, PAPI_MAX_STR_LEN, "mic%d:c2x3", adapterNum);
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 5].description, PAPI_MAX_STR_LEN, "2x3 connector power (uWatts)");
		native_events_table[adapterNum*EVENTS_PER_DEVICE + 5].resources.selector = adapterNum*EVENTS_PER_DEVICE + 6;
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 5].units, PAPI_MIN_STR_LEN, "uW");

        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 6].name, PAPI_MAX_STR_LEN, "mic%d:c2x4", adapterNum);
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 6].description, PAPI_MAX_STR_LEN, "2x4 connector power (uWatts)");
		native_events_table[adapterNum*EVENTS_PER_DEVICE + 6].resources.selector = adapterNum*EVENTS_PER_DEVICE + 7;
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 6].units, PAPI_MIN_STR_LEN, "uW");

        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 7].name, PAPI_MAX_STR_LEN, "mic%d:vccp", adapterNum);
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 7].description, PAPI_MAX_STR_LEN, "Core rail (uVolts)");
		native_events_table[adapterNum*EVENTS_PER_DEVICE + 7].resources.selector = adapterNum*EVENTS_PER_DEVICE + 8;
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 7].units, PAPI_MIN_STR_LEN, "uV");

        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 8].name, PAPI_MAX_STR_LEN, "mic%d:vddg", adapterNum);
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 8].description, PAPI_MAX_STR_LEN, "Uncore rail (uVolts)");
		native_events_table[adapterNum*EVENTS_PER_DEVICE + 8].resources.selector = adapterNum*EVENTS_PER_DEVICE + 9;
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 8].units, PAPI_MIN_STR_LEN, "uV");

        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 9].name, PAPI_MAX_STR_LEN, "mic%d:vddq", adapterNum);
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 9].description, PAPI_MAX_STR_LEN, "Memory subsystem rail (uVolts)");
		native_events_table[adapterNum*EVENTS_PER_DEVICE + 9].resources.selector = adapterNum*EVENTS_PER_DEVICE + 10;
        snprintf(native_events_table[adapterNum*EVENTS_PER_DEVICE + 9].units, PAPI_MIN_STR_LEN, "uV");
	}

  fn_exit:
	_host_micpower_vector.cmp_info.num_cntrs = EVENTS_PER_DEVICE*nAdapters;
	_host_micpower_vector.cmp_info.num_mpx_cntrs = EVENTS_PER_DEVICE*nAdapters;
	_host_micpower_vector.cmp_info.num_native_events = EVENTS_PER_DEVICE*nAdapters;

    _papi_hwd[cidx]->cmp_info.disabled = retval;
    return retval;

  fn_fail:
	nAdapters = 0;
    goto fn_exit;
}

int _host_micpower_init_thread( hwd_context_t *ctx) {
	(void)ctx;
	return PAPI_OK;
}

int
_host_micpower_shutdown_component( void ) {
	U32 i = 0;
	for( i=0; i<nAdapters; i++) {
		MicCloseAdapterPtr( handles[i] );
	}

	papi_free(native_events_table);
	return PAPI_OK;
}
	
int
_host_micpower_shutdown_thread( hwd_context_t *ctx ) {
    (void) ctx;
	return PAPI_OK;
}

int _host_micpower_init_control_state ( hwd_control_state_t *ctl ) {
	host_micpower_control_state_t *state = (host_micpower_control_state_t*) ctl;
	memset( state, 0, sizeof(host_micpower_control_state_t));

	return PAPI_OK;
}

int _host_micpower_update_control_state(hwd_control_state_t *ctl, 
										NativeInfo_t *info, 
										int count,
										hwd_context_t* ctx ) {

	(void) ctx;
	int i, index;
	
	host_micpower_control_state_t *state = (host_micpower_control_state_t*)ctl;

	for (i=0; i<MAX_DEVICES*EVENTS_PER_DEVICE; i++)
		state->resident[i] = 0;

	for (i=0; i < count; i++) {
		index = info[i].ni_event&PAPI_NATIVE_AND_MASK;
		info[i].ni_position=native_events_table[index].resources.selector-1;
		state->resident[index] = 1;
	}
	state->num_events = count;

	return PAPI_OK;
}

int
_host_micpower_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
	(void) ctx;
	(void) ctl;
	return PAPI_OK;
}

static int 
read_power( struct powers *pwr, int which_one ) 
{
	MicPwrUsage power;
	U32 ret = MIC_ACCESS_API_ERROR_UNKNOWN;

	if ( which_one < 0 || which_one > (int)nAdapters )
		return PAPI_ENOEVNT;
	

	ret = MicGetPowerUsagePtr(handles[which_one], &power);
	if (MIC_ACCESS_API_SUCCESS != ret) {
			fprintf(stderr,"Oops MicGetPowerUsage failed: %s\n", 
							MicGetErrorStringPtr(ret));
			return PAPI_ECMP;
	}

	pwr->total0 = power.total0.prr;
	pwr->total1 = power.total1.prr;
	pwr->inst = power.inst.prr;
	pwr->imax = power.imax.prr;
	pwr->pcie = power.pcie.prr;
	pwr->c2x3 = power.c2x3.prr;
	pwr->c2x4 = power.c2x4.prr;
	pwr->vccp = power.vccp.pwr;
	pwr->vddg = power.vddg.pwr;
	pwr->vddq = power.vddq.pwr;

	return PAPI_OK;
}

int
_host_micpower_read( hwd_context_t *ctx, hwd_control_state_t *ctl, 
					 long long **events, int flags) 
{
	(void)flags;
	(void)events;
	(void)ctx;
	unsigned int i,j;
	int needs_update = 0;
	host_micpower_control_state_t* control = (host_micpower_control_state_t*)ctl;
	long long now = PAPI_get_real_usec();

	for( i=0; i<nAdapters; i++) {
			needs_update = 0;
			for (j=0; j<EVENTS_PER_DEVICE; j++) {
				if ( control->resident[EVENTS_PER_DEVICE*i+j]) {
						needs_update = 1;
						break;
				}
			}

			if ( needs_update ) {
					/* Do the global update */
					if ( now >= lastupdate[i] + UPDATEFREQ) {
							read_power( &cached_values[i].power, i );
							lastupdate[i] = now;
					}
					/* update from cached values */
					if ( control->lastupdate[i] < lastupdate[i]) {
							control->lastupdate[i] = lastupdate[i];
					}
					for (j=0; j<EVENTS_PER_DEVICE; j++) {
						if ( control->resident[EVENTS_PER_DEVICE*i+j] ) {
							control->counts[EVENTS_PER_DEVICE*i+j] = (long long)cached_values[i].array[j];
						}
					}
			}
	}

	*events = control->counts;
	return PAPI_OK;
}

int
_host_micpower_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
	(void)ctx;
	int needs_update = 0;
	unsigned int i,j;
	host_micpower_control_state_t* control = (host_micpower_control_state_t*)ctl;
	long long now = PAPI_get_real_usec();

	for( i=0; i<nAdapters; i++) {
			needs_update = 0;
			for (j=0; j<EVENTS_PER_DEVICE; j++) {
				if ( control->resident[EVENTS_PER_DEVICE*i+j]) {
						needs_update = 1;
						break;
				}
			}

			if ( needs_update ) {
					/* Do the global update */
					if ( now >= lastupdate[i] + UPDATEFREQ) {
							read_power( &cached_values[i].power, i );
							lastupdate[i] = now;
					}
					/* update from cached values */
					if ( control->lastupdate[i] < lastupdate[i]) {
							control->lastupdate[i] = lastupdate[i];
					}
					for (j=0; j<EVENTS_PER_DEVICE; j++) {
						if ( control->resident[EVENTS_PER_DEVICE*i+j] ) {
							control->counts[EVENTS_PER_DEVICE*i+j] = (long long)cached_values[i].array[j];
						}
					}
			}
	}
	return PAPI_OK;

}

int _host_micpower_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	int index;
	switch (modifier) {
		case PAPI_ENUM_FIRST:
			if (0 == _host_micpower_vector.cmp_info.num_cntrs)
				return PAPI_ENOEVNT;
			*EventCode = 0;
			return PAPI_OK;
		case PAPI_ENUM_EVENTS:
			index = *EventCode;
			if ( index < _host_micpower_vector.cmp_info.num_cntrs - 1) {
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

int
_host_micpower_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
	unsigned int code = EventCode & PAPI_NATIVE_AND_MASK;
	if ( code < _host_micpower_vector.cmp_info.num_cntrs ) {
		strncpy( name, native_events_table[code].name, len);
		return PAPI_OK;
	}

	return PAPI_ENOEVNT;
}

int
_host_micpower_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
	unsigned int code = EventCode & PAPI_NATIVE_AND_MASK;
	if ( code < _host_micpower_vector.cmp_info.num_cntrs ) {
		strncpy( name, native_events_table[code].description, len );
		return PAPI_OK;
	}

	return PAPI_ENOEVNT;
}

int
_host_micpower_ntv_code_to_info( unsigned int EventCode, PAPI_event_info_t *info)
{
	unsigned int code = EventCode & PAPI_NATIVE_AND_MASK;
	if ( code >= _host_micpower_vector.cmp_info.num_cntrs)
		return PAPI_ENOEVNT;
	strncpy( info->symbol, native_events_table[code].name, sizeof(info->symbol) );
	strncpy( info->long_descr, native_events_table[code].description, sizeof(info->long_descr) );
	strncpy( info->units, native_events_table[code].units, sizeof(info->units) );
	return PAPI_OK;
}

int
_host_micpower_ctl( hwd_context_t* ctx, int code, _papi_int_option_t *option)
{
	(void)ctx;
	(void)code;
	(void)option;
	return PAPI_OK;
}

int
_host_micpower_set_domain( hwd_control_state_t* ctl, int domain)
{
	(void)ctl;
	if ( PAPI_DOM_ALL != domain )
	    return PAPI_EINVAL;
	return PAPI_OK;
}

papi_vector_t _host_micpower_vector = {
	.cmp_info = {
		.name = "host_micpower", 
		.short_name = "host_micpower", 
		.description = "A host-side component to read power usage on MIC guest cards.",
		.version = "0.1",
		.support_version = "n/a",
		.kernel_version = "n/a",
		.num_cntrs = 0,
		.num_mpx_cntrs = 0,
		.default_domain 			= PAPI_DOM_ALL,
		.available_domains 			= PAPI_DOM_ALL,
		.default_granularity 		= PAPI_GRN_SYS,
		.available_granularities 	= PAPI_GRN_SYS,
		.hardware_intr_sig 			= PAPI_INT_SIGNAL,
	}, 

	.size  = {
		.context 		= sizeof(host_micpower_context_t), 
		.control_state	= sizeof(host_micpower_control_state_t),
		.reg_value		= sizeof(host_micpower_register_t),
		.reg_alloc		= sizeof(host_micpower_reg_alloc_t),
	},

	.start					= _host_micpower_start,
	.stop					= _host_micpower_start,
	.read					= _host_micpower_read, 
	.reset					= NULL,
	.write					= NULL,
	.init_component			= _host_micpower_init_component,
	.init_thread			= _host_micpower_init_thread,
	.init_control_state		= _host_micpower_init_control_state,
	.update_control_state	= _host_micpower_update_control_state,
	.ctl					= _host_micpower_ctl, 
	.shutdown_thread		= _host_micpower_shutdown_thread,
	.shutdown_component		= _host_micpower_shutdown_component,
	.set_domain				= _host_micpower_set_domain,

	.ntv_enum_events		= _host_micpower_ntv_enum_events, 
	.ntv_code_to_name		= _host_micpower_ntv_code_to_name,
	.ntv_code_to_descr		= _host_micpower_ntv_code_to_descr,
	.ntv_code_to_info		= _host_micpower_ntv_code_to_info,

};
