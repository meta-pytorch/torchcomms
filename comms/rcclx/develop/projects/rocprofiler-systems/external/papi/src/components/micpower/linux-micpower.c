#include <string.h>
#include <unistd.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#include "linux-micpower.h"

/* Intel says
----
The power measurements can be obtained from the host as well as the MIC card 
over a 50msec interval. The SMC is designed to sample power consumption only 
every 50mSecs.
----
**/
#define REFRESH_LAT 50000

#define INVALID_RESULT -1000000L
#define MICPOWER_NUMBER_OF_NATIVE_EVENTS 16 

papi_vector_t _micpower_vector;

static MICPOWER_native_event_entry_t _micpower_native_events[] = {
		{ 	.name 				= "tot0",
				.units 				= "uW",
				.description 		= "Total power, win 0", 
				.resources.selector	= 1
		},
		{ 	.name 				= "tot1",
				.units 				= "uW",
				.description 		= "Total power, win 1", 
				.resources.selector	= 2
		},
		{ 	.name 				= "pcie",
				.units 				= "uW",
				.description 		= "PCI-E connector power", 
				.resources.selector	= 3
		},
		{ 	.name 				= "inst",
				.units 				= "uW",
				.description 		= "Instantaneous power", 
				.resources.selector	= 4
		},
		{ 	.name 				= "imax",
				.units 				= "uW",
				.description 		= "Max Instantaneous power", 
				.resources.selector	= 5
		},
		{ 	.name 				= "c2x3",
				.units 				= "uW",
				.description 		= "2x3 connector power", 
				.resources.selector	= 6
		},
		{ 	.name 				= "c2x4",
				.units 				= "uW",
				.description 		= "2x4 connector power", 
				.resources.selector	= 7
		},
		{ 	.name 				= "vccp:pwr",
				.units 				= "uW",
				.description 		= "Core rail; Power reading", 
				.resources.selector	= 8
		},
		{ 	.name 				= "vccp:cur",
				.units 				= "uA",
				.description 		= "Core rail; Current", 
				.resources.selector	= 9
		},
		{ 	.name 				= "vccp:volt",
				.units 				= "uV",
				.description 		= "Core rail; Voltage", 
				.resources.selector	= 10
		},
		{ 	.name 				= "vddg:pwr",
				.units 				= "uW",
				.description 		= "Uncore rail; Power reading", 
				.resources.selector	= 11
		},
		{ 	.name 				= "vddg:cur",
				.units 				= "uA",
				.description 		= "Uncore rail; Current", 
				.resources.selector	= 12
		},
		{ 	.name 				= "vddg:volt",
				.units 				= "uV",
				.description 		= "Uncore rail; Voltage", 
				.resources.selector	= 13
		},
		{ 	.name 				= "vddq:pwr",
				.units 				= "uW",
				.description 		= "Memory subsystem rail; Power reading", 
				.resources.selector	= 14
		},
		{ 	.name 				= "vddq:cur",
				.units 				= "uA",
				.description 		= "Memory subsystem rail; Current", 
				.resources.selector	= 15
		},
		{ 	.name 				= "vddq:volt",
				.units 				= "uV",
				.description 		= "Memory subsystem rail; Voltage", 
				.resources.selector	= 16
		}
};

static int num_events		= 0;
static int is_initialized	= 0;

/***************************************************************************/
/******  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT *******/
/***************************************************************************/

#if 0
From Intel docs, power readings are exported via sysfs at
/sys/class/micras/power

typedeftruct mr_rsp_pws {	/* Power status */
  uint32_t	prr;				/* Current reading, in uW */
  uint8_t p_val;                /* Valid bits, power */
} MrRspPws;

typedef struct mr_rsp_vrr {	/* Voltage regulator status */
  uint32_t pwr;                 /* Power reading, in uW */
  uint32_t cur;                 /* Current, in uA */
  uint32_t volt;                /* Voltage, in uV */
  uint8_t p_val;                /* Valid bits, power */
  uint8_t c_val;                /* Valid bits, current */
  uint8_t v_val;                /* Valid bits, voltage */
} MrRspVrr;


I am assuming for the purposes of this component that only
the readings are exported.
typedef struct mr_rsp_power {
  MrRspPws tot0;                /* Total power, win 0 */
  MrRspPws tot1;                /* Total power, win 1 */
  MrRspPws	pcie;				/* PCI-E connector power */
  MrRspPws	inst;				/* Instantaneous power */
  MrRspPws	imax;				/* Max Instantaneous power */
  MrRspPws	c2x3;				/* 2x3 connector power */
  MrRspPws	c2x4;				/* 2x4 connector power */
  MrRspVrr	vccp;				/* Core rail */
  MrRspVrr	vddg;				/* Uncore rail */
  MrRspVrr	vddq;				/* Memory subsystem rail */
} MrRspPower;

#endif
static int 
read_sysfs_file( long long* counts) 
{
		FILE* fp = NULL;
		int i;
		int retval = 1;
		fp = fopen( "/sys/class/micras/power", "r" );
		if (!fp)
		    return 0;

		for (i=0; i < MICPOWER_MAX_COUNTERS-9; i++) {
				retval&= fscanf(fp, "%lld", &counts[i]);
		}
		for (i=MICPOWER_MAX_COUNTERS-9; i < MICPOWER_MAX_COUNTERS; i+=3) {
				retval&= fscanf(fp, "%lld %lld %lld", &counts[i], &counts[i+1], &counts[i+2] );
		}

		fclose(fp);
		return retval;
}

/*****************************************************************************
 *******************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *************
 *****************************************************************************/

/*
 * This is called whenever a thread is initialized
 */
static int 
_micpower_init_thread( hwd_context_t *ctx )
{
		( void ) ctx;
		return PAPI_OK;
}



/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
static int 
_micpower_init_component( int cidx )
{
    int retval = PAPI_OK;
		if ( is_initialized )
				return (PAPI_OK );

		is_initialized = 1;

		/* Check that /sys/class/micras/power is readable */
		if ( 0 != access( "/sys/class/micras/power", R_OK ) ) {
				strncpy(_micpower_vector.cmp_info.disabled_reason,
								"Cannot read /sys/class/micras/power",PAPI_MAX_STR_LEN);
                retval = PAPI_ECMP;
                goto fn_fail;
		}


		/* Export the total number of events available */
		num_events =
				_micpower_vector.cmp_info.num_native_events = MICPOWER_NUMBER_OF_NATIVE_EVENTS;

		/* Export the component id */
		_micpower_vector.cmp_info.CmpIdx = cidx;

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
_micpower_init_control_state( hwd_control_state_t * ctl)
{
		int retval = 0;
		MICPOWER_control_state_t *micpower_ctl = (MICPOWER_control_state_t *) ctl;

		retval = read_sysfs_file(micpower_ctl->counts);

		/* Set last access time for caching results */
		micpower_ctl->lastupdate = PAPI_get_real_usec();

		return (retval)?PAPI_OK:PAPI_ESYS;
}

static int 
_micpower_start( hwd_context_t *ctx, hwd_control_state_t *ctl)
{
		( void ) ctx;
		( void ) ctl;

		return PAPI_OK;
}

static int 
_micpower_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
				long long ** events, int flags)
{
		(void) flags;
		(void) ctx;
		int retval = 1;

		MICPOWER_control_state_t* control = (MICPOWER_control_state_t*) ctl;
		long long now = PAPI_get_real_usec();

		/* Only read the values from the kernel if enough time has passed */
		/* since the last read.  Otherwise return cached values.          */

		if ( now - control->lastupdate > REFRESH_LAT ) {
				retval = read_sysfs_file(control->counts);
				control->lastupdate = now;
		}

		/* Pass back a pointer to our results */
		*events = control->counts;

		return (retval)?PAPI_OK:PAPI_ESYS;
}

static int 
_micpower_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
		(void) ctx;
		int retval = 1;
		long long now = PAPI_get_real_usec();
		/* read values */
		MICPOWER_control_state_t* control = (MICPOWER_control_state_t*) ctl;

		if ( now - control->lastupdate > REFRESH_LAT ) {
				retval = read_sysfs_file(control->counts);
				control->lastupdate = now;
		}
		return (retval)?PAPI_OK:PAPI_ESYS;
}

/* Shutdown a thread */
static int 
_micpower_shutdown_thread( hwd_context_t * ctx )
{
		( void ) ctx;
		return PAPI_OK;
}


/*
 * Clean up what was setup in  micpower_init_component().
 */
static int 
_micpower_shutdown_component( ) 
{
		if ( is_initialized ) {
				is_initialized = 0;
				num_events = 0;
		}
		return PAPI_OK;
}


/* This function sets various options in the component
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
static int 
_micpower_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
		( void ) ctx;
		( void ) code;
		( void ) option;

		return PAPI_OK;
}


static int 
_micpower_update_control_state(	hwd_control_state_t *ptr,
				NativeInfo_t * native, int count,
				hwd_context_t * ctx )
{
		int i, index;
		( void ) ctx;
		( void ) ptr;

		for ( i = 0; i < count; i++ ) {
				index = native[i].ni_event&PAPI_NATIVE_AND_MASK;
				native[i].ni_position = _micpower_native_events[index].resources.selector - 1;
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
_micpower_set_domain( hwd_control_state_t * cntl, int domain )
{
		( void ) cntl;

		if ( PAPI_DOM_ALL != domain )
		    return PAPI_EINVAL;

		return PAPI_OK;
}


static int 
_micpower_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
		( void ) ctx;
		( void ) ctl;

		return PAPI_OK;
}


/*
 * Native Event functions
 */
static int 
_micpower_ntv_enum_events( unsigned int *EventCode, int modifier )
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

						index = *EventCode&PAPI_NATIVE_AND_MASK;

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
_micpower_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
		int index = EventCode&PAPI_NATIVE_AND_MASK;

		if ( index >= 0 && index < num_events ) {
				strncpy( name, _micpower_native_events[index].name, len );
				return PAPI_OK;
		}
		return PAPI_ENOEVNT;
}

/*
 *
 */
static int 
_micpower_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
		int index = EventCode&PAPI_NATIVE_AND_MASK;

		if ( index >= 0 && index < num_events ) {
				strncpy( name, _micpower_native_events[index].description, len );
				return PAPI_OK;
		}
		return PAPI_ENOEVNT;
}

static int
_micpower_ntv_code_to_info(unsigned int EventCode, PAPI_event_info_t *info) 
{

		int index = EventCode&PAPI_NATIVE_AND_MASK;

		if ( ( index < 0) || (index >= num_events )) return PAPI_ENOEVNT;

		strncpy( info->symbol, _micpower_native_events[index].name, sizeof(info->symbol));
		strncpy( info->long_descr, _micpower_native_events[index].description, sizeof(info->long_descr));
		strncpy( info->units, _micpower_native_events[index].units, sizeof(info->units));
		info->units[sizeof(info->units)-1] = '\0';

		return PAPI_OK;
}



/*
 *
 */
papi_vector_t _micpower_vector = {
		.cmp_info = {
				/* default component information (unspecified values are initialized to 0) */
				.name = "micpower",
				.short_name = "micpower",
				.description = "Component for reading power on Intel Xeon Phi (MIC)",
				.version = "5.1",
				.num_mpx_cntrs = MICPOWER_NUMBER_OF_NATIVE_EVENTS,
				.num_cntrs = MICPOWER_NUMBER_OF_NATIVE_EVENTS,
				.default_domain = PAPI_DOM_ALL,
				.available_domains = PAPI_DOM_ALL,
				.default_granularity = PAPI_GRN_SYS,
				.available_granularities = PAPI_GRN_SYS,
				.hardware_intr_sig = PAPI_INT_SIGNAL,

				/* component specific cmp_info initializations */
				.fast_real_timer = 0,
				.fast_virtual_timer = 0,
				.attach = 0,
				.attach_must_ptrace = 0,
		}
		,

				/* sizes of framework-opaque component-private structures */
				.size = {
						.context = sizeof ( MICPOWER_context_t ),
						.control_state = sizeof ( MICPOWER_control_state_t ),
						.reg_value = sizeof ( MICPOWER_register_t ),
						.reg_alloc = sizeof ( MICPOWER_reg_alloc_t ),
				}
		,
				/* function pointers in this component */
				.init_thread =          _micpower_init_thread,
				.init_component =       _micpower_init_component,
				.init_control_state =   _micpower_init_control_state,
				.start =                _micpower_start,
				.stop =                 _micpower_stop,
				.read =                 _micpower_read,
				.shutdown_thread =      _micpower_shutdown_thread,
				.shutdown_component =   _micpower_shutdown_component,
				.ctl =                  _micpower_ctl,

				.update_control_state = _micpower_update_control_state,
				.set_domain =           _micpower_set_domain,
				.reset =                _micpower_reset,

				.ntv_enum_events =      _micpower_ntv_enum_events,
				.ntv_code_to_name =     _micpower_ntv_code_to_name,
				.ntv_code_to_descr =    _micpower_ntv_code_to_descr,
				.ntv_code_to_info =     _micpower_ntv_code_to_info,
};
