/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-emon.c
 * @author  Heike Jagode
 *          jagode@eecs.utk.edu
 * BGPM / emon component 
 * 
 * @brief
 *  This file has the source code for a component that enables PAPI-C to 
 *  access hardware power data for BG/Q through the EMON interface.
 */

#include <stdint.h>
#include <string.h>
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "extras.h"

#define EMON_DEFINE_GLOBALS
#include <hwi/include/bqc/A2_inlines.h>
#include <spi/include/emon/emon.h> // the emon library header file (no linking required)

#define EMON_MAX_COUNTERS 8
#define EMON_TOTAL_EVENTS 8

#ifndef DEBUG
#define EMONDBG( fmt, args...) do {} while(0)
#else
#define EMONDBG( fmt, args... ) do { printf("%s:%d\t"fmt, __func__, __LINE__, ##args); } while(0)
#endif

/* Stores private information for each event */
typedef struct EMON_register
{
	unsigned int selector;
	/* Signifies which counter slot is being used */
	/* Indexed from 1 as 0 has a special meaning  */
} EMON_register_t;

/** This structure is used to build the table of events  */
/*   The contents of this structure will vary based on   */
/*   your component, however having name and description */
/*   fields are probably useful.                         */
typedef struct EMON_native_event_entry
{
	EMON_register_t resources;	        /**< Per counter resources       */
	char *name;	    /**< Name of the counter         */
	char *description; /**< Description of the counter  */
	int return_type;
} EMON_native_event_entry_t;


/* Used when doing register allocation */
typedef struct EMON_reg_alloc
{
	EMON_register_t ra_bits;
} EMON_reg_alloc_t;

typedef struct EMON_overflow
{
  	int threshold;
	int EventIndex;
} EMON_overflow_t;

/* Holds control flags */
typedef struct EMON_control_state
{
    int count;
    long long counters[EMON_MAX_COUNTERS];
    int being_measured[EMON_MAX_COUNTERS];
    long long last_update;
} EMON_control_state_t;

/* Holds per-thread information */
typedef struct EMON_context
{
	EMON_control_state_t state;
} EMON_context_t;

/* Declare our vector in advance */
papi_vector_t _emon2_vector;

static void _check_EMON_error( char* emon2func, int err )
{
	( void ) emon2func;
	if ( err < 0 ) {
		printf( "Error: EMON API function '%s' returned %d.\n",
                	emon2func, err );
	}
}


/** This table contains the native events 
  * So with the EMON interface, we get every domain at a time.
  */
static EMON_native_event_entry_t EMON_native_table[] =
{
	{ 
		.name = "DOMAIN1",
		.description = "Chip core",
		.resources.selector = 1,
		.return_type = PAPI_DATATYPE_FP64,
	},
	{
		.name = "DOMAIN2",
		.description = "Chip Memory Interface and Dramm",
		.resources.selector = 2,
		.return_type = PAPI_DATATYPE_FP64,
	},
	{
		.name = "DOMAIN3",
		.description = "Optics",
		.resources.selector = 3,
		.return_type = PAPI_DATATYPE_FP64,
	},
	{
		.name = "DOMAIN4",
		.description = "Optics + PCIExpress",
		.resources.selector = 4,
		.return_type = PAPI_DATATYPE_FP64,
	},
	{
		.name = "DOMAIN6",
		.description = "HSS Network and Link Chip",
		.resources.selector = 5,
		.return_type = PAPI_DATATYPE_FP64,
	},
	{
		.name = "DOMAIN8",
		.description = "Link Chip Core",
		.resources.selector = 6,
		.return_type = PAPI_DATATYPE_FP64,
	},
	{
		.name = "DOMAIN7",
		.description = "Chip SRAM",
		.resources.selector = 7,
		.return_type = PAPI_DATATYPE_FP64,
	},
	{   .name="EMON_DOMAIN_ALL",
		.description = "Measures power on all domains.",
		.resources.selector = 8,
		.return_type = PAPI_DATATYPE_FP64,
	},
};




/*****************************************************************************
 *******************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *************
 *****************************************************************************/

/*
 * This is called whenever a thread is initialized
 */
int
EMON_init_thread( hwd_context_t * ctx )
{
	EMONDBG( "EMON_init_thread\n" );
	
	( void ) ctx;
	return PAPI_OK;
}


/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
int
EMON_init_component( int cidx )
{ 
	int ret = 0;
	_emon2_vector.cmp_info.CmpIdx = cidx;
	EMONDBG( "EMON_init_component cidx = %d\n", cidx );
	/* Setup connection with the fpga:
	 * NOTE: any other threads attempting to call into the EMON API 
	 *	    will be turned away.  */
	ret = EMON_SetupPowerMeasurement();
	_check_EMON_error("EMON_SetupPowerMeasurement", ret );

	_emon2_vector.cmp_info.num_native_events = EMON_TOTAL_EVENTS;

	_emon2_vector.cmp_info.num_cntrs = EMON_TOTAL_EVENTS;
	_emon2_vector.cmp_info.num_mpx_cntrs = EMON_TOTAL_EVENTS;

	
	return ( PAPI_OK );
}


/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
int
EMON_init_control_state( hwd_control_state_t * ptr )
{
	EMONDBG( "EMON_init_control_state\n" );

	EMON_control_state_t * this_state = ( EMON_control_state_t * ) ptr;
	memset( this_state, 0, sizeof ( EMON_control_state_t ) );
	
	return PAPI_OK;
}

static int 
_emon_accessor( EMON_control_state_t * this_state ) 
{
	union {
		long long ll;
		double fp;
	} return_value;
	return_value.fp = -1;

	double volts[14],amps[14];
	double cpu 		= 0;
	double dram 		= 0;
	double link_chip	= 0;
	double network 		= 0;
	double optics 		= 0;
	double pci		= 0;
	double sram		= 0;
	unsigned k_const; 
    
	EMONDBG( "_emon_accessor, enter this_state = %x\n", this_state);
	return_value.fp = EMON_GetPower_impl( volts, amps );
	EMONDBG("_emon_accessor, after EMON_GetPower %lf \n", return_value.fp);
	if ( -1 == return_value.fp ) {
		PAPIERROR("EMON_GetPower() failed!\n");
		return ( PAPI_ESYS );
	}

	this_state->counters[7] = return_value.ll;

/*  We just stuff everything in counters, there is no extra overhead here */
	k_const 	= domain_info[0].k_const; /* Chip Core Voltage */
	cpu += volts[0] * amps[0] * k_const;
	cpu += volts[1] * amps[1] * k_const;

	k_const 	= domain_info[1].k_const; /* Chip Core Voltage */
	dram += volts[2] * amps[2] * k_const;
	dram += volts[3] * amps[3] * k_const;

	k_const 	= domain_info[2].k_const; /* Chip Core Voltage */
	optics += volts[4] * amps[4] * k_const;
	optics += volts[5] * amps[5] * k_const;

	k_const 	= domain_info[3].k_const; /* Chip Core Voltage */
	pci += volts[6] * amps[6] * k_const;
	pci += volts[7] * amps[7] * k_const;

	k_const 	= domain_info[4].k_const; /* Chip Core Voltage */
	network += volts[8] * amps[8] * k_const;
	network += volts[9] * amps[9] * k_const;

	k_const 	= domain_info[5].k_const; /* Chip Core Voltage */
	link_chip += volts[10] * amps[10] * k_const;
	link_chip += volts[11] * amps[11] * k_const;

	k_const 	= domain_info[6].k_const; /* Chip Core Voltage */
	sram += volts[12] * amps[12] * k_const;
	sram += volts[13] * amps[13] * k_const;

	this_state->counters[0] = *(long long*)&cpu;
	this_state->counters[1] = *(long long*)&dram;
	this_state->counters[2] = *(long long*)&optics;
	this_state->counters[3] = *(long long*)&pci;
	this_state->counters[4] = *(long long*)&link_chip;
	this_state->counters[5] = *(long long*)&network;
	this_state->counters[6] = *(long long*)&sram;

	EMONDBG("CPU = %lf\n", *(double*)&this_state->counters[0]);
	EMONDBG("DRAM = %lf\n", *(double*)&this_state->counters[1]);
	EMONDBG("Optics = %lf\n", *(double*)&this_state->counters[2]);
	EMONDBG("PCI = %lf\n", *(double*)&this_state->counters[3]);
	EMONDBG("Link Chip = %lf\n", *(double*)&this_state->counters[4]);
	EMONDBG("Network = %lf\n", *(double*)&this_state->counters[5]);
	EMONDBG("SRAM = %lf\n", *(double*)&this_state->counters[6]);
	EMONDBG("TOTAL = %lf\n", *(double*)&this_state->counters[7] );

	return ( PAPI_OK );
}

/*
 *
 */
int
EMON_start( hwd_context_t * ctx, hwd_control_state_t * ptr )
{
	EMONDBG( "EMON_start\n" );
	( void ) ctx;
	( void ) ptr;
	/*EMON_control_state_t * this_state = ( EMON_control_state_t * ) ptr;*/

	return ( PAPI_OK );
}


/*
 *
 */
int
EMON_stop( hwd_context_t * ctx, hwd_control_state_t * ptr )
{
	EMONDBG( "EMON_stop\n" );
	( void ) ctx;
	EMON_control_state_t * this_state = ( EMON_control_state_t * ) ptr;

	return _emon_accessor( this_state );
}


/*
 *
 */
int
EMON_read( hwd_context_t * ctx, hwd_control_state_t * ptr,
		   long long ** events, int flags )
{
	EMONDBG( "EMON_read\n" );
	( void ) ctx;
	( void ) flags;
	int ret;
	EMON_control_state_t * this_state = ( EMON_control_state_t * ) ptr;

	ret = _emon_accessor( this_state );
	*events = this_state->counters;
	return ret;
}


/*
 *
 */
int
EMON_shutdown_thread( hwd_context_t * ctx )
{
	EMONDBG( "EMON_shutdown_thread\n" );
	
	( void ) ctx;
	return ( PAPI_OK );
}

int
EMON_shutdown_component( void )
{
	EMONDBG( "EMON_shutdown_component\n" );
	
	return ( PAPI_OK );
}

/* This function sets various options in the component
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
int
EMON_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
	EMONDBG( "EMON_ctl\n" );
	
	( void ) ctx;
	( void ) code;
	( void ) option;
	return ( PAPI_OK );
}


/*
 * PAPI Cleanup Eventset
 */
int
EMON_cleanup_eventset( hwd_control_state_t * ctrl )
{
	EMONDBG( "EMON_cleanup_eventset\n" );
	
	EMON_control_state_t * this_state = ( EMON_control_state_t * ) ctrl;
	( void ) this_state;
 	
	return ( PAPI_OK );
}


/*
 *
 */
int
EMON_update_control_state( hwd_control_state_t * ptr,
						   NativeInfo_t * native, int count,
						   hwd_context_t * ctx )
{
	EMONDBG( "EMON_update_control_state: count = %d\n", count );

	( void ) ctx;
	int index, i;
	EMON_control_state_t * this_state = ( EMON_control_state_t * ) ptr;
	( void ) ptr;


	
	// otherwise, add the events to the eventset
	for ( i = 0; i < count; i++ ) {
		index = ( native[i].ni_event ) ;
		
		native[i].ni_position = i;
		
		EMONDBG("EMON_update_control_state: ADD event: i = %d, index = %d\n", i, index );
	}
	
	// store how many events we added to an EventSet
	this_state->count = count;
	
	return ( PAPI_OK );
}


/*
 * As a system wide count, PAPI_DOM_ALL is all we support
 */
int
EMON_set_domain( hwd_control_state_t * cntrl, int domain )
{
	EMONDBG( "EMON_set_domain\n" );
    ( void ) cntrl;

	if ( PAPI_DOM_ALL != domain )
		return ( PAPI_EINVAL );

	return ( PAPI_OK );
}


/*
 *
 */
int
EMON_reset( hwd_context_t * ctx, hwd_control_state_t * ptr )
{
	EMONDBG( "EMON_reset\n" );
	( void ) ctx;
	int retval;
	EMON_control_state_t * this_state = ( EMON_control_state_t * ) ptr;
	( void ) this_state;
	( void ) retval;

	memset( this_state->counters, 0x0, sizeof(long long) * EMON_MAX_COUNTERS);

	return ( PAPI_OK );
}


/*
 * Native Event functions
 */
int
EMON_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	EMONDBG( "EMON_ntv_enum_events, EventCode = %#x\n", *EventCode );

	switch ( modifier ) {
	case PAPI_ENUM_FIRST:
		*EventCode = 0;

		return ( PAPI_OK );
		break;

	case PAPI_ENUM_EVENTS:
	{
		int index = ( *EventCode );

		if ( index < EMON_TOTAL_EVENTS ) {
			*EventCode = *EventCode + 1;
			return ( PAPI_OK );
		} else {
			return ( PAPI_ENOEVNT );
		}

		break;
	}
	default:
		return ( PAPI_EINVAL );
	}
	return ( PAPI_EINVAL );
}

/*
 *
 */
int
EMON_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
	EMONDBG( "EMON_ntv_code_to_name\n" );
	int index;
	( void ) name;
	( void ) len;
	
	index = ( EventCode );

	if ( index >= EMON_TOTAL_EVENTS || index < 0 ) {
		return PAPI_ENOEVNT;
	}

	strncpy( name, EMON_native_table[index].name, len );
	return ( PAPI_OK );
}

/*
 * 
 */
int
EMON_ntv_name_to_code( const char *name, unsigned int *code )
{
	int index;

	for ( index = 0; index < EMON_TOTAL_EVENTS; index++ ) {
		if ( 0 == strcmp( name, EMON_native_table[index].name ) ) {
			*code = index;
		}
	}
	return ( PAPI_OK );
}

int
EMON_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
	EMONDBG( "EMON_ntv_code_to_descr\n" );
	int index;
	( void ) name;
	( void ) len;
	
	index = ( EventCode ) ;

	if ( index >= EMON_TOTAL_EVENTS || index < 0 ) {
		return PAPI_ENOEVNT;
	}
	strncpy( name, EMON_native_table[index].description, len );

	return ( PAPI_OK );
}


/*
 *
 */
int
EMON_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
	EMONDBG( "EMON_ntv_code_to_bits\n" );
	( void ) EventCode;
	( void ) bits;
	return ( PAPI_OK );
}

int
EMON_ntv_code_to_info(unsigned int EventCode, PAPI_event_info_t *info) 
{

  int index = EventCode;

  if ( ( index < 0) || (index >= EMON_TOTAL_EVENTS )) return PAPI_ENOEVNT;

  strncpy( info->symbol, EMON_native_table[index].name, 
	   sizeof(info->symbol));

  strncpy( info->long_descr, EMON_native_table[index].description, 
	   sizeof(info->symbol));

  //strncpy( info->units, rapl_native_events[index].units, 
	   //sizeof(info->units));

  info->data_type = EMON_native_table[index].return_type;

  return PAPI_OK;
}

/*
 *
 */
papi_vector_t _emon_vector = {
	.cmp_info = {
				 /* default component information (unspecified values are initialized to 0) */
				 .name = "EMON",
				 .short_name = "EMON",
				 .description = "Blue Gene/Q EMON component",
				 .num_native_events = EMON_MAX_COUNTERS,
				 .num_cntrs = EMON_MAX_COUNTERS,
				 .num_mpx_cntrs = EMON_MAX_COUNTERS,
				 .default_domain = PAPI_DOM_ALL,
				 .available_domains = PAPI_DOM_ALL,
				 .default_granularity = PAPI_GRN_SYS,
				 .available_granularities = PAPI_GRN_SYS,
		
				 .hardware_intr_sig = PAPI_INT_SIGNAL,
				 .hardware_intr = 1,
		
				 .kernel_multiplex = 0,

				 /* component specific cmp_info initializations */
				 .fast_real_timer = 0,
				 .fast_virtual_timer = 0,
				 .attach = 0,
				 .attach_must_ptrace = 0,
				 }
	,

	/* sizes of framework-opaque component-private structures */
	.size = {
			 .context = sizeof ( EMON_context_t ),
			 .control_state = sizeof ( EMON_control_state_t ),
			 .reg_value = sizeof ( EMON_register_t ),
			 .reg_alloc = sizeof ( EMON_reg_alloc_t ),
			 }
	,
	/* function pointers in this component */
	.init_thread =          EMON_init_thread,
	.init_component =       EMON_init_component,
	.init_control_state =   EMON_init_control_state,
	.start =                EMON_start,
	.stop =                 EMON_stop,
	.read =                 EMON_read,
	.shutdown_thread =      EMON_shutdown_thread,
	.shutdown_component =   EMON_shutdown_component,
	.cleanup_eventset =     EMON_cleanup_eventset,
	.ctl =                  EMON_ctl,

	.update_control_state = EMON_update_control_state,
	.set_domain =           EMON_set_domain,
	.reset =                EMON_reset,

	.ntv_enum_events =      EMON_ntv_enum_events,
	.ntv_code_to_name =     EMON_ntv_code_to_name,
	.ntv_code_to_descr =    EMON_ntv_code_to_descr,
	.ntv_code_to_bits =     EMON_ntv_code_to_bits,
	.ntv_code_to_info = EMON_ntv_code_to_info,
};
