/**
 * @file    example.c
 * @author  Joachim Protze
 *          joachim.protze@zih.tu-dresden.de
 * @author  Vince Weaver
 *          vweaver1@eecs.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *	This is an example component, it demos the component interface
 *  and implements three example counters.
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"    /* defines papi_malloc(), etc. */

/** This driver supports three counters counting at once      */
/*  This is artificially low to allow testing of multiplexing */
#define EXAMPLE_MAX_SIMULTANEOUS_COUNTERS 3
#define EXAMPLE_MAX_MULTIPLEX_COUNTERS 4

// The following macro follows if a string function has an error. It should 
// never happen; but it is necessary to prevent compiler warnings. We print 
// something just in case there is programmer error in invoking the function.
#define HANDLE_STRING_ERROR {fprintf(stderr,"%s:%i unexpected string function error.\n",__FILE__,__LINE__); exit(-1);}

/* Declare our vector in advance */
/* This allows us to modify the component info */
papi_vector_t _example_vector;

/** Structure that stores private information for each event */
typedef struct example_register
{
   unsigned int selector;
		           /**< Signifies which counter slot is being used */
			   /**< Indexed from 1 as 0 has a special meaning  */
} example_register_t;

/** This structure is used to build the table of events  */
/*   The contents of this structure will vary based on   */
/*   your component, however having name and description */
/*   fields are probably useful.                         */
typedef struct example_native_event_entry
{
	example_register_t resources;	    /**< Per counter resources       */
	char name[PAPI_MAX_STR_LEN];	    /**< Name of the counter         */
	char description[PAPI_MAX_STR_LEN]; /**< Description of the counter  */
	int writable;			    /**< Whether counter is writable */
	/* any other counter parameters go here */
} example_native_event_entry_t;

/** This structure is used when doing register allocation 
    it possibly is not necessary when there are no 
    register constraints */
typedef struct example_reg_alloc
{
	example_register_t ra_bits;
} example_reg_alloc_t;

/** Holds control flags.  
 *    There's one of these per event-set.
 *    Use this to hold data specific to the EventSet, either hardware
 *      counter settings or things like counter start values 
 */
typedef struct example_control_state
{
  int num_events;
  int domain;
  int multiplexed;
  int overflow;
  int inherit;
  int which_counter[EXAMPLE_MAX_SIMULTANEOUS_COUNTERS]; 
  long long counter[EXAMPLE_MAX_MULTIPLEX_COUNTERS];   /**< Copy of counts, holds results when stopped */
} example_control_state_t;

/** Holds per-thread information */
typedef struct example_context
{
     long long autoinc_value;
} example_context_t;

/** This table contains the native events */
static example_native_event_entry_t *example_native_table;

/** number of events in the table*/
static int num_events = 0;


/*************************************************************************/
/* Below is the actual "hardware implementation" of our example counters */
/*************************************************************************/

#define EXAMPLE_ZERO_REG             0
#define EXAMPLE_CONSTANT_REG         1
#define EXAMPLE_AUTOINC_REG          2
#define EXAMPLE_GLOBAL_AUTOINC_REG   3

#define EXAMPLE_TOTAL_EVENTS         4

static long long example_global_autoinc_value = 0;

/** Code that resets the hardware.  */
static void
example_hardware_reset( example_context_t *ctx )
{
   /* reset per-thread count */
   ctx->autoinc_value=0;
   /* reset global count */
   example_global_autoinc_value = 0;

}

/** Code that reads event values.                         */
/*   You might replace this with code that accesses       */
/*   hardware or reads values from the operatings system. */
static long long
example_hardware_read( int which_one, example_context_t *ctx )
{
	long long old_value;

	switch ( which_one ) {
	case EXAMPLE_ZERO_REG:
		return 0;
	case EXAMPLE_CONSTANT_REG:
		return 42;
	case EXAMPLE_AUTOINC_REG:
		old_value = ctx->autoinc_value;
		ctx->autoinc_value++;
		return old_value;
	case EXAMPLE_GLOBAL_AUTOINC_REG:
		old_value = example_global_autoinc_value;
		example_global_autoinc_value++;
		return old_value;
	default:
	        fprintf(stderr,"Invalid counter read %#x\n",which_one );
		return -1;
	}

	return 0;
}

/** Code that writes event values.                        */
static int
example_hardware_write( int which_one, 
			example_context_t *ctx,
			long long value)
{

	switch ( which_one ) {
	case EXAMPLE_ZERO_REG:
	case EXAMPLE_CONSTANT_REG:
		return PAPI_OK; /* can't be written */
	case EXAMPLE_AUTOINC_REG:
		ctx->autoinc_value=value;
		return PAPI_OK;
	case EXAMPLE_GLOBAL_AUTOINC_REG:
	        example_global_autoinc_value=value;
		return PAPI_OK;
	default:
		perror( "Invalid counter write" );
		return -1;
	}

	return 0;
}

static int
detect_example(void) {
 
   return PAPI_OK;
}

/********************************************************************/
/* Below are the functions required by the PAPI component interface */
/********************************************************************/


/** Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
static int
_example_init_component( int cidx )
{
    int retval = PAPI_OK;

	SUBDBG( "_example_init_component..." );
   
   /* First, detect that our hardware is available */
   if (detect_example()!=PAPI_OK) {
      int strErr=snprintf(_example_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
         "Example Hardware not present.");
      _example_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
      if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
      retval = PAPI_ENOSUPP;
      goto fn_fail;
   }
   
	/* we know in advance how many events we want                       */
	/* for actual hardware this might have to be determined dynamically */
	num_events = EXAMPLE_TOTAL_EVENTS;

	/* Allocate memory for the our native event table */
	example_native_table =
		( example_native_event_entry_t * )
		papi_calloc( num_events, sizeof(example_native_event_entry_t) );
	if ( example_native_table == NULL ) {
      int strErr=snprintf(_example_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN,
      "Could not allocate %lu bytes of memory for EXAMPLE device structure.", num_events*sizeof(example_native_event_entry_t));
      _example_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;    // force null termination.
      if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
        retval = PAPI_ENOMEM;
        goto fn_fail;
	}

	/* fill in the event table parameters */
	/* for complicated components this will be done dynamically */
	/* or by using an external library                          */

	strcpy( example_native_table[0].name, "EXAMPLE_ZERO" );
	strcpy( example_native_table[0].description,
			"This is an example counter, that always returns 0" );
	example_native_table[0].writable = 0;

	strcpy( example_native_table[1].name, "EXAMPLE_CONSTANT" );
	strcpy( example_native_table[1].description,
			"This is an example counter, that always returns a constant value of 42" );
	example_native_table[1].writable = 0;

	strcpy( example_native_table[2].name, "EXAMPLE_AUTOINC" );
	strcpy( example_native_table[2].description,
			"This is an example counter, that reports a per-thread  auto-incrementing value" );
	example_native_table[2].writable = 1;

	strcpy( example_native_table[3].name, "EXAMPLE_GLOBAL_AUTOINC" );
	strcpy( example_native_table[3].description,
			"This is an example counter, that reports a global auto-incrementing value" );
	example_native_table[3].writable = 1;

	/* Export the total number of events available */
	_example_vector.cmp_info.num_native_events = num_events;

	/* Export the component id */
	_example_vector.cmp_info.CmpIdx = cidx;

  fn_exit:
    _papi_hwd[cidx]->cmp_info.disabled = retval;
    return retval;
  fn_fail:
    goto fn_exit;
}

/** This is called whenever a thread is initialized */
static int
_example_init_thread( hwd_context_t *ctx )
{

        example_context_t *example_context = (example_context_t *)ctx;

        example_context->autoinc_value=0;
   
	SUBDBG( "_example_init_thread %p...", ctx );

	return PAPI_OK;
}



/** Setup a counter control state.
 *   In general a control state holds the hardware info for an
 *   EventSet.
 */

static int
_example_init_control_state( hwd_control_state_t * ctl )
{
   SUBDBG( "example_init_control_state... %p\n", ctl );

   example_control_state_t *example_ctl = ( example_control_state_t * ) ctl;
   memset( example_ctl, 0, sizeof ( example_control_state_t ) );

   return PAPI_OK;
}


/** Triggered by eventset operations like add or remove */
static int
_example_update_control_state( hwd_control_state_t *ctl, 
				    NativeInfo_t *native,
				    int count, 
				    hwd_context_t *ctx )
{
   
   (void) ctx;
   int i, index;

   example_control_state_t *example_ctl = ( example_control_state_t * ) ctl;   

   SUBDBG( "_example_update_control_state %p %p...", ctl, ctx );

   /* if no events, return */
   if (count==0) return PAPI_OK;

   for( i = 0; i < count; i++ ) {
      index = native[i].ni_event;
      
      /* Map counter #i to Measure Event "index" */
      example_ctl->which_counter[i]=index;

      /* We have no constraints on event position, so any event */
      /* can be in any slot.                                    */
      native[i].ni_position = i;
   }

   example_ctl->num_events=count;

   return PAPI_OK;
}

/** Triggered by PAPI_start() */
static int
_example_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

        (void) ctx;
        (void) ctl;

	SUBDBG( "example_start %p %p...", ctx, ctl );

	/* anything that would need to be set at counter start time */

	/* reset counters? */
        /* For hardware that cannot reset counters, store initial        */
        /*     counter state to the ctl and subtract it off at read time */
	 
	/* start the counting ?*/

	return PAPI_OK;
}


/** Triggered by PAPI_stop() */
static int
_example_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

        (void) ctx;
        (void) ctl;

	SUBDBG( "example_stop %p %p...", ctx, ctl );

	/* anything that would need to be done at counter stop time */

	

	return PAPI_OK;
}


/** Triggered by PAPI_read()     */
/*     flags field is never set? */
static int
_example_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
			  long long **events, int flags )
{

   (void) flags;

   example_context_t *example_ctx = (example_context_t *) ctx;
   example_control_state_t *example_ctl = ( example_control_state_t *) ctl;   

   SUBDBG( "example_read... %p %d", ctx, flags );

   int i;

   /* Read counters into expected slot */
   for(i=0;i<example_ctl->num_events;i++) {
      example_ctl->counter[i] =
		example_hardware_read( example_ctl->which_counter[i], 
				       example_ctx );
   }

   /* return pointer to the values we read */
   *events = example_ctl->counter;

   return PAPI_OK;
}

/** Triggered by PAPI_write(), but only if the counters are running */
/*    otherwise, the updated state is written to ESI->hw_start      */
static int
_example_write( hwd_context_t *ctx, hwd_control_state_t *ctl,
			   long long *events )
{

        example_context_t *example_ctx = (example_context_t *) ctx;
        example_control_state_t *example_ctl = ( example_control_state_t *) ctl;   
   
        int i;
   
	SUBDBG( "example_write... %p %p", ctx, ctl );

        /* Write counters into expected slot */
        for(i=0;i<example_ctl->num_events;i++) {
	   example_hardware_write( example_ctl->which_counter[i],
				   example_ctx,
				   events[i] );
	}
   
	return PAPI_OK;
}


/** Triggered by PAPI_reset() but only if the EventSet is currently running */
/*  If the eventset is not currently running, then the saved value in the   */
/*  EventSet is set to zero without calling this routine.                   */
static int
_example_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
        example_context_t *event_ctx = (example_context_t *)ctx;
	(void) ctl;

	SUBDBG( "example_reset ctx=%p ctrl=%p...", ctx, ctl );

	/* Reset the hardware */
	example_hardware_reset( event_ctx );

	return PAPI_OK;
}

/** Triggered by PAPI_shutdown() */
static int
_example_shutdown_component(void)
{

	SUBDBG( "example_shutdown_component..." );

        /* Free anything we allocated */
   
	papi_free(example_native_table);

	return PAPI_OK;
}

/** Called at thread shutdown */
static int
_example_shutdown_thread( hwd_context_t *ctx )
{

        (void) ctx;

	SUBDBG( "example_shutdown_thread... %p", ctx );

	/* Last chance to clean up thread */

	return PAPI_OK;
}



/** This function sets various options in the component
  @param[in] ctx -- hardware context
  @param[in] code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, 
                        PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
  @param[in] option -- options to be set
 */
static int
_example_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{

        (void) ctx;
	(void) code;
	(void) option;

	SUBDBG( "example_ctl..." );

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
static int
_example_set_domain( hwd_control_state_t * cntrl, int domain )
{
        (void) cntrl;

	int found = 0;
	SUBDBG( "example_set_domain..." );

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
	if ( !found )
		return ( PAPI_EINVAL );

	return PAPI_OK;
}


/**************************************************************/
/* Naming functions, used to translate event numbers to names */
/**************************************************************/


/** Enumerate Native Events
 *   @param EventCode is the event of interest
 *   @param modifier is one of PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS
 *  If your component has attribute masks then these need to
 *   be handled here as well.
 */
static int
_example_ntv_enum_events( unsigned int *EventCode, int modifier )
{
  int index;


  switch ( modifier ) {

		/* return EventCode of first event */
	case PAPI_ENUM_FIRST:
	   /* return the first event that we support */

	   *EventCode = 0;
	   return PAPI_OK;

		/* return EventCode of next available event */
	case PAPI_ENUM_EVENTS:
	   index = *EventCode;

	   /* Make sure we have at least 1 more event after us */
	   if ( index < num_events - 1 ) {

	      /* This assumes a non-sparse mapping of the events */
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

/** Takes a native event code and passes back the name 
 * @param EventCode is the native event code
 * @param name is a pointer for the name to be copied to
 * @param len is the size of the name string
 */
static int
_example_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
  int index;

  index = EventCode;

  /* Make sure we are in range */
  if (index >= 0 && index < num_events) {
     strncpy( name, example_native_table[index].name, len );  
     return PAPI_OK;
  }
   
  return PAPI_ENOEVNT;
}

/** Takes a native event code and passes back the event description
 * @param EventCode is the native event code
 * @param descr is a pointer for the description to be copied to
 * @param len is the size of the descr string
 */
static int
_example_ntv_code_to_descr( unsigned int EventCode, char *descr, int len )
{
  int index;
  index = EventCode;

  /* make sure event is in range */
  if (index >= 0 && index < num_events) {
     strncpy( descr, example_native_table[index].description, len );
     return PAPI_OK;
  }
  
  return PAPI_ENOEVNT;
}

/** Vector that points to entry points for our component */
papi_vector_t _example_vector = {
	.cmp_info = {
		/* default component information */
		/* (unspecified values are initialized to 0) */
                /* we explicitly set them to zero in this example */
                /* to show what settings are available            */

		.name = "example",
		.short_name = "example",
		.description = "A simple example component",
		.version = "1.15",
		.support_version = "n/a",
		.kernel_version = "n/a",
		.num_cntrs =               EXAMPLE_MAX_SIMULTANEOUS_COUNTERS, 
		.num_mpx_cntrs =           EXAMPLE_MAX_SIMULTANEOUS_COUNTERS,
		.default_domain =          PAPI_DOM_USER,
		.available_domains =       PAPI_DOM_USER,
		.default_granularity =     PAPI_GRN_THR,
		.available_granularities = PAPI_GRN_THR,
		.hardware_intr_sig =       PAPI_INT_SIGNAL,

		/* component specific cmp_info initializations */
	},

	/* sizes of framework-opaque component-private structures */
	.size = {
	        /* once per thread */
		.context = sizeof ( example_context_t ),
	        /* once per eventset */
		.control_state = sizeof ( example_control_state_t ),
	        /* ?? */
		.reg_value = sizeof ( example_register_t ),
	        /* ?? */
		.reg_alloc = sizeof ( example_reg_alloc_t ),
	},

	/* function pointers */
        /* by default they are set to NULL */
   
	/* Used for general PAPI interactions */
	.start =                _example_start,
	.stop =                 _example_stop,
	.read =                 _example_read,
	.reset =                _example_reset,	
	.write =                _example_write,
	.init_component =       _example_init_component,	
	.init_thread =          _example_init_thread,
	.init_control_state =   _example_init_control_state,
	.update_control_state = _example_update_control_state,	
	.ctl =                  _example_ctl,	
	.shutdown_thread =      _example_shutdown_thread,
	.shutdown_component =   _example_shutdown_component,
	.set_domain =           _example_set_domain,
	/* .cleanup_eventset =     NULL, */
	/* called in add_native_events() */
	/* .allocate_registers =   NULL, */

	/* Used for overflow/profiling */
	/* .dispatch_timer =       NULL, */
	/* .get_overflow_address = NULL, */
	/* .stop_profiling =       NULL, */
	/* .set_overflow =         NULL, */
	/* .set_profile =          NULL, */

	/* ??? */
	/* .user =                 NULL, */

	/* Name Mapping Functions */
	.ntv_enum_events =   _example_ntv_enum_events,
	.ntv_code_to_name =  _example_ntv_code_to_name,
	.ntv_code_to_descr = _example_ntv_code_to_descr,
        /* if .ntv_name_to_code not available, PAPI emulates  */
        /* it by enumerating all events and looking manually  */
   	.ntv_name_to_code  = NULL,

   
	/* These are only used by _papi_hwi_get_native_event_info() */
	/* Which currently only uses the info for printing native   */
	/* event info, not for any sort of internal use.            */
	/* .ntv_code_to_bits =  NULL, */

};

