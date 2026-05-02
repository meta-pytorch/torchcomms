/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    coretemp_freebsd.c
 * @author  Joachim Protze
 *          joachim.protze@zih.tu-dresden.de
 * @author  Vince Weaver
 *          vweaver1@eecs.utk.edu
 * @author  Harald Servat
 *          harald.servat@gmail.com
 *
 * @ingroup papi_components
 *
 * @brief
 *   This component is intended to access CPU On-Die Thermal Sensors in 
 *   the Intel Core architecture in a FreeBSD machine using the coretemp.ko
 *   kernel module.
 */

#include <sys/types.h>
#include <sys/resource.h>
#include <sys/sysctl.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#define CORETEMP_MAX_COUNTERS 32 /* Can we tune this dynamically? */
#define TRUE  (1==1)
#define FALSE (1!=1)
#define UNREFERENCED(x) (void)x

/* Structure that stores private information for each event */
typedef struct coretemp_register
{
	int mib[4];
	/* Access to registers through these MIBs + sysctl (3) call */
	
	unsigned int selector;
	/**< Signifies which counter slot is being used */
	/**< Indexed from 1 as 0 has a special meaning  */
} coretemp_register_t;

/** This structure is used to build the table of events */
typedef struct coretemp_native_event_entry
{
	coretemp_register_t resources;            /**< Per counter resources       */
	char name[PAPI_MAX_STR_LEN];             /**< Name of the counter         */
	char description[PAPI_MAX_STR_LEN];      /**< Description of the counter  */
} coretemp_native_event_entry_t;

/* This structure is used when doing register allocation 
   it possibly is not necessary when there are no 
   register constraints */
typedef struct coretemp_reg_alloc
{
	coretemp_register_t ra_bits;
} coretemp_reg_alloc_t;

/* Holds control flags, usually out-of band configuration of the hardware */
typedef struct coretemp_control_state
{
	int added[CORETEMP_MAX_COUNTERS];
	long_long counters[CORETEMP_MAX_COUNTERS];	/**< Copy of counts, used for caching */
} coretemp_control_state_t;

/* Holds per-thread information */
typedef struct coretemp_context
{
	coretemp_control_state_t state;
} coretemp_context_t;

/** This table contains the native events */
static coretemp_native_event_entry_t *coretemp_native_table;

/** number of events in the table*/
static int CORETEMP_NUM_EVENTS = 0;


/********************************************************************/
/* Below are the functions required by the PAPI component interface */
/********************************************************************/

/** This is called whenever a thread is initialized */
int coretemp_init_thread (hwd_context_t * ctx)
{
	int mib[4];
	size_t len;
	UNREFERENCED(ctx);

	( void ) mib; /*unused */
	( void ) len; /*unused */

	SUBDBG("coretemp_init_thread %p...\n", ctx);

#if 0
	/* what does this do?  VMW */

	len = 4;
	if (sysctlnametomib ("dev.coretemp.0.%driver", mib, &len) == -1)
		return PAPI_ECMP;
#endif

	return PAPI_OK;
}


/** Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the
 * PAPI process is initialized (IE PAPI_library_init)
 */
int coretemp_init_component ()
{
	int ret;
	int i;
	int mib[4];
	size_t len;
	char tmp[128];
    int retval = PAPI_OK;

	SUBDBG("coretemp_init_component...\n");

	/* Count the number of cores (counters) that have sensors allocated */
	i = 0;
	CORETEMP_NUM_EVENTS = 0;
	sprintf (tmp, "dev.coretemp.%d.%%driver", i);
	len = 4;
	ret = sysctlnametomib (tmp, mib, &len);
	while (ret != -1)
	{
		CORETEMP_NUM_EVENTS++;
		i++;
		sprintf (tmp, "dev.coretemp.%d.%%driver", i);
		len = 4;
		ret = sysctlnametomib (tmp, mib, &len);
	}

	if (CORETEMP_NUM_EVENTS == 0)
        goto fn_exit;

	/* Allocate memory for the our event table */
	coretemp_native_table = (coretemp_native_event_entry_t *)
		papi_malloc (sizeof (coretemp_native_event_entry_t) * CORETEMP_NUM_EVENTS);
	if (coretemp_native_table == NULL)
	{
		perror( "malloc():Could not get memory for coretemp events table" );
        retval = PAPI_ENOMEM;
        goto fn_fail;
	}

	/* Allocate native events internal structures */
	for (i = 0; i < CORETEMP_NUM_EVENTS; i++)
	{
		/* Event name */
		sprintf (coretemp_native_table[i].name, "CORETEMP_CPU_%d", i);

		/* Event description */
		sprintf (coretemp_native_table[i].description, "CPU On-Die Thermal Sensor #%d", i);

		/* Event extra bits -> save MIB to faster access later */
		sprintf (tmp, "dev.cpu.%d.temperature", i);
		len = 4;
        if (sysctlnametomib (tmp, coretemp_native_table[i].resources.mib, &len) == -1) {
            retval = PAPI_ECMP;
            goto fn_fail;
        }

		coretemp_native_table[i].resources.selector = i+1;
	}

  fn_exit:
    _papi_hwd[cidx]->cmp_info.disabled = retval;
    return retval;
  fn_fail:
    goto fn_exit;
}


/** Setup the counter control structure */
int coretemp_init_control_state (hwd_control_state_t * ctrl)
{
	int i;

	SUBDBG("coretemp_init_control_state... %p\n", ctrl);
	coretemp_control_state_t *c = (coretemp_control_state_t *) ctrl;

	for (i = 0; i < CORETEMP_MAX_COUNTERS; i++)
		c->added[i] = FALSE;

	return PAPI_OK;
}


/** Enumerate Native Events 
   @param EventCode is the event of interest
   @param modifier is one of PAPI_ENUM_FIRST, PAPI_ENUM_EVENTS
*/
int coretemp_ntv_enum_events (unsigned int *EventCode, int modifier)
{

	switch ( modifier )
	{
		/* return EventCode of first event */
		case PAPI_ENUM_FIRST:
		*EventCode = 0;
		return PAPI_OK;
		break;

		/* return EventCode of passed-in Event */
		case PAPI_ENUM_EVENTS:
		{
			int index = *EventCode;

			if ( index < CORETEMP_NUM_EVENTS - 1 )
			{
				*EventCode = *EventCode + 1;
				return PAPI_OK;
			}
			else
				return PAPI_ENOEVNT;
			break;
		}

		default:
			return PAPI_EINVAL;
	}

	return PAPI_EINVAL;
}

/** Takes a native event code and passes back the name 
 @param EventCode is the native event code
 @param name is a pointer for the name to be copied to
 @param len is the size of the string
 */
int coretemp_ntv_code_to_name (unsigned int EventCode, char *name, int len)
{
	int index = EventCode;

	strncpy( name, coretemp_native_table[index].name, len );

	return PAPI_OK;
}

/** Takes a native event code and passes back the event description
 @param EventCode is the native event code
 @param name is a pointer for the description to be copied to
 @param len is the size of the string
 */
int coretemp_ntv_code_to_descr (unsigned int EventCode, char *name, int len)
{
	int index = EventCode;

	strncpy( name, coretemp_native_table[index].description, len );

	return PAPI_OK;
}

/** This takes an event and returns the bits that would be written
    out to the hardware device (this is very much tied to CPU-type support */
int coretemp_ntv_code_to_bits (unsigned int EventCode, hwd_register_t * bits)
{
	UNREFERENCED(EventCode);
	UNREFERENCED(bits);

	return PAPI_OK;
}

/** Triggered by eventset operations like add or remove */
int coretemp_update_control_state( hwd_control_state_t * ptr,
	NativeInfo_t * native, int count, hwd_context_t * ctx )
{
	int i, index;
	coretemp_control_state_t *c = (coretemp_control_state_t *) ptr;
	UNREFERENCED(ctx);

	SUBDBG("coretemp_update_control_state %p %p...\n", ptr, ctx);

	for (i = 0; i < count; i++)
	{
		index = native[i].ni_event;
		native[i].ni_position = coretemp_native_table[index].resources.selector - 1;
		c->added[native[i].ni_position] = TRUE;

		SUBDBG ("\nnative[%i].ni_position = coretemp_native_table[%i].resources.selector-1 = %i;\n",
		  i, index, native[i].ni_position );
	}

	return PAPI_OK;
}

/** Triggered by PAPI_start() */
int coretemp_start (hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
	UNREFERENCED(ctx);
	UNREFERENCED(ctrl);

	SUBDBG( "coretemp_start %p %p...\n", ctx, ctrl );

	/* Nothing to be done */

	return PAPI_OK;
}


/** Triggered by PAPI_stop() */
int coretemp_stop (hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
	UNREFERENCED(ctx);
	UNREFERENCED(ctrl);

	SUBDBG("coretemp_stop %p %p...\n", ctx, ctrl);

	/* Nothing to be done */

	return PAPI_OK;
}


/** Triggered by PAPI_read() */
int coretemp_read (hwd_context_t * ctx, hwd_control_state_t * ctrl,
	long_long ** events, int flags)
{
	int i;
	coretemp_control_state_t *c = (coretemp_control_state_t *) ctrl;
	UNREFERENCED(ctx);
	UNREFERENCED(flags);

	SUBDBG("coretemp_read... %p %d\n", ctx, flags);

	for (i = 0; i < CORETEMP_MAX_COUNTERS; i++)
		if (c->added[i])
		{
			int tmp;
			size_t len = sizeof(tmp);

			if (sysctl (coretemp_native_table[i].resources.mib, 4, &tmp, &len, NULL, 0) == -1)
				c->counters[i] = 0;
			else
				c->counters[i] = tmp/10;
				/* Coretemp module returns temperature in tenths of kelvin 
				   Kelvin are useful to avoid negative values... but will have
				   negative temperatures ??? */
		}

	*events = c->counters;

	return PAPI_OK;
}

/** Triggered by PAPI_write(), but only if the counters are running */
/*    otherwise, the updated state is written to ESI->hw_start      */
int coretemp_write (hwd_context_t * ctx, hwd_control_state_t * ctrl,
	long_long events[] )
{
	UNREFERENCED(ctx);
	UNREFERENCED(events);
	UNREFERENCED(ctrl);

	SUBDBG("coretemp_write... %p %p\n", ctx, ctrl);

	/* These sensor counters cannot be writtn */

	return PAPI_OK;
}


/** Triggered by PAPI_reset */
int coretemp_reset(hwd_context_t * ctx, hwd_control_state_t * ctrl)
{
	UNREFERENCED(ctx);
	UNREFERENCED(ctrl);

	SUBDBG("coretemp_reset ctx=%p ctrl=%p...\n", ctx, ctrl);

	/* These sensors cannot be reseted */

	return PAPI_OK;
}

/** Triggered by PAPI_shutdown() */
int coretemp_shutdown_component (void)
{

	SUBDBG( "coretemp_shutdown_component...\n");

	/* Last chance to clean up */
	papi_free (coretemp_native_table);

	return PAPI_OK;
}



/** This function sets various options in the component
  @param ctx unused
  @param code valid are PAPI_SET_DEFDOM, PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL and PAPI_SET_INHERIT
  @param option unused
 */
int coretemp_ctl (hwd_context_t * ctx, int code, _papi_int_option_t * option)
{
	UNREFERENCED(ctx);
	UNREFERENCED(code);
	UNREFERENCED(option);

	SUBDBG( "coretemp_ctl... %p %d %p\n", ctx, code, option );

	/* FIXME.  This should maybe set up more state, such as which counters are active and */
	/*         counter mappings. */

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
int coretemp_set_domain (hwd_control_state_t * cntrl, int domain)
{
	UNREFERENCED(cntrl);

	SUBDBG ("coretemp_set_domain... %p %d\n", cntrl, domain);

	if (PAPI_DOM_ALL & domain)
	{
		SUBDBG( " PAPI_DOM_ALL \n" );
		return PAPI_OK;
	}
	return PAPI_EINVAL ;

}


/** Vector that points to entry points for our component */
papi_vector_t _coretemp_freebsd_vector = {
	.cmp_info = {
				 /* default component information (unspecified values are initialized to 0) */
				 .name = "coretemp_freebsd",
				 .short_name = "coretemp",
				 .version = "5.0",
				 .num_mpx_cntrs = CORETEMP_MAX_COUNTERS,
				 .num_cntrs = CORETEMP_MAX_COUNTERS,
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
			 .context = sizeof ( coretemp_context_t ),
			 .control_state = sizeof ( coretemp_control_state_t ),
			 .reg_value = sizeof ( coretemp_register_t ),
			 .reg_alloc = sizeof ( coretemp_reg_alloc_t ),
			 }
	,
	/* function pointers in this component */
	.init_thread = coretemp_init_thread,
	.init_component = coretemp_init_component,
	.init_control_state = coretemp_init_control_state,
	.start = coretemp_start,
	.stop = coretemp_stop,
	.read = coretemp_read,
	.write = coretemp_write,
	.shutdown_component = coretemp_shutdown_component,
	.ctl = coretemp_ctl,

	.update_control_state = coretemp_update_control_state,
	.set_domain = coretemp_set_domain,
	.reset = coretemp_reset,

	.ntv_enum_events = coretemp_ntv_enum_events,
	.ntv_code_to_name = coretemp_ntv_code_to_name,
	.ntv_code_to_descr = coretemp_ntv_code_to_descr,
	.ntv_code_to_bits = coretemp_ntv_code_to_bits,
};

