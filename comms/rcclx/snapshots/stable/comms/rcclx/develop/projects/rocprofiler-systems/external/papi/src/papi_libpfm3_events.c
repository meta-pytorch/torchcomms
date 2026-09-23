/*
* File:    papi_libpfm3_events.c
* Author:  Dan Terpstra: blantantly extracted from Phil's perfmon.c
*          mucci@cs.utk.edu
*
*/

#include <ctype.h>
#include <string.h>
#include <errno.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#include "perfmon/perfmon.h"
#include "perfmon/pfmlib.h"

#include "papi_libpfm_events.h"

/* Native events consist of a flag field, an event field, and a unit mask field.
 * These variables define the characteristics of the event and unit mask fields.
 */
unsigned int PAPI_NATIVE_EVENT_AND_MASK = 0x000003ff;
unsigned int PAPI_NATIVE_EVENT_SHIFT = 0;
unsigned int PAPI_NATIVE_UMASK_AND_MASK = 0x03fffc00;
unsigned int PAPI_NATIVE_UMASK_MAX = 16;
unsigned int PAPI_NATIVE_UMASK_SHIFT = 10;

/* Globals */
int num_native_events=0;


/* NOTE: PAPI stores umask info in a variable sized (16 bit?) bitfield.
    Perfmon2 stores umask info in a large (48 element?) array of values.
    Native event encodings for perfmon2 contain array indices
    encoded as bits in this bitfield. These indices must be converted
    into a umask value before programming the counters. For Perfmon,
    this is done by converting back to an array of values; for 
    perfctr, it must be done by looking up the values.
*/

/* This routine is used to step through all possible combinations of umask
    values. It assumes that mask contains a valid combination of array indices
    for this event. */
static inline int
encode_native_event_raw( unsigned int event, unsigned int mask )
{
	unsigned int tmp = event << PAPI_NATIVE_EVENT_SHIFT;
	SUBDBG( "Old native index was %#08x with %#08x mask\n", tmp, mask );
	tmp = tmp | ( mask << PAPI_NATIVE_UMASK_SHIFT );
	SUBDBG( "New encoding is %#08x\n", tmp | PAPI_NATIVE_MASK );
	return ( int ) ( tmp | PAPI_NATIVE_MASK );
}

/* This routine converts array indices contained in the mask_values array
    into bits in the umask field that is OR'd into the native event code.
    These bits are NOT the mask values themselves, but indices into an array
    of mask values contained in the native event table. */
static inline int
encode_native_event( unsigned int event, unsigned int num_mask,
					 unsigned int *mask_values )
{
	unsigned int i;
	unsigned int tmp = event << PAPI_NATIVE_EVENT_SHIFT;
	SUBDBG( "Native base event is %#08x with %d masks\n", tmp, num_mask );
	for ( i = 0; i < num_mask; i++ ) {
		SUBDBG( "Mask index is %#08x\n", mask_values[i] );
		tmp = tmp | ( ( 1 << mask_values[i] ) << PAPI_NATIVE_UMASK_SHIFT );
	}
	SUBDBG( "Full native encoding is 0x%08x\n", tmp | PAPI_NATIVE_MASK );
	return ( int ) ( tmp | PAPI_NATIVE_MASK );
}


/* Break a PAPI native event code into its composite event code and pfm mask bits */
int
_pfm_decode_native_event( unsigned int EventCode, unsigned int *event,
						  unsigned int *umask )
{
	unsigned int tevent, major, minor;

	tevent = EventCode & PAPI_NATIVE_AND_MASK;
	major = ( tevent & PAPI_NATIVE_EVENT_AND_MASK ) >> PAPI_NATIVE_EVENT_SHIFT;
	if ( ( int ) major >= num_native_events )
		return PAPI_ENOEVNT;

	minor = ( tevent & PAPI_NATIVE_UMASK_AND_MASK ) >> PAPI_NATIVE_UMASK_SHIFT;
	*event = major;
	*umask = minor;
	SUBDBG( "EventCode %#08x is event %d, umask %#x\n", EventCode, major,
			minor );
	return PAPI_OK;
}

/* convert a collection of pfm mask bits into an array of pfm mask indices */
int
prepare_umask( unsigned int foo, unsigned int *values )
{
	unsigned int tmp = foo, i;
	int j = 0;

	SUBDBG( "umask %#x\n", tmp );
	while ( ( i = ( unsigned int ) ffs( ( int ) tmp ) ) ) {
		tmp = tmp ^ ( 1 << ( i - 1 ) );
		values[j] = i - 1;
		SUBDBG( "umask %d is %d\n", j, values[j] );
		j++;
	}
	return ( j );
}

/* convert the mask values in a pfm event structure into a PAPI unit mask */
static inline unsigned int
convert_pfm_masks( pfmlib_event_t * gete )
{
	int ret;
	unsigned int i, code, tmp = 0;

	for ( i = 0; i < gete->num_masks; i++ ) {
		if ( ( ret =
			   pfm_get_event_mask_code( gete->event, gete->unit_masks[i],
										&code ) ) == PFMLIB_SUCCESS ) {
			SUBDBG( "Mask value is %#08x\n", code );
			tmp |= code;
		} else {
			PAPIERROR( "pfm_get_event_mask_code(%#x,%d,%p): %s", gete->event,
					   i, &code, pfm_strerror( ret ) );
		}
	}
	return ( tmp );
}

/* convert an event code and pfm unit mask into a PAPI unit mask */
unsigned int
_pfm_convert_umask( unsigned int event, unsigned int umask )
{
	pfmlib_event_t gete;
	memset( &gete, 0, sizeof ( gete ) );
	gete.event = event;
	gete.num_masks = ( unsigned int ) prepare_umask( umask, gete.unit_masks );
	return ( convert_pfm_masks( &gete ) );
}

/* convert libpfm error codes to PAPI error codes for 
	more informative error reporting */
int
_papi_libpfm_error( int pfm_error )
{
	switch ( pfm_error ) {
		case PFMLIB_SUCCESS:		return PAPI_OK;			/* success */
		case PFMLIB_ERR_NOTSUPP:	return PAPI_ENOSUPP;	/* function not supported */
		case PFMLIB_ERR_INVAL:		return PAPI_EINVAL;		/* invalid parameters */
		case PFMLIB_ERR_NOINIT:		return PAPI_ENOINIT;	/* library was not initialized */
		case PFMLIB_ERR_NOTFOUND:	return PAPI_ENOEVNT;	/* event not found */
		case PFMLIB_ERR_NOASSIGN:	return PAPI_ECNFLCT;	/* cannot assign events to counters */
		case PFMLIB_ERR_FULL:		return PAPI_EBUF;		/* buffer is full or too small */
		case PFMLIB_ERR_EVTMANY:	return PAPI_EMISC;		/* event used more than once */
		case PFMLIB_ERR_MAGIC:		return PAPI_EBUG;		/* invalid library magic number */
		case PFMLIB_ERR_FEATCOMB:	return PAPI_ECOMBO;		/* invalid combination of features */
		case PFMLIB_ERR_EVTSET:		return PAPI_ENOEVST;	/* incompatible event sets */
		case PFMLIB_ERR_EVTINCOMP:	return PAPI_ECNFLCT;	/* incompatible event combination */
		case PFMLIB_ERR_TOOMANY:	return PAPI_ECOUNT;		/* too many events or unit masks */
		case PFMLIB_ERR_BADHOST:	return PAPI_ESYS;		/* not supported by host CPU */
		case PFMLIB_ERR_UMASK:		return PAPI_EATTR;		/* invalid or missing unit mask */
		case PFMLIB_ERR_NOMEM:		return PAPI_ENOMEM;		/* out of memory */

		/* Itanium only */
		case PFMLIB_ERR_IRRTOOBIG:		/* code range too big */
		case PFMLIB_ERR_IRREMPTY:		/* empty code range */
		case PFMLIB_ERR_IRRINVAL:		/* invalid code range */
		case PFMLIB_ERR_IRRTOOMANY:		/* too many code ranges */
		case PFMLIB_ERR_DRRINVAL:		/* invalid data range */
		case PFMLIB_ERR_DRRTOOMANY:		/* too many data ranges */
		case PFMLIB_ERR_IRRALIGN:		/* bad alignment for code range */
		case PFMLIB_ERR_IRRFLAGS:		/* code range missing flags */
		default:
			return PAPI_EINVAL;
	}
}

int
_papi_libpfm_ntv_name_to_code( const char *name, unsigned int *event_code )
{
	pfmlib_event_t event;
	unsigned int i;
	int ret;

	SUBDBG( "pfm_find_full_event(%s,%p)\n", name, &event );
	ret = pfm_find_full_event( name, &event );
	if ( ret == PFMLIB_SUCCESS ) {
		SUBDBG( "Full event name found\n" );
		/* we can only capture PAPI_NATIVE_UMASK_MAX or fewer masks */
		if ( event.num_masks > PAPI_NATIVE_UMASK_MAX ) {
			SUBDBG( "num_masks (%d) > max masks (%d)\n", event.num_masks,
					PAPI_NATIVE_UMASK_MAX );
			return PAPI_ENOEVNT;
		} else {
			/* no mask index can exceed PAPI_NATIVE_UMASK_MAX */
			for ( i = 0; i < event.num_masks; i++ ) {
				if ( event.unit_masks[i] > PAPI_NATIVE_UMASK_MAX ) {
					SUBDBG( "mask index (%d) > max masks (%d)\n",
							event.unit_masks[i], PAPI_NATIVE_UMASK_MAX );
					return PAPI_ENOEVNT;
				}
			}
			*event_code =
				encode_native_event( event.event, event.num_masks,
									 event.unit_masks );
			return PAPI_OK;
		}
	} else if ( ret == PFMLIB_ERR_UMASK ) {
		SUBDBG( "UMASK error, looking for base event only\n" );
		ret = pfm_find_event( name, &event.event );
		if ( ret == PFMLIB_SUCCESS ) {
			*event_code = encode_native_event( event.event, 0, 0 );
			return PAPI_EATTR;
		}
	}
	return PAPI_ENOEVNT;
}

int
_papi_libpfm_ntv_code_to_name( unsigned int EventCode, char *ntv_name, int len )
{
	int ret;
	unsigned int event, umask;
	pfmlib_event_t gete;

	memset( &gete, 0, sizeof ( gete ) );

	if ( _pfm_decode_native_event( EventCode, &event, &umask ) != PAPI_OK )
		return ( PAPI_ENOEVNT );

	gete.event = event;
	gete.num_masks = ( unsigned int ) prepare_umask( umask, gete.unit_masks );
	if ( gete.num_masks == 0 )
		ret = pfm_get_event_name( gete.event, ntv_name, ( size_t ) len );
	else
		ret = pfm_get_full_event_name( &gete, ntv_name, ( size_t ) len );
	if ( ret != PFMLIB_SUCCESS ) {
		char tmp[PAPI_2MAX_STR_LEN];
		pfm_get_event_name( gete.event, tmp, sizeof ( tmp ) );
		/* Skip error message if event is not supported by host cpu;
		 * we don't need to give this info away for papi_native_avail util */
		if ( ret != PFMLIB_ERR_BADHOST )
			PAPIERROR
				( "pfm_get_full_event_name(%p(event %d,%s,%d masks),%p,%d): %d -- %s",
				  &gete, gete.event, tmp, gete.num_masks, ntv_name, len, ret,
				  pfm_strerror( ret ) );
		if ( ret == PFMLIB_ERR_FULL ) {
			return PAPI_EBUF;
		}

		return PAPI_EMISC;
	}
	return PAPI_OK;
}

int
_papi_libpfm_ntv_code_to_descr( unsigned int EventCode, char *ntv_descr, int len )
{
	unsigned int event, umask;
	char *eventd, **maskd, *tmp;
	int i, ret;
	pfmlib_event_t gete;
	size_t total_len = 0;

	memset( &gete, 0, sizeof ( gete ) );

	if ( _pfm_decode_native_event( EventCode, &event, &umask ) != PAPI_OK )
		return ( PAPI_ENOEVNT );

	ret = pfm_get_event_description( event, &eventd );
	if ( ret != PFMLIB_SUCCESS ) {
		PAPIERROR( "pfm_get_event_description(%d,%p): %s",
				   event, &eventd, pfm_strerror( ret ) );
		return ( PAPI_ENOEVNT );
	}

	if ( ( gete.num_masks =
		   ( unsigned int ) prepare_umask( umask, gete.unit_masks ) ) ) {
		maskd = ( char ** ) malloc( gete.num_masks * sizeof ( char * ) );
		if ( maskd == NULL ) {
			free( eventd );
			return ( PAPI_ENOMEM );
		}
		for ( i = 0; i < ( int ) gete.num_masks; i++ ) {
			ret =
				pfm_get_event_mask_description( event, gete.unit_masks[i],
												&maskd[i] );
			if ( ret != PFMLIB_SUCCESS ) {
				PAPIERROR( "pfm_get_event_mask_description(%d,%d,%p): %s",
						   event, umask, &maskd, pfm_strerror( ret ) );
				free( eventd );
				for ( ; i >= 0; i-- )
					free( maskd[i] );
				free( maskd );
				return ( PAPI_EINVAL );
			}
			total_len += strlen( maskd[i] );
		}
		tmp =
			( char * ) malloc( strlen( eventd ) + strlen( ", masks:" ) +
							   total_len + gete.num_masks + 1 );
		if ( tmp == NULL ) {
			for ( i = ( int ) gete.num_masks - 1; i >= 0; i-- )
				free( maskd[i] );
			free( maskd );
			free( eventd );
		}
		tmp[0] = '\0';
		strcat( tmp, eventd );
		strcat( tmp, ", masks:" );
		for ( i = 0; i < ( int ) gete.num_masks; i++ ) {
			if ( i != 0 )
				strcat( tmp, "," );
			strcat( tmp, maskd[i] );
			free( maskd[i] );
		}
		free( maskd );
	} else {
		tmp = ( char * ) malloc( strlen( eventd ) + 1 );
		if ( tmp == NULL ) {
			free( eventd );
			return ( PAPI_ENOMEM );
		}
		tmp[0] = '\0';
		strcat( tmp, eventd );
		free( eventd );
	}
	strncpy( ntv_descr, tmp, ( size_t ) len );
	if ( ( int ) strlen( tmp ) > len - 1 )
		ret = PAPI_EBUF;
	else
		ret = PAPI_OK;
	free( tmp );
	return ( ret );
}


int
_papi_libpfm_ntv_code_to_info(unsigned int EventCode, PAPI_event_info_t *info)
{

  SUBDBG("ENTER %#x\n",EventCode);

  _papi_libpfm_ntv_code_to_name(EventCode,info->symbol,
                                 sizeof(info->symbol));

  _papi_libpfm_ntv_code_to_descr(EventCode,info->long_descr,
                                 sizeof(info->long_descr));

  return PAPI_OK;
}


int
_papi_libpfm_ntv_enum_events( unsigned int *EventCode, int modifier )
{
	unsigned int event, umask, num_masks;
	int ret;

	if ( modifier == PAPI_ENUM_FIRST ) {
		*EventCode = PAPI_NATIVE_MASK;	/* assumes first native event is always 0x4000000 */
		return ( PAPI_OK );
	}

	if ( _pfm_decode_native_event( *EventCode, &event, &umask ) != PAPI_OK )
		return ( PAPI_ENOEVNT );

	ret = pfm_get_num_event_masks( event, &num_masks );
	if ( ret != PFMLIB_SUCCESS ) {
		PAPIERROR( "pfm_get_num_event_masks(%d,%p): %s", event, &num_masks,
				   pfm_strerror( ret ) );
		return ( PAPI_ENOEVNT );
	}
	if ( num_masks > PAPI_NATIVE_UMASK_MAX )
		num_masks = PAPI_NATIVE_UMASK_MAX;
	SUBDBG( "This is umask %d of %d\n", umask, num_masks );

	if ( modifier == PAPI_ENUM_EVENTS ) {
		if ( event < ( unsigned int ) num_native_events - 1 ) {
			*EventCode =
				( unsigned int ) encode_native_event_raw( event + 1, 0 );
			return ( PAPI_OK );
		}
		return ( PAPI_ENOEVNT );
	} else if ( modifier == PAPI_NTV_ENUM_UMASK_COMBOS ) {
		if ( umask + 1 < ( unsigned int ) ( 1 << num_masks ) ) {
			*EventCode =
				( unsigned int ) encode_native_event_raw( event, umask + 1 );
			return ( PAPI_OK );
		}
		return ( PAPI_ENOEVNT );
	} else if ( modifier == PAPI_NTV_ENUM_UMASKS ) {
		int thisbit = ffs( ( int ) umask );

		SUBDBG( "First bit is %d in %08x\b\n", thisbit - 1, umask );
		thisbit = 1 << thisbit;

		if ( thisbit & ( ( 1 << num_masks ) - 1 ) ) {
			*EventCode =
				( unsigned int ) encode_native_event_raw( event,
														  ( unsigned int )
														  thisbit );
			return ( PAPI_OK );
		}
		return ( PAPI_ENOEVNT );
	} else
		return ( PAPI_EINVAL );
}

int
_papi_libpfm_ntv_code_to_bits( unsigned int EventCode, hwd_register_t * bits )
{
    unsigned int event, umask;
    pfmlib_event_t gete;

    /* For PFM & Perfmon, native info is just an index into PFM event table. */
    if ( _pfm_decode_native_event( EventCode, &event, &umask ) != PAPI_OK )
       return PAPI_ENOEVNT;

    memset( &gete, 0x0, sizeof ( pfmlib_event_t ) );

    gete.event = event;
    gete.num_masks = prepare_umask( umask, gete.unit_masks );

    memcpy( bits, &gete, sizeof ( pfmlib_event_t ) );

    return PAPI_OK;

}


/* used by linux-timer.c for ia64 */
int _perfmon2_pfm_pmu_type = -1;


int
_papi_libpfm_init(papi_vector_t *my_vector, int cidx) {

   int retval;
   unsigned int ncnt;
   unsigned int version;
   char pmu_name[PAPI_MIN_STR_LEN];


   /* The following checks the version of the PFM library
      against the version PAPI linked to... */
   SUBDBG( "pfm_initialize()\n" );
   if ( ( retval = pfm_initialize(  ) ) != PFMLIB_SUCCESS ) {
      PAPIERROR( "pfm_initialize(): %s", pfm_strerror( retval ) );
      return PAPI_ESYS;
   }

   /* Get the libpfm3 version */
   SUBDBG( "pfm_get_version(%p)\n", &version );
   if ( pfm_get_version( &version ) != PFMLIB_SUCCESS ) {
      PAPIERROR( "pfm_get_version(%p): %s", version, pfm_strerror( retval ) );
      return PAPI_ESYS;
   }

   /* Set the version */
   sprintf( my_vector->cmp_info.support_version, "%d.%d",
	    PFM_VERSION_MAJOR( version ), PFM_VERSION_MINOR( version ) );

   /* Complain if the compiled-against version doesn't match current version */
   if ( PFM_VERSION_MAJOR( version ) != PFM_VERSION_MAJOR( PFMLIB_VERSION ) ) {
      PAPIERROR( "Version mismatch of libpfm: compiled %#x vs. installed %#x\n",
				   PFM_VERSION_MAJOR( PFMLIB_VERSION ),
				   PFM_VERSION_MAJOR( version ) );
      return PAPI_ESYS;
   }

   /* Always initialize globals dynamically to handle forks properly. */

   _perfmon2_pfm_pmu_type = -1;

   /* Opened once for all threads. */
   SUBDBG( "pfm_get_pmu_type(%p)\n", &_perfmon2_pfm_pmu_type );
   if ( pfm_get_pmu_type( &_perfmon2_pfm_pmu_type ) != PFMLIB_SUCCESS ) {
      PAPIERROR( "pfm_get_pmu_type(%p): %s", _perfmon2_pfm_pmu_type,
				   pfm_strerror( retval ) );
      return PAPI_ESYS;
   }

   pmu_name[0] = '\0';
   if ( pfm_get_pmu_name( pmu_name, PAPI_MIN_STR_LEN ) != PFMLIB_SUCCESS ) {
      PAPIERROR( "pfm_get_pmu_name(%p,%d): %s", pmu_name, PAPI_MIN_STR_LEN,
				   pfm_strerror( retval ) );
      return PAPI_ESYS;
   }
   SUBDBG( "PMU is a %s, type %d\n", pmu_name, _perfmon2_pfm_pmu_type );

   /* Setup presets */
   retval = _papi_load_preset_table( pmu_name, _perfmon2_pfm_pmu_type, cidx );
   if ( retval )
      return retval;

   /* Fill in cmp_info */

   SUBDBG( "pfm_get_num_events(%p)\n", &ncnt );
   if ( ( retval = pfm_get_num_events( &ncnt ) ) != PFMLIB_SUCCESS ) {
      PAPIERROR( "pfm_get_num_events(%p): %s\n", &ncnt,
				   pfm_strerror( retval ) );
      return PAPI_ESYS;
   }
   SUBDBG( "pfm_get_num_events: %d\n", ncnt );
   my_vector->cmp_info.num_native_events = ncnt;
   num_native_events = ncnt;

   pfm_get_num_counters( ( unsigned int * ) &my_vector->cmp_info.num_cntrs );
   SUBDBG( "pfm_get_num_counters: %d\n", my_vector->cmp_info.num_cntrs );


   if ( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_INTEL ) {
     /* Pentium4 */
     if ( _papi_hwi_system_info.hw_info.cpuid_family == 15 ) {
       PAPI_NATIVE_EVENT_AND_MASK = 0x000000ff;
       PAPI_NATIVE_UMASK_AND_MASK = 0x0fffff00;
       PAPI_NATIVE_UMASK_SHIFT = 8;
       /* Itanium2 */
     } else if ( _papi_hwi_system_info.hw_info.cpuid_family == 31 ||
		 _papi_hwi_system_info.hw_info.cpuid_family == 32 ) {
       PAPI_NATIVE_EVENT_AND_MASK = 0x00000fff;
       PAPI_NATIVE_UMASK_AND_MASK = 0x0ffff000;
       PAPI_NATIVE_UMASK_SHIFT = 12;
     }
   }


   return PAPI_OK;
}

long long generate_p4_event(long long escr,
			    long long cccr,
			    long long escr_addr) {
		   
/*
 * RAW events specification
 *
 * Bits                Meaning
 * -----       -------
 *  0-6        Metric value from enum P4_PEBS_METRIC (if needed)
 *  7-11       Reserved, set to 0
 * 12-31       Bits 12-31 of CCCR register (Intel SDM Vol 3)
 * 32-56       Bits  0-24 of ESCR register (Intel SDM Vol 3)
 * 57-62       Event key from enum P4_EVENTS
 *    63       Reserved, set to 0
 */
		   
 enum P4_EVENTS {
      P4_EVENT_TC_DELIVER_MODE,
      P4_EVENT_BPU_FETCH_REQUEST,
      P4_EVENT_ITLB_REFERENCE,
      P4_EVENT_MEMORY_CANCEL,
      P4_EVENT_MEMORY_COMPLETE,
      P4_EVENT_LOAD_PORT_REPLAY,
      P4_EVENT_STORE_PORT_REPLAY,
      P4_EVENT_MOB_LOAD_REPLAY,
      P4_EVENT_PAGE_WALK_TYPE,
      P4_EVENT_BSQ_CACHE_REFERENCE,
      P4_EVENT_IOQ_ALLOCATION,
      P4_EVENT_IOQ_ACTIVE_ENTRIES,
      P4_EVENT_FSB_DATA_ACTIVITY,
      P4_EVENT_BSQ_ALLOCATION,
      P4_EVENT_BSQ_ACTIVE_ENTRIES,
      P4_EVENT_SSE_INPUT_ASSIST,
      P4_EVENT_PACKED_SP_UOP,
      P4_EVENT_PACKED_DP_UOP,
      P4_EVENT_SCALAR_SP_UOP,
      P4_EVENT_SCALAR_DP_UOP,
      P4_EVENT_64BIT_MMX_UOP,
      P4_EVENT_128BIT_MMX_UOP,
      P4_EVENT_X87_FP_UOP,
      P4_EVENT_TC_MISC,
      P4_EVENT_GLOBAL_POWER_EVENTS,
      P4_EVENT_TC_MS_XFER,
      P4_EVENT_UOP_QUEUE_WRITES,
      P4_EVENT_RETIRED_MISPRED_BRANCH_TYPE,
      P4_EVENT_RETIRED_BRANCH_TYPE,
      P4_EVENT_RESOURCE_STALL,
      P4_EVENT_WC_BUFFER,
      P4_EVENT_B2B_CYCLES,
      P4_EVENT_BNR,
      P4_EVENT_SNOOP,
      P4_EVENT_RESPONSE,
      P4_EVENT_FRONT_END_EVENT,
      P4_EVENT_EXECUTION_EVENT,
      P4_EVENT_REPLAY_EVENT,
      P4_EVENT_INSTR_RETIRED,
      P4_EVENT_UOPS_RETIRED,
      P4_EVENT_UOP_TYPE,
      P4_EVENT_BRANCH_RETIRED,
      P4_EVENT_MISPRED_BRANCH_RETIRED,
      P4_EVENT_X87_ASSIST,
      P4_EVENT_MACHINE_CLEAR,
      P4_EVENT_INSTR_COMPLETED,
   };
		   
		  		   
    int eventsel=(escr>>25)&0x3f;
    int cccrsel=(cccr>>13)&0x7;
    int event_key=-1;
    long long pe_event;
		   
    switch(eventsel) {
       case 0x1: if (cccrsel==1) {
		    if (escr_addr>0x3c8) {
		       // tc_escr0,1 0x3c4 
		       event_key=P4_EVENT_TC_DELIVER_MODE; 
		    }
		    else {
		       // alf_escr0, 0x3ca    
		       event_key=P4_EVENT_RESOURCE_STALL;
		    }
		 }
		 if (cccrsel==4) {	    
		    if (escr_addr<0x3af) {
		       // pmh_escr0,1 0x3ac
		       event_key=P4_EVENT_PAGE_WALK_TYPE;
		    }
		    else {
		       // cru_escr0, 3b8 cccr=04
		       event_key=P4_EVENT_UOPS_RETIRED;
		    }
		 }
		 break;
		    case 0x2: if (cccrsel==5) {
		                 if (escr_addr<0x3a8) { 
		                    // MSR_DAC_ESCR0 / MSR_DAC_ESCR1
		                    event_key=P4_EVENT_MEMORY_CANCEL; 
				 } else {
				   //MSR_CRU_ESCR2, MSR_CRU_ESCR3
				   event_key=P4_EVENT_MACHINE_CLEAR;
				 }
			      } else if (cccrsel==1) {
		      	         event_key=P4_EVENT_64BIT_MMX_UOP;
			      } else if (cccrsel==4) {
			         event_key=P4_EVENT_INSTR_RETIRED;
			      } else if (cccrsel==2) {
			         event_key=P4_EVENT_UOP_TYPE;
			      }
			      break;
		    case 0x3: if (cccrsel==0) {
		                 event_key=P4_EVENT_BPU_FETCH_REQUEST;
		              }
                              if (cccrsel==2) {
		                 event_key=P4_EVENT_MOB_LOAD_REPLAY;
			      }
		              if (cccrsel==6) {
			         event_key=P4_EVENT_IOQ_ALLOCATION;
			      }
		              if (cccrsel==4) {
			         event_key=P4_EVENT_MISPRED_BRANCH_RETIRED;
		              }
			      if (cccrsel==5) { 
				 event_key=P4_EVENT_X87_ASSIST;
		              }
			      break;
		    case 0x4: if (cccrsel==2) {
		                 if (escr_addr<0x3b0) {
				    // saat, 0x3ae 
		                    event_key=P4_EVENT_LOAD_PORT_REPLAY; 
		                 }
		                 else {
				    // tbpu 0x3c2
		                    event_key=P4_EVENT_RETIRED_BRANCH_TYPE;
				 }
		              }
		              if (cccrsel==1) {
		      	         event_key=P4_EVENT_X87_FP_UOP;
		              }
			      if (cccrsel==3) {
			         event_key=P4_EVENT_RESPONSE;
		              }
			      break;
                    case 0x5: if (cccrsel==2) {
		                 if (escr_addr<0x3b0) {
		                    // saat, 0x3ae 
		                    event_key=P4_EVENT_STORE_PORT_REPLAY;
				 }
		                 else {
		                    // tbpu, 0x3c2
		                    event_key=P4_EVENT_RETIRED_MISPRED_BRANCH_TYPE;
				 }
		              }
		              if (cccrsel==7) {
		      	         event_key=P4_EVENT_BSQ_ALLOCATION;
		              }
		              if (cccrsel==0) {
			         event_key=P4_EVENT_TC_MS_XFER;
		              }
			      if (cccrsel==5) {
			         event_key=P4_EVENT_WC_BUFFER;
		              }
			      break;
		    case 0x6: if (cccrsel==7) {
		                 event_key=P4_EVENT_BSQ_ACTIVE_ENTRIES; 
		              }
		              if (cccrsel==1) {
		      	         event_key=P4_EVENT_TC_MISC;
			      }
			      if (cccrsel==3) {
				 event_key=P4_EVENT_SNOOP;
			      }
		              if (cccrsel==5) {
			         event_key=P4_EVENT_BRANCH_RETIRED;
			      }
			      break;
		    case 0x7: event_key=P4_EVENT_INSTR_COMPLETED; break;
		    case 0x8: if (cccrsel==2) {
		                 event_key=P4_EVENT_MEMORY_COMPLETE; 
		              }
		      	      if (cccrsel==1) {
				 event_key=P4_EVENT_PACKED_SP_UOP;
			      }
			      if (cccrsel==3) {
				 event_key=P4_EVENT_BNR;
		              }
			      if (cccrsel==5) {
				 event_key=P4_EVENT_FRONT_END_EVENT;
		              }
			      break;
                    case 0x9: if (cccrsel==0) {
		                 event_key=P4_EVENT_UOP_QUEUE_WRITES; 
		              }
		      	      if (cccrsel==5) {
				 event_key=P4_EVENT_REPLAY_EVENT;
			      }
			      break;
                    case 0xa: event_key=P4_EVENT_SCALAR_SP_UOP; break;
                    case 0xc: if (cccrsel==7) {
		                 event_key=P4_EVENT_BSQ_CACHE_REFERENCE; 
		              }
		              if (cccrsel==1) {
		      	         event_key=P4_EVENT_PACKED_DP_UOP;
			      }
			      if (cccrsel==5) {
				 event_key=P4_EVENT_EXECUTION_EVENT;
			      }
			      break;
		    case 0xe: event_key=P4_EVENT_SCALAR_DP_UOP; break;
		    case 0x13: event_key=P4_EVENT_GLOBAL_POWER_EVENTS; break;
                    case 0x16: event_key=P4_EVENT_B2B_CYCLES; break;
		    case 0x17: event_key=P4_EVENT_FSB_DATA_ACTIVITY; break;
		    case 0x18: event_key=P4_EVENT_ITLB_REFERENCE; break;
                    case 0x1a: if (cccrsel==6) {
		                  event_key=P4_EVENT_IOQ_ACTIVE_ENTRIES; 
		               }
		               if (cccrsel==1) {
			          event_key=P4_EVENT_128BIT_MMX_UOP;
		  }
		  break;
       case 0x34: event_key= P4_EVENT_SSE_INPUT_ASSIST; break;
    }
		   
    pe_event=(escr&0x1ffffff)<<32;
    pe_event|=(cccr&0xfffff000);		    
    pe_event|=(((long long)(event_key))<<57);
   
    return pe_event;
}

typedef pfmlib_event_t pfm_register_t;

int
_papi_libpfm_setup_counters( struct perf_event_attr *attr, 
			   hwd_register_t *ni_bits ) {

  int ret,pe_event;
  (void)ni_bits;

    /*
     * We need an event code that is common across all counters.
     * The implementation is required to know how to translate the supplied
     * code to whichever counter it ends up on.
     */

#if defined(__powerpc__)
    int code;
    ret = pfm_get_event_code_counter( ( ( pfm_register_t * ) ni_bits )->event, 0, &code );
    if ( ret ) {
       /* Unrecognized code, but should never happen */
       return PAPI_EBUG;
    }
    pe_event = code;
    SUBDBG( "Stuffing native event index (code %#x, raw code %#x) into events array.\n",
				  ( ( pfm_register_t * ) ni_bits )->event, code );
#else

   pfmlib_input_param_t inp;
   pfmlib_output_param_t outp;

   memset( &inp, 0, sizeof ( inp ) );
   memset( &outp, 0, sizeof ( outp ) );
   inp.pfp_event_count = 1;
   inp.pfp_dfl_plm = PAPI_DOM_USER;
   pfm_regmask_set( &inp.pfp_unavail_pmcs, 16 );	// mark fixed counters as unavailable

    inp.pfp_events[0] = *( ( pfm_register_t * ) ni_bits );
    ret = pfm_dispatch_events( &inp, NULL, &outp, NULL );
    if (ret != PFMLIB_SUCCESS) {
       SUBDBG( "Error: pfm_dispatch_events returned: %d\n", ret);
       return PAPI_ESYS;
    }
		   	
       /* Special case p4 */
    if (( _papi_hwi_system_info.hw_info.vendor == PAPI_VENDOR_INTEL ) && 
        ( _papi_hwi_system_info.hw_info.cpuid_family == 15)) {

	pe_event=generate_p4_event( outp.pfp_pmcs[0].reg_value, /* escr */  
		                    outp.pfp_pmcs[1].reg_value, /* cccr */
		                    outp.pfp_pmcs[0].reg_addr); /* escr_addr */
    }
    else {
        pe_event = outp.pfp_pmcs[0].reg_value;   
    }
    SUBDBG( "pe_event: %#llx\n", outp.pfp_pmcs[0].reg_value );
#endif

    attr->config=pe_event;

    /* for libpfm3 we currently only handle RAW type */
    attr->type=PERF_TYPE_RAW;

    return PAPI_OK;
}

int 
_papi_libpfm_shutdown(void) {

  SUBDBG("shutdown\n");

  return PAPI_OK;
}


