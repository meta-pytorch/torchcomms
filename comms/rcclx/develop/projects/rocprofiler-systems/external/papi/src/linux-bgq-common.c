/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-bgq-common.c
 * CVS:     $Id$
 * @author  Heike Jagode
 *          jagode@eecs.utk.edu
 * Mods:	< your name here >
 *			< your email address >
 * BGPM component 
 * 
 * Tested version of bgpm (early access)
 *
 * @brief
 *  This file is part of the source code for a component that enables PAPI-C to 
 *  access hardware monitoring counters for BG/Q through the bgpm library.
 */

#include "linux-bgq-common.h"

/*******************************************************************************
 ********  BEGIN FUNCTIONS USED INTERNALLY SPECIFIC TO THIS COMPONENT **********
 ******************************************************************************/

int _check_BGPM_error( int err, char* bgpmfunc )
{
    char  buffer[PAPI_MAX_STR_LEN];
    int retval;
    
	if ( err < 0 ) {
        sprintf( buffer, "Error: ret value is %d for BGPM API function '%s'.",
                err, bgpmfunc);
        retval =  _papi_hwi_publish_error( buffer );
        return retval;
	}
    
    return PAPI_OK;
}


/*
 * Returns all event values from the BGPM eventGroup
 */
long_long
_common_getEventValue( unsigned event_id, int EventGroup )
{	
	uint64_t value;
    int retval;
	
	retval = Bgpm_ReadEvent( EventGroup, event_id, &value );
	retval = _check_BGPM_error( retval, "Bgpm_ReadEvent" );
	if ( retval < 0 ) return retval;

	return ( ( long_long ) value );	
}


/*
 * Delete BGPM eventGroup and create an new empty one
 */
int
_common_deleteRecreate( int *EventGroup_ptr )
{
#ifdef DEBUG_BGQ
	printf( _AT_ " _common_deleteRecreate: *EventGroup_ptr=%d\n", *EventGroup_ptr);
#endif
	int retval;
	
	// delete previous bgpm eventset
	retval = Bgpm_DeleteEventSet( *EventGroup_ptr );
	retval = _check_BGPM_error( retval, "Bgpm_DeleteEventSet" );
	if ( retval < 0 ) return retval;

	// create a new empty bgpm eventset
	*EventGroup_ptr = Bgpm_CreateEventSet();
	retval = _check_BGPM_error( *EventGroup_ptr, "Bgpm_CreateEventSet" );
	if ( retval < 0 ) return retval;

#ifdef DEBUG_BGQ
	printf( _AT_ " _common_deleteRecreate: *EventGroup_ptr=%d\n", *EventGroup_ptr);
#endif
	return PAPI_OK;
}


/*
 * Rebuild BGPM eventGroup with the events as it was prior to deletion 
 */
int
_common_rebuildEventgroup( int count, int *EventGroup_local, int *EventGroup_ptr )
{
#ifdef DEBUG_BGQ
	printf( "_common_rebuildEventgroup\n" );
#endif	
	int i, retval;
	
	// rebuild BGPM EventGroup
	for ( i = 0; i < count; i++ ) {
		retval = Bgpm_AddEvent( *EventGroup_ptr, EventGroup_local[i] );
		retval = _check_BGPM_error( retval, "Bgpm_AddEvent" );
		if ( retval < 0 ) return retval;

#ifdef DEBUG_BGQ
		printf( "_common_rebuildEventgroup: After emptying EventGroup, event re-added: %d\n",
			    EventGroup_local[i] );
#endif
	}
	return PAPI_OK;
}


/*
 * _common_set_overflow_BGPM
 *
 * since update_control_state trashes overflow settings, this puts things
 * back into balance for BGPM 
 */
int
_common_set_overflow_BGPM( int EventGroup, 
						   int evt_idx,
						   int threshold, 
						   void (*handler)(int, uint64_t, uint64_t, const ucontext_t *) )
{
	int retval;
	uint64_t threshold_for_bgpm;
	
	/* convert threadhold value assigned by PAPI user to value that is
	 * programmed into the counter. This value is required by Bgpm_SetOverflow() */ 
	threshold_for_bgpm = BGPM_PERIOD2THRES( threshold );
	
#ifdef DEBUG_BGQ
	printf("_common_set_overflow_BGPM\n");
	
	int i;
	int numEvts = Bgpm_NumEvents( EventGroup );
	for ( i = 0; i < numEvts; i++ ) {
		printf("_common_set_overflow_BGPM: %d = %s\n", i, Bgpm_GetEventLabel( EventGroup, i) );
	}
#endif	
	
	
	retval = Bgpm_SetOverflow( EventGroup, 
							   evt_idx,
							   threshold_for_bgpm );
	retval = _check_BGPM_error( retval, "Bgpm_SetOverflow" );
	if ( retval < 0 ) return retval;

	retval = Bgpm_SetEventUser1( EventGroup, 
								 evt_idx,
								 1024 );
	retval = _check_BGPM_error( retval, "Bgpm_SetEventUser1" );
	if ( retval < 0 ) return retval;

	/* user signal handler for overflow case */
	retval = Bgpm_SetOverflowHandler( EventGroup, 
									  handler );
	retval = _check_BGPM_error( retval, "Bgpm_SetOverflowHandler" );	
	if ( retval < 0 ) return retval;

	return PAPI_OK;
}



