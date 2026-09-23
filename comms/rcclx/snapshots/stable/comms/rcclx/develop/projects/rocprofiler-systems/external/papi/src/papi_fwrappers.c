/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    papi_fwrappers.c
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    Nils Smeds
*          smeds@pdc.kth.se
*          Anders Nilsson
*          anni@pdc.kth.se
*	   Kevin London
*	   london@cs.utk.edu
*	   dan terpstra
*	   terpstra@cs.utk.edu
*          Min Zhou
*	   min@cs.utk.edu
*/

#pragma GCC visibility push(default)

#include <stdio.h>
#include <assert.h>
#include <string.h>
#include "papi.h"

/* Lets use defines to rename all the files */

#ifdef FORTRANUNDERSCORE
#define PAPI_FCALL(function,caps,args) void function##_ args
#elif FORTRANDOUBLEUNDERSCORE
#define PAPI_FCALL(function,caps,args) void function##__ args
#elif FORTRANCAPS
#define PAPI_FCALL(function,caps,args) void caps args
#else
#define PAPI_FCALL(function,caps,args) void function args
#endif

/* Many Unix systems passes Fortran string lengths as extra arguments */
#if defined(_AIX) || defined(sun) || defined(linux)
#define _FORTRAN_STRLEN_AT_END
#endif
/* The Low Level Wrappers */

/** \internal @defgroup PAPIF PAPI Fortran Low Level API */

/* helper routine to convert Fortran strings to C strings */
#if defined(_FORTRAN_STRLEN_AT_END)
static void Fortran2cstring( char *cstring, char *Fstring, int clen , int Flen )
{
	int slen, i;

	/* What is the maximum number of chars to copy ? */
	slen = Flen < clen ? Flen : clen;
	strncpy( cstring, Fstring, ( size_t ) slen );

	/* Remove trailing blanks from initial Fortran string */
	for ( i = slen - 1; i > -1 && cstring[i] == ' '; cstring[i--] = '\0' );

	/* Make sure string is NULL terminated */
	cstring[clen - 1] = '\0';
	if ( slen < clen )
		cstring[slen] = '\0';
}
#endif

/**	@class PAPIF_accum
 *	@ingroup PAPIF
 *	@brief accumulate and reset counters in an event set 
 *	
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_accum( C_INT EventSet, C_LONG_LONG(*) values, C_INT  check )
 *
 *	@see PAPI_accum
 */
PAPI_FCALL( papif_accum, PAPIF_ACCUM,
			( int *EventSet, long long *values, int *check ) )
{
	*check = PAPI_accum( *EventSet, values );
}

/**	@class PAPIF_add_event
 *	@ingroup PAPIF
 *	@brief add PAPI preset or native hardware event to an event set
 *	
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_add_event( C_INT EventSet, C_INT EventCode, C_INT check )
 *
 *	@see PAPI_add_event
 */
PAPI_FCALL( papif_add_event, PAPIF_ADD_EVENT,
			( int *EventSet, int *Event, int *check ) )
{
	*check = PAPI_add_event( *EventSet, *Event );
}

/**	@class PAPIF_add_named_event
 *	@ingroup PAPIF
 *	@brief add PAPI preset or native hardware event to an event set by name
 *	
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_add_named_event( C_INT EventSet, C_STRING EventName, C_INT check )
 *
 *	@see PAPI_add_named_event
 */
#if defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL( papif_add_named_event, PAPIF_ADD_NAMED_EVENT,
			( int *EventSet, char *EventName, int *check, int Event_len ) )
{
	char tmp[PAPI_MAX_STR_LEN];
	Fortran2cstring( tmp, EventName, PAPI_MAX_STR_LEN, Event_len );
	*check = PAPI_add_named_event( *EventSet, tmp );	
}
#else
PAPI_FCALL( papif_add_named_event, PAPIF_ADD_NAMED_EVENT,
			( int *EventSet, char *EventName, int *check ) )
{
	*check = PAPI_add_named_event( *EventSet, EventName );
}
#endif

/**	@class PAPIF_add_events
 *	@ingroup PAPIF
 *	@brief add multiple PAPI presets or native hardware events to an event set
 *	
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_add_events( C_INT EventSet, C_INT(*) EventCodes, C_INT number, C_INT  check )
 *
 *	@see PAPI_add_events
 */
PAPI_FCALL( papif_add_events, PAPIF_ADD_EVENTS,
			( int *EventSet, int *Events, int *number, int *check ) )
{
	*check = PAPI_add_events( *EventSet, Events, *number );
}

/**	@class PAPIF_cleanup_eventset
 *	@ingroup PAPIF
 *	@brief empty and destroy an EventSet 
 *	
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_cleanup_eventset( C_INT EventSet, C_INT check )
 *
 *	@see PAPI_cleanup_eventset
 */
PAPI_FCALL( papif_cleanup_eventset, PAPIF_CLEANUP_EVENTSET,
			( int *EventSet, int *check ) )
{
	*check = PAPI_cleanup_eventset( *EventSet );
}

/**	@class PAPIF_create_eventset
 *	@ingroup PAPIF
 *	@brief create a new empty PAPI EventSet 
 *	
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_create_eventset( C_INT EventSet, C_INT check )
 *
 *	@see PAPI_create_eventset
 */
PAPI_FCALL( papif_create_eventset, PAPIF_CREATE_EVENTSET,
			( int *EventSet, int *check ) )
{
	*check = PAPI_create_eventset( EventSet );
}

/**	@class PAPIF_assign_eventset_component
 *	@ingroup PAPIF
 *	@brief assign a component index to an existing but empty EventSet 
 *	
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_assign_eventset_component( C_INT EventSet, C_INT EventSet, C_INT check )
 *
 *	@see PAPI_assign_eventset_component
 */
PAPI_FCALL( papif_assign_eventset_component, PAPIF_ASSIGN_EVENTSET_COMPONENT,
			( int *EventSet, int *cidx, int *check ) )
{
	*check = PAPI_assign_eventset_component( *EventSet, *cidx );
}

/**	@class PAPIF_destroy_eventset
 *	@ingroup PAPIF
 *	@brief empty and destroy an EventSet 
 *	
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_destroy_eventset( C_INT EventSet, C_INT check )
 *
 *	@see PAPI_destroy_eventset
 */
PAPI_FCALL( papif_destroy_eventset, PAPIF_DESTROY_EVENTSET,
			( int *EventSet, int *check ) )
{
	*check = PAPI_destroy_eventset( EventSet );
}

/**	@class PAPIF_get_dmem_info
 *	@ingroup PAPIF
 *	@brief get information about the dynamic memory usage of the current program 
 *	
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_get_dmem_info( C_INT EventSet, C_INT check )
 *
 *	@see PAPI_get_dmem_info
 */
 /* XXX This looks totally broken. Should be passed all the members of the dmem_info struct. */
PAPI_FCALL( papif_get_dmem_info, PAPIF_GET_DMEM_INFO,
			( long long *dest, int *check ) )
{
	*check = PAPI_get_dmem_info( ( PAPI_dmem_info_t * ) dest );
}

/**	@class PAPIF_get_exe_info
 *	@ingroup PAPIF
 *	@brief get information about the dynamic memory usage of the current program 
 *	
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_get_exe_info( C_STRING fullname, C_STRING name, @n
 *						C_LONG_LONG text_start,  C_LONG_LONG text_end, @n
 *						C_LONG_LONG data_start,  C_LONG_LONG data_end, @n
 *						C_LONG_LONG bss_start,   C_LONG_LONG bss_end, C_INT check )
 *
 *	@see PAPI_get_executable_info
 */
#if defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL( papif_get_exe_info, PAPIF_GET_EXE_INFO,
			( char *fullname, char *name, long long *text_start,
			  long long *text_end, long long *data_start, long long *data_end,
			  long long *bss_start, long long *bss_end,
			  int *check, int fullname_len, int name_len ) )
#else
PAPI_FCALL( papif_get_exe_info, PAPIF_GET_EXE_INFO,
			( char *fullname, char *name, long long *text_start,
			  long long *text_end, long long *data_start, long long *data_end,
			  long long *bss_start, long long *bss_end, int *check ) )
#endif
{
	PAPI_option_t e;
/* WARNING: The casts from vptr_t to long below WILL BREAK on systems with
    64-bit addresses. I did it here because I was lazy. And because I wanted
    to get rid of those pesky gcc warnings. If you find a 64-bit system,
    conditionalize the cast with (yet another) #ifdef...
*/
	if ( ( *check = PAPI_get_opt( PAPI_EXEINFO, &e ) ) == PAPI_OK ) {
#if defined(_FORTRAN_STRLEN_AT_END)
		int i;
		strncpy( fullname, e.exe_info->fullname, ( size_t ) fullname_len );
		for ( i = ( int ) strlen( e.exe_info->fullname ); i < fullname_len;
			  fullname[i++] = ' ' );
		strncpy( name, e.exe_info->address_info.name, ( size_t ) name_len );
		for ( i = ( int ) strlen( e.exe_info->address_info.name ); i < name_len;
			  name[i++] = ' ' );
#else
		strncpy( fullname, e.exe_info->fullname, PAPI_MAX_STR_LEN );
		strncpy( name, e.exe_info->address_info.name, PAPI_MAX_STR_LEN );
#endif
		*text_start = ( long ) e.exe_info->address_info.text_start;
		*text_end = ( long ) e.exe_info->address_info.text_end;
		*data_start = ( long ) e.exe_info->address_info.data_start;
		*data_end = ( long ) e.exe_info->address_info.data_end;
		*bss_start = ( long ) e.exe_info->address_info.bss_start;
		*bss_end = ( long ) e.exe_info->address_info.bss_end;
	}
}

/**	@class PAPIF_get_hardware_info 
 *	@ingroup PAPIF
 *	@brief get information about the system hardware
 *	
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_get_hardware_info( C_INT ncpu, C_INT nnodes,  C_INT totalcpus,@n
 *						C_INT vendor, C_STRING vendor_str, C_INT model, C_STRING model_str, @n
 *						C_FLOAT revision, C_FLOAT mhz )
 *
 *	@see PAPI_get_hardware_info
 */
#if defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL( papif_get_hardware_info, PAPIF_GET_HARDWARE_INFO, ( int *ncpu,
																int *nnodes,
																int *totalcpus,
																int *vendor,
																char
																*vendor_str,
																int *model,
																char *model_str,
																float *revision,
																float *mhz,
																int vendor_len,
																int
																model_len ) )
#else
PAPI_FCALL( papif_get_hardware_info, PAPIF_GET_HARDWARE_INFO, ( int *ncpu,
																int *nnodes,
																int *totalcpus,
																int *vendor,
																char
																*vendor_string,
																int *model,
																char
																*model_string,
																float *revision,
																float *mhz ) )
#endif
{
	const PAPI_hw_info_t *hwinfo;
	int i;
	hwinfo = PAPI_get_hardware_info(  );
	if ( hwinfo == NULL ) {
		*ncpu = 0;
		*nnodes = 0;
		*totalcpus = 0;
		*vendor = 0;
		*model = 0;
		*revision = 0;
		*mhz = 0;
	} else {
		*ncpu = hwinfo->ncpu;
		*nnodes = hwinfo->nnodes;
		*totalcpus = hwinfo->totalcpus;
		*vendor = hwinfo->vendor;
		*model = hwinfo->model;
		*revision = hwinfo->revision;
		*mhz = hwinfo->cpu_max_mhz;
#if defined(_FORTRAN_STRLEN_AT_END)
		strncpy( vendor_str, hwinfo->vendor_string, ( size_t ) vendor_len );
		for ( i = ( int ) strlen( hwinfo->vendor_string ); i < vendor_len;
			  vendor_str[i++] = ' ' );
		strncpy( model_str, hwinfo->model_string, ( size_t ) model_len );
		for ( i = ( int ) strlen( hwinfo->model_string ); i < model_len;
			  model_str[i++] = ' ' );
#else
		(void)i; /* unused...  */
		/* This case needs the passed strings to be of sufficient size *
		 * and will include the NULL character in the target string    */
		strcpy( vendor_string, hwinfo->vendor_string );
		strcpy( model_string, hwinfo->model_string );
#endif
	}
	return;
}

/** @class PAPIF_num_hwctrs
 *	@ingroup PAPIF
 *  @brief Return the number of hardware counters on the cpu.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_num_hwctrs( C_INT num )
 *
 * @see PAPI_num_hwctrs
 * @see PAPI_num_cmp_hwctrs
 */
PAPI_FCALL( papif_num_hwctrs, PAPIF_num_hwctrs, ( int *num ) )
{
	*num = PAPI_num_hwctrs(  );
}

/** @class PAPIF_num_cmp_hwctrs
 *	@ingroup PAPIF
 *  @brief Return the number of hardware counters on the specified component.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_num_cmp_hwctrs( C_INT cidx, C_INT num )
 *
 * @see PAPI_num_hwctrs
 * @see PAPI_num_cmp_hwctrs
 */
PAPI_FCALL( papif_num_cmp_hwctrs, PAPIF_num_cmp_hwctrs,
			( int *cidx, int *num ) )
{
	*num = PAPI_num_cmp_hwctrs( *cidx );
}

/** @class PAPIF_get_real_cyc
 *	@ingroup PAPIF
 *	@brief Get real time counter value in clock cycles.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_get_real_cyc( C_LONG_LONG real_cyc )
 *
 * @see PAPI_get_real_cyc
 */
PAPI_FCALL( papif_get_real_cyc, PAPIF_GET_REAL_CYC, ( long long *real_cyc ) )
{
	*real_cyc = PAPI_get_real_cyc(  );
}

/** @class PAPIF_get_real_usec
 *	@ingroup PAPIF
 *	@brief Get real time counter value in microseconds.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_get_real_usec( C_LONG_LONG time )
 *
 * @see PAPI_get_real_usec
 */
PAPI_FCALL( papif_get_real_usec, PAPIF_GET_REAL_USEC, ( long long *time ) )
{
	*time = PAPI_get_real_usec(  );
}

/** @class PAPIF_get_real_nsec
 *	@ingroup PAPIF
 *	@brief Get real time counter value in nanoseconds.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_get_real_nsec( C_LONG_LONG time )
 *
 * @see PAPI_get_real_nsec
 */
PAPI_FCALL( papif_get_real_nsec, PAPIF_GET_REAL_NSEC, ( long long *time ) )
{
	*time = PAPI_get_real_nsec(  );
}

/** @class PAPIF_get_virt_cyc
 *	@ingroup PAPIF
 *	@brief Get virtual time counter value in clock cycles.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_get_virt_cyc( C_LONG_LONG virt_cyc )
 *
 * @see PAPI_get_virt_cyc
 */
PAPI_FCALL( papif_get_virt_cyc, PAPIF_GET_VIRT_CYC, ( long long *virt_cyc ) )
{
	*virt_cyc = PAPI_get_virt_cyc(  );
}

/** @class PAPIF_get_virt_usec
 *	@ingroup PAPIF
 *	@brief Get virtual time counter value in microseconds.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_get_virt_usec( C_LONG_LONG time )
 *
 * @see PAPI_get_virt_usec
 */
PAPI_FCALL( papif_get_virt_usec, PAPIF_GET_VIRT_USEC, ( long long *time ) )
{
	*time = PAPI_get_virt_usec(  );
}

/** @class PAPIF_is_initialized
 *	@ingroup PAPIF
 *	@brief Check for initialization.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_is_initialized( C_INT level )
 *
 * @see PAPI_is_initialized
 */
PAPI_FCALL( papif_is_initialized, PAPIF_IS_INITIALIZED, ( int *level ) )
{
	*level = PAPI_is_initialized(  );
}

/** @class PAPIF_library_init
 *	@ingroup PAPIF
 *	@brief Initialize the PAPI library. 
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_library_init( C_INT check )
 *
 * @see PAPI_library_init
 */
PAPI_FCALL( papif_library_init, PAPIF_LIBRARY_INIT, ( int *check ) )
{
	*check = PAPI_library_init( *check );
}

/** @class PAPIF_thread_id
 *	@ingroup PAPIF
 *  @brief Get the thread identifier of the current thread.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_thread_id( C_INT id )
 *
 * @see PAPI_thread_id
 */
PAPI_FCALL( papif_thread_id, PAPIF_THREAD_ID, ( unsigned long *id ) )
{
	*id = PAPI_thread_id(  );
}

/** @class PAPIF_register_thread
 *	@ingroup PAPIF
 *  @brief Notify PAPI that a thread has 'appeared'.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_register_thread( C_INT check )
 *
 * @see PAPI_register_thread
 */
PAPI_FCALL( papif_register_thread, PAPIF_REGISTER_THREAD, ( int *check ) )
{
	*check = PAPI_register_thread(  );
}

/** @class PAPIF_unregister_thread
 *	@ingroup PAPIF
 *  @brief Notify PAPI that a thread has 'disappeared'.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_unregister_thread( C_INT check )
 *
 * @see PAPI_unregister_thread
 */
PAPI_FCALL( papif_unregister_thread, PAPIF_UNREGISTER_THREAD, ( int *check ) )
{
	*check = PAPI_unregister_thread(  );
}

/* There was a long (10+ years!) typo that was not noticed here */
/* Leaving it here as not to break any existing code out there */

PAPI_FCALL( papif_unregster_thread, PAPIF_UNREGSTER_THREAD, ( int *check ) )
{
	*check = PAPI_unregister_thread(  );
}

/** @class PAPIF_thread_init
 *	@ingroup PAPIF
 *  @brief Initialize thread support in the PAPI library.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_thread_init( C_INT FUNCTION  handle,  C_INT  check )
 *
 * @see PAPI_thread_init
 */
/* This must be passed an EXTERNAL or INTRINISIC FUNCTION not a SUBROUTINE */
PAPI_FCALL( papif_thread_init, PAPIF_THREAD_INIT,
			( unsigned long int ( *handle ) ( void ), int *check ) )
{
	*check = PAPI_thread_init( handle );
}

/** @class PAPI_list_events
 *	@ingroup PAPIF
 *	@brief List the events in an event set.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPI_list_events( C_INT EventSet, C_INT(*) Events, C_INT number, C_INT check )
 *
 * @see PAPI_list_events
 */
PAPI_FCALL( papif_list_events, PAPIF_LIST_EVENTS,
			( int *EventSet, int *Events, int *number, int *check ) )
{
	*check = PAPI_list_events( *EventSet, Events, number );
}

/** @class PAPIF_multiplex_init
 *	@ingroup PAPIF
 *	@brief Initialize multiplex support in the PAPI library.
 *
 * @par Fortran Interface:
 * \#include "fpapi.h" @n
 * PAPIF_multiplex_init( C_INT check )
 *
 * @see PAPI_multiplex_init
 */
PAPI_FCALL( papif_multiplex_init, PAPIF_MULTIPLEX_INIT, ( int *check ) )
{
	*check = PAPI_multiplex_init(  );
}

/** @class PAPIF_get_multiplex
 *	@ingroup PAPIF
 *	@brief Get the multiplexing status of specified event set.
 *
 * @par Fortran Interface:
 * \#include "fpapi.h" @n
 * PAPIF_get_multiplex( C_INT  EventSet,  C_INT  check )
 *
 * @see PAPI_get_multiplex
 */
PAPI_FCALL( papif_get_multiplex, PAPIF_GET_MULTIPLEX,
			( int *EventSet, int *check ) )
{
	*check = PAPI_get_multiplex( *EventSet );
}

/** @class PAPIF_set_multiplex
 *	@ingroup PAPIF
 *	@brief Convert a standard event set to a multiplexed event set. 
 *
 * @par Fortran Interface:
 * \#include "fpapi.h" @n
 * PAPIF_set_multiplex( C_INT  EventSet,  C_INT  check )
 *
 * @see PAPI_set_multiplex
 */
PAPI_FCALL( papif_set_multiplex, PAPIF_SET_MULTIPLEX,
			( int *EventSet, int *check ) )
{
	*check = PAPI_set_multiplex( *EventSet );
}

/** @class PAPIF_perror
 *	@ingroup PAPIF
 *  @brief Convert PAPI error codes to strings, and print error message to stderr. 
 *
 * @par Fortran Interface:
 * \#include "fpapi.h" @n
 *     PAPIF_perror( C_STRING message )
 *
 * @see PAPI_perror
 */
#if defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL( papif_perror, PAPIF_PERROR,
			( char *message,
			  int message_len ) )
#else
PAPI_FCALL( papif_perror, PAPIF_PERROR,
			( char *message ) )
#endif
{
#if defined(_FORTRAN_STRLEN_AT_END)
		char tmp[PAPI_MAX_STR_LEN];
		Fortran2cstring( tmp, message, PAPI_MAX_STR_LEN, message_len );

	PAPI_perror( tmp );
#else
	PAPI_perror( message );
#endif
}

/* This will not work until Fortran2000 :)
 * PAPI_FCALL(papif_profil, PAPIF_PROFIL, (unsigned short *buf, unsigned *bufsiz, unsigned long *offset, unsigned *scale, unsigned *eventset, 
 *            unsigned *eventcode, unsigned *threshold, unsigned *flags, unsigned *check))
 * {
 * *check = PAPI_profil(buf, *bufsiz, *offset, *scale, *eventset, *eventcode, *threshold, *flags);
 * }
 */

/** @class PAPIF_query_event
 *	@ingroup PAPIF
 *  @brief Query if PAPI event exists.
 *
 * @par Fortran Interface:
 * \#include "fpapi.h" @n
 * PAPIF_query_event(C_INT EventCode, C_INT check )
 *
 * @see PAPI_query_event
 */
PAPI_FCALL( papif_query_event, PAPIF_QUERY_EVENT,
			( int *EventCode, int *check ) )
{
	*check = PAPI_query_event( *EventCode );
}

/** @class PAPIF_query_named_event
 *	@ingroup PAPIF
 *  @brief Query if named PAPI event exists.
 *
 * @par Fortran Interface:
 * \#include "fpapi.h" @n
 * PAPIF_query_named_event(C_STRING EventName, C_INT check )
 *
 * @see PAPI_query_named_event
 */
#if defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL( papif_query_named_event, PAPIF_QUERY_NAMED_EVENT,
			( char *EventName, int *check, int Event_len ) )
{
	char tmp[PAPI_MAX_STR_LEN];
	Fortran2cstring( tmp, EventName, PAPI_MAX_STR_LEN, Event_len );
	*check = PAPI_query_named_event( tmp );	
}
#else
PAPI_FCALL( papif_query_named_event, PAPIF_QUERY_NAMED_EVENT,
			( char *EventName, int *check ) )
{
	*check = PAPI_query_named_event( EventName );
}
#endif

/** @class PAPIF_get_event_info
 *	@ingroup PAPIF
 *	@brief Get the event's name and description info.
 *
 * @par Fortran Interface:
 * \#include "fpapi.h" @n
 * PAPIF_get_event_info(C_INT EventCode, C_STRING symbol, C_STRING long_descr,
 						C_STRING short_descr, C_INT count, C_STRING event_note,
 						C_INT flags, C_INT check )
 *
 * @see PAPI_get_event_info
 */
#if defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL( papif_get_event_info, PAPIF_GET_EVENT_INFO,
			( int *EventCode, char *symbol, char *long_descr, char *short_descr,
			  int *count, char *event_note, int *flags, int *check,
			  int symbol_len, int long_descr_len, int short_descr_len,
			  int event_note_len ) )
#else
PAPI_FCALL( papif_get_event_info, PAPIF_GET_EVENT_INFO,
			( int *EventCode, char *symbol, char *long_descr, char *short_descr,
			  int *count, char *event_note, int *flags, int *check ) )
#endif
{
	PAPI_event_info_t info;
	( void ) flags;			 /*Unused */
#if defined(_FORTRAN_STRLEN_AT_END)
	int i;
	if ( ( *check = PAPI_get_event_info( *EventCode, &info ) ) == PAPI_OK ) {
		strncpy( symbol, info.symbol, ( size_t ) symbol_len );
		for ( i = ( int ) strlen( info.symbol ); i < symbol_len;
			  symbol[i++] = ' ' );
		strncpy( long_descr, info.long_descr, ( size_t ) long_descr_len );
		for ( i = ( int ) strlen( info.long_descr ); i < long_descr_len;
			  long_descr[i++] = ' ' );
		strncpy( short_descr, info.short_descr, ( size_t ) short_descr_len );
		for ( i = ( int ) strlen( info.short_descr ); i < short_descr_len;
			  short_descr[i++] = ' ' );

		*count = ( int ) info.count;

		int note_len=0;

		strncpy( event_note, info.note,  ( size_t ) event_note_len );
		note_len=strlen(info.note);

		for ( i =  note_len; i < event_note_len;
			  event_note[i++] = ' ' );
	}
#else
/* printf("EventCode: %d\n", *EventCode ); -KSL */
	if ( ( *check = PAPI_get_event_info( *EventCode, &info ) ) == PAPI_OK ) {
		strncpy( symbol, info.symbol, PAPI_MAX_STR_LEN );
		strncpy( long_descr, info.long_descr, PAPI_MAX_STR_LEN );
		strncpy( short_descr, info.short_descr, PAPI_MAX_STR_LEN );
		*count = info.count;
		if (info.note) 
                   strncpy( event_note, info.note, 
			    PAPI_MAX_STR_LEN );
	}
/*  printf("Check: %d\n", *check); -KSL */
#endif
}

/** @class PAPIF_event_code_to_name
 *	@ingroup PAPIF
 *	@brief Convert a numeric hardware event code to a name.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_event_code_to_name( C_INT EventCode, C_STRING EventName, C_INT check )
 *
 * @see PAPI_event_code_to_name
 */
#if defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL( papif_event_code_to_name, PAPIF_EVENT_CODE_TO_NAME,
			( int *EventCode, char *out_str, int *check, int out_len ) )
#else
PAPI_FCALL( papif_event_code_to_name, PAPIF_EVENT_CODE_TO_NAME,
			( int *EventCode, char *out, int *check ) )
#endif
{
#if defined(_FORTRAN_STRLEN_AT_END)
	char tmp[PAPI_MAX_STR_LEN];
	int i;
	*check = PAPI_event_code_to_name( *EventCode, tmp );
	/* tmp has \0 within PAPI_MAX_STR_LEN chars so strncpy is safe */
	strncpy( out_str, tmp, ( size_t ) out_len );
	/* overwrite any NULLs and trailing garbage in out_str */
	for ( i = ( int ) strlen( tmp ); i < out_len; out_str[i++] = ' ' );
#else
	/* The array "out" passed by the user must be sufficiently long */
	*check = PAPI_event_code_to_name( *EventCode, out );
#endif
}

/** @class PAPIF_event_name_to_code
 *	@ingroup PAPIF
 *	@brief Convert a name to a numeric hardware event code. 
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_event_name_to_code( C_STRING EventName, C_INT EventCode, C_INT check )
 *
 * @see PAPI_event_name_to_code
 */
#if defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL( papif_event_name_to_code, PAPIF_EVENT_NAME_TO_CODE,
			( char *in_str, int *out, int *check, int in_len ) )
#else
PAPI_FCALL( papif_event_name_to_code, PAPIF_EVENT_NAME_TO_CODE,
			( char *in, int *out, int *check ) )
#endif
{
#if defined(_FORTRAN_STRLEN_AT_END)
	int slen, i;
	char tmpin[PAPI_MAX_STR_LEN];

	/* What is the maximum number of chars to copy ? */
	slen = in_len < PAPI_MAX_STR_LEN ? in_len : PAPI_MAX_STR_LEN;
	strncpy( tmpin, in_str, ( size_t ) slen );

	/* Remove trailing blanks from initial Fortran string */
	for ( i = slen - 1; i > -1 && tmpin[i] == ' '; tmpin[i--] = '\0' );

	/* Make sure string is NULL terminated before call */
	tmpin[PAPI_MAX_STR_LEN - 1] = '\0';
	if ( slen < PAPI_MAX_STR_LEN )
		tmpin[slen] = '\0';

	*check = PAPI_event_name_to_code( tmpin, out );
#else
	/* This will have trouble if argument in is not null terminated */
	*check = PAPI_event_name_to_code( in, out );
#endif
}

/** @class PAPIF_num_events
 *	@ingroup PAPIF
 *	@brief Enumerate PAPI preset or native events. 
 *
 * @par Fortran Interface:
 * \#include "fpapi.h" @n
 * PAPIF_num_events(C_INT  EventSet,  C_INT  count)
 *
 * @see PAPI_num_events
 */
PAPI_FCALL( papif_num_events, PAPIF_NUM_EVENTS, ( int *EventCode, int *count ) )
{
	*count = PAPI_num_events( *EventCode );
}

/** @class PAPIF_enum_event
 *	@ingroup PAPIF
 *  @brief Return the number of events in an event set.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_enum_event( C_INT  EventCode,  C_INT  modifier,  C_INT  check )
 *
 * @see PAPI_enum_event
 */
PAPI_FCALL( papif_enum_event, PAPIF_ENUM_EVENT,
			( int *EventCode, int *modifier, int *check ) )
{
	*check = PAPI_enum_event( EventCode, *modifier );
}

/** @class PAPIF_read
 *	@ingroup PAPIF
 *  @brief Read hardware counters from an event set.
 *
 *  @par Fortran Interface:
 * \#include "fpapi.h" @n
 *  PAPIF_read(C_INT EventSet, C_LONG_LONG(*) values, C_INT check )
 *
 * @see PAPI_read
 */
PAPI_FCALL( papif_read, PAPIF_READ,
			( int *EventSet, long long *values, int *check ) )
{
	*check = PAPI_read( *EventSet, values );
}

/** @class PAPIF_read_ts
 *	@ingroup PAPIF
 *  @brief Read hardware counters with a timestamp.
 *
 *  @par Fortran Interface:
 *  \#include "fpapi.h" @n
 *  PAPIF_read_ts(C_INT EventSet, C_LONG_LONG(*) values, C_LONG_LONG(*) cycles, C_INT check)
 *
 * @see PAPI_read_ts
 */
PAPI_FCALL( papif_read_ts, PAPIF_READ_TS,
			( int *EventSet, long long *values, long long *cycles, int *check ) )
{
	*check = PAPI_read_ts( *EventSet, values, cycles );
}

/** @class PAPIF_remove_event
 *	@ingroup PAPIF
 *  @brief Remove a hardware event from a PAPI event set. 
 *
 *   @par Fortran interface:
 *   \#include "fpapi.h" @n
 *   PAPIF_remove_event( C_INT EventSet, C_INT EventCode, C_INT check )
 *
 * @see PAPI_remove_event
 */
PAPI_FCALL( papif_remove_event, PAPIF_REMOVE_EVENT,
			( int *EventSet, int *Event, int *check ) )
{
	*check = PAPI_remove_event( *EventSet, *Event );
}

/** @class PAPIF_remove_named_event
 *	@ingroup PAPIF
 *  @brief Remove a named hardware event from a PAPI event set. 
 *
 *   @par Fortran interface:
 *   \#include "fpapi.h" @n
 *   PAPIF_remove_named_event( C_INT EventSet, C_STRING EventName, C_INT check )
 *
 * @see PAPI_remove_named_event
 */
#if defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL( papif_remove_named_event, PAPIF_REMOVE_NAMED_EVENT,
			( int *EventSet, char *EventName, int *check, int Event_len ) )
{
	char tmp[PAPI_MAX_STR_LEN];
	Fortran2cstring( tmp, EventName, PAPI_MAX_STR_LEN, Event_len );
	*check = PAPI_remove_named_event( *EventSet, tmp );	
}
#else
PAPI_FCALL( papif_remove_named_event, PAPIF_REMOVE_NAMED_EVENT,
			( int *EventSet, char *EventName, int *check ) )
{
	*check = PAPI_remove_named_event( *EventSet, EventName );
}
#endif

/** @class PAPIF_remove_events
 *	@ingroup PAPIF
 * @brief Remove an array of hardware event codes from a PAPI event set.
 *
 *	@par Fortran Prototype:
 *		\#include "fpapi.h" @n
 *		PAPIF_remove_events( C_INT EventSet, C_INT(*) EventCode, C_INT number, C_INT check )
 * 
 * @see PAPI_remove_events
 */
PAPI_FCALL( papif_remove_events, PAPIF_REMOVE_EVENTS,
			( int *EventSet, int *Events, int *number, int *check ) )
{
	*check = PAPI_remove_events( *EventSet, Events, *number );
}

/** @class PAPIF_reset
 *	@ingroup PAPIF
 *  @brief Reset the hardware event counts in an event set.
 *
 *	@par Fortran Prototype:
 *		\#include "fpapi.h" @n
 *		PAPIF_reset( C_INT EventSet, C_INT check )
 *
 * @see PAPI_reset
 */
PAPI_FCALL( papif_reset, PAPIF_RESET, ( int *EventSet, int *check ) )
{
	*check = PAPI_reset( *EventSet );
}

/** @class PAPIF_set_debug
 *	@ingroup PAPIF
 * @brief Set the current debug level for error output from PAPI.
 *
 * @par Fortran Prototype:
 *		\#include "fpapi.h" @n
 *		PAPIF_set_debug( C_INT level, C_INT check )
 * 
 * @see PAPI_set_debug
 */
PAPI_FCALL( papif_set_debug, PAPIF_SET_DEBUG, ( int *debug, int *check ) )
{
	*check = PAPI_set_debug( *debug );
}

/** @class PAPIF_set_domain
 *	@ingroup PAPIF
 *	@brief Set the default counting domain for new event sets bound to the cpu component.
 *
 *	@par Fortran Prototype:
 *		\#include "fpapi.h" @n
 *		PAPIF_set_domain( C_INT domain, C_INT check )
 *
 * @see PAPI_set_domain
 */
PAPI_FCALL( papif_set_domain, PAPIF_SET_DOMAIN, ( int *domain, int *check ) )
{
	*check = PAPI_set_domain( *domain );
}

/** @class PAPIF_set_cmp_domain
 *	@ingroup PAPIF
 *	@brief Set the default counting domain for new event sets bound to the specified component.
 *
 *	@par Fortran Prototype:
 *		\#include "fpapi.h" @n
 *		PAPIF_set_cmp_domain( C_INT domain, C_INT cidx, C_INT check )
 *
 * @see PAPI_set_cmp_domain
 */
PAPI_FCALL( papif_set_cmp_domain, PAPIF_SET_CMP_DOMAIN,
			( int *domain, int *cidx, int *check ) )
{
	*check = PAPI_set_cmp_domain( *domain, *cidx );
}

/** @class PAPIF_set_granularity
 *	@ingroup PAPIF
 *	@brief Set the default counting granularity for eventsets bound to the cpu component.
 *
 *	@par Fortran Prototype:
 *		\#include "fpapi.h" @n
 *		PAPIF_set_granularity( C_INT granularity, C_INT check )
 *
 * @see PAPI_set_granularity
 */
PAPI_FCALL( papif_set_granularity, PAPIF_SET_GRANULARITY,
			( int *granularity, int *check ) )
{
	*check = PAPI_set_granularity( *granularity );
}

/** @class PAPIF_set_cmp_granularity
 *	@ingroup PAPIF
 *	@brief Set the default counting granularity for eventsets bound to the specified component.
 *
 *	@par Fortran Prototype:
 *		\#include "fpapi.h" @n
 *		PAPIF_set_cmp_granularity( C_INT granularity, C_INT cidx, C_INT check )
 *
 * @see PAPI_set_cmp_granularity
 */
PAPI_FCALL( papif_set_cmp_granularity, PAPIF_SET_CMP_GRANULARITY,
			( int *granularity, int *cidx, int *check ) )
{
	*check = PAPI_set_cmp_granularity( *granularity, *cidx );
}

/** @class PAPIF_shutdown
 *	@ingroup PAPIF
 *	@brief finish using PAPI and free all related resources. 
 *
 *	@par Fortran Prototype:
 *		\#include "fpapi.h" @n
 *		PAPIF_shutdown( )
 *
 * @see PAPI_shutdown
 */
PAPI_FCALL( papif_shutdown, PAPIF_SHUTDOWN, ( void ) )
{
	PAPI_shutdown(  );
}

/** @class PAPIF_start
 *	@ingroup PAPIF
 *	@brief Start counting hardware events in an event set.
 *
 * @par Fortran Interface:
 *     \#include "fpapi.h" @n
 *     PAPIF_start( C_INT EventSet, C_INT check )
 *
 * @see PAPI_start
 */
PAPI_FCALL( papif_start, PAPIF_START, ( int *EventSet, int *check ) )
{
	*check = PAPI_start( *EventSet );
}

/** @class PAPIF_state
 *	@ingroup PAPIF
 * @brief Return the counting state of an EventSet.
 *
 * @par Fortran Interface:
 *     \#include "fpapi.h" @n
 *     PAPIF_state(C_INT EventSet, C_INT status, C_INT check )
 *
 * @see PAPI_state
 */
PAPI_FCALL( papif_state, PAPIF_STATE,
			( int *EventSet, int *status, int *check ) )
{
	*check = PAPI_state( *EventSet, status );
}

/** @class PAPIF_stop
 *	@ingroup PAPIF
 *	@brief Stop counting hardware events in an EventSet.
 *
 * @par Fortran Interface:
 *     \#include "fpapi.h" @n
 *     PAPIF_stop( C_INT EventSet, C_LONG_LONG(*) values, C_INT check )
 *
 * @see PAPI_stop
 */
PAPI_FCALL( papif_stop, PAPIF_STOP,
			( int *EventSet, long long *values, int *check ) )
{
	*check = PAPI_stop( *EventSet, values );
}

/** @class PAPIF_write
 *	@ingroup PAPIF
 *	@brief Write counter values into counters.
 *
 * @par Fortran Interface:
 *     \#include "fpapi.h" @n
 *     PAPIF_write( C_INT EventSet, C_LONG_LONG(*) values, C_INT check )
 *
 * @see PAPI_write
 */
PAPI_FCALL( papif_write, PAPIF_WRITE,
			( int *EventSet, long long *values, int *check ) )
{
	*check = PAPI_write( *EventSet, values );
}

/** @class PAPIF_lock
 *	@ingroup PAPIF
 *  @brief Lock one of two mutex variables defined in papi.h.
 *
 *  @par Fortran Interface:
 *  \#include "fpapi.h" @n
 *  PAPIF_lock( C_INT lock )
 *
 * @see PAPI_lock
 */
PAPI_FCALL( papif_lock, PAPIF_LOCK,
			( int *lock, int *check ) )
{
	*check = PAPI_lock( *lock );
}

/** @class PAPIF_unlock
 *	@ingroup PAPIF
 *	@brief Unlock one of the mutex variables defined in papi.h.
 *
 * @par Fortran Interface:
 *     \#include "fpapi.h" @n
 *  PAPIF_unlock( C_INT lock )
 *
 * @see PAPI_unlock
 */
PAPI_FCALL( papif_unlock, PAPIF_unlock,
			( int *lock, int *check ) )
{
	*check = PAPI_unlock( *lock );
}


/* Fortran only APIs for get_opt and set_opt functionality */

/** @class PAPIF_get_clockrate
 *	@ingroup PAPIF
 *	@brief Get the clockrate in MHz for the current cpu.
 *
 *	@par Fortran Prototype:
 *		\#include "fpapi.h" @n
 *		PAPIF_set_domain( C_INT cr )
 *
 * @note This is a Fortran only interface that returns a value from the PAPI_get_opt call.
 *
 * @see PAPI_get_opt
 */
PAPI_FCALL( papif_get_clockrate, PAPIF_GET_CLOCKRATE, ( int *cr ) )
{
	*cr = PAPI_get_opt( PAPI_CLOCKRATE, (PAPI_option_t *)  NULL );
}

/** @class PAPIF_get_preload
 *	@ingroup PAPIF
 *	@brief Get the LD_PRELOAD environment variable.
 *
 *	@par Fortran Prototype:
 *		\#include "fpapi.h" @n
 *		PAPIF_get_preload( C_STRING lib_preload_env, C_INT check )
 *
 * @note This is a Fortran only interface that returns a value from the PAPI_get_opt call.
 *
 * @see PAPI_get_opt
 */
#if defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL( papif_get_preload, PAPIF_GET_PRELOAD,
			( char *lib_preload_env, int *check, int lib_preload_env_len ) )
#else
PAPI_FCALL( papif_get_preload, PAPIF_GET_PRELOAD,
			( char *lib_preload_env, int *check ) )
#endif
{
	PAPI_option_t p;
#if defined(_FORTRAN_STRLEN_AT_END)
	int i;

	if ( ( *check = PAPI_get_opt( PAPI_PRELOAD, &p ) ) == PAPI_OK ) {
		strncpy( lib_preload_env, p.preload.lib_preload_env,
				 ( size_t ) lib_preload_env_len );
		for ( i = ( int ) strlen( p.preload.lib_preload_env );
			  i < lib_preload_env_len; lib_preload_env[i++] = ' ' );
	}
#else
	if ( ( *check = PAPI_get_opt( PAPI_PRELOAD, &p ) ) == PAPI_OK ) {
		strncpy( lib_preload_env, p.preload.lib_preload_env, PAPI_MAX_STR_LEN );
	}
#endif
}

/** @class PAPIF_get_granularity
 *	@ingroup PAPIF
 *	@brief Get the granularity setting for the specified EventSet.
 *
 *	@par Fortran Prototype:
 *		\#include "fpapi.h" @n
 *		PAPIF_get_granularity( C_INT eventset, C_INT granularity, C_INT mode, C_INT check )
 *
 * @see PAPI_get_opt
 */
PAPI_FCALL( papif_get_granularity, PAPIF_GET_GRANULARITY,
			( int *eventset, int *granularity, int *mode, int *check ) )
{
	PAPI_option_t g;

	if ( *mode == PAPI_DEFGRN ) {
		*granularity = PAPI_get_opt( *mode, &g );
		*check = PAPI_OK;
	} else if ( *mode == PAPI_GRANUL ) {
		g.granularity.eventset = *eventset;
		if ( ( *check = PAPI_get_opt( *mode, &g ) ) == PAPI_OK ) {
			*granularity = g.granularity.granularity;
		}
	} else {
		*check = PAPI_EINVAL;
	}
}

/** @class PAPIF_get_domain
 *	@ingroup PAPIF
 *	@brief Get the domain setting for the specified EventSet.
 *
 *	@par Fortran Prototype:
 *		\#include "fpapi.h" @n
 *		PAPIF_get_domain( C_INT eventset, C_INT domain, C_INT mode, C_INT check )
 *
 * @see PAPI_get_opt
 */
PAPI_FCALL( papif_get_domain, PAPIF_GET_DOMAIN,
			( int *eventset, int *domain, int *mode, int *check ) )
{
	PAPI_option_t d;

	if ( *mode == PAPI_DEFDOM ) {
		*domain = PAPI_get_opt( *mode, (PAPI_option_t *) NULL );
		*check = PAPI_OK;
	} else if ( *mode == PAPI_DOMAIN ) {
		d.domain.eventset = *eventset;
		if ( ( *check = PAPI_get_opt( *mode, &d ) ) == PAPI_OK ) {
			*domain = d.domain.domain;
		}
	} else {
		*check = PAPI_EINVAL;
	}
}

#if 0
PAPI_FCALL( papif_get_inherit, PAPIF_GET_INHERIT, ( int *inherit, int *check ) )
{
	PAPI_option_t i;

	if ( ( *check = PAPI_get_opt( PAPI_INHERIT, &i ) ) == PAPI_OK ) {
		*inherit = i.inherit.inherit;
	}
}
#endif

/** @class PAPIF_set_event_domain
 *	@ingroup PAPIF
 *	@brief Set the default counting domain for specified EventSet.
 *
 *	@par Fortran Prototype:
 *		\#include "fpapi.h" @n
 *		PAPIF_set_event_domain( C_INT EventSet, C_INT domain, C_INT check )
 *
 * @see PAPI_set_domain
 * @see PAPI_set_opt
 */
PAPI_FCALL( papif_set_event_domain, PAPIF_SET_EVENT_DOMAIN,
			( int *es, int *domain, int *check ) )
{
	PAPI_option_t d;

	d.domain.domain = *domain;
	d.domain.eventset = *es;
	*check = PAPI_set_opt( PAPI_DOMAIN, &d );
}

/** @class PAPIF_set_inherit
 *	@ingroup PAPIF
 *	@brief Turn on inheriting of counts from daughter to parent process.
 *
 *	@par Fortran Prototype:
 *		\#include "fpapi.h" @n
 *		PAPIF_set_inherit( C_INT inherit, C_INT check )
 *
 * @see PAPI_set_opt
 */
PAPI_FCALL( papif_set_inherit, PAPIF_SET_INHERIT, ( int *inherit, int *check ) )
{
	PAPI_option_t i;

	i.inherit.inherit = *inherit;
	*check = PAPI_set_opt( PAPI_INHERIT, &i );
}

/** @class PAPIF_ipc
 *	@ingroup PAPIF
 *	@brief Get instructions per cycle, real and processor time.
 *	
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_ipc( C_FLOAT real_time, C_FLOAT proc_time, C_LONG_LONG ins, C_FLOAT ipc, C_INT check )
 *
 * @see PAPI_ipc
 */
PAPI_FCALL( papif_ipc, PAPIF_IPC,
			( float *rtime, float *ptime, long long *ins, float *ipc,
			  int *check ) )
{
	*check = PAPI_ipc( rtime, ptime, ins, ipc );
}

/** @class PAPIF_epc
 *	@ingroup PAPIF
 *	@brief Get named events per cycle, real and processor time, reference and core cycles.
 *	
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_epc( C_INT EventCode, C_FLOAT real_time, C_FLOAT proc_time, C_LONG_LONG ref, C_LONG_LONG core, C_LONG_LONG evt, C_FLOAT epc, C_INT check )
 *
 * @see PAPI_epc
 */
PAPI_FCALL( papif_epc, PAPIF_EPC,
			( int *EventCode, float *rtime, float *ptime, 
			  long long *ref, long long *core, long long *evt, float *epc,
			  int *check) )
{
	*check = PAPI_epc( *EventCode, rtime, ptime, ref, core, evt, epc );
}

/** @class PAPIF_flips_rate
 *	@ingroup PAPIF
 *	@brief Simplified call to get Mflips/s (floating point instruction rate), real and processor time. 
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *	PAPIF_flips_rate ( C_INT EventCode, C_FLOAT real_time, C_FLOAT proc_time, C_LONG_LONG flpins, C_FLOAT mflips, C_INT check )
 *
 * @see PAPI_flips_rate
 */
PAPI_FCALL( papif_flips_rate, PAPIF_FLIPS_RATE,
			( int *EventCode, float *real_time, float *proc_time, long long *flpins,
			  float *mflips, int *check ) )
{
	*check = PAPI_flips_rate( *EventCode, real_time, proc_time, flpins, mflips );
}

/** @class PAPIF_flops_rate
 *	@ingroup PAPIF
 *	@brief Simplified call to get Mflops/s (floating point instruction rate), real and processor time. 
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *  PAPIF_flops_rate( C_INT EventCode, C_FLOAT real_time, C_FLOAT proc_time, C_LONG_LONG flpops, C_FLOAT mflops, C_INT check )
 *
 * @see PAPI_flops_rate
 */
PAPI_FCALL( papif_flops_rate, PAPIF_FLOPS_RATE,
			( int *EventCode, float *real_time, float *proc_time, long long *flpops,
			  float *mflops, int *check ) )
{
	*check = PAPI_flops_rate( *EventCode, real_time, proc_time, flpops, mflops );
}

/** @class PAPIF_rate_stop
 *	@ingroup PAPIF
 *	@brief Stop a running event set of a rate function.
 *
 *	@par Fortran Interface:
 *	\#include "fpapi.h" @n
 *  PAPIF_rate_stop( C_INT check )
 *
 * @see PAPI_rate_stop
 */
PAPI_FCALL( papif_rate_stop, PAPIF_RATE_STOP,
			( int *check ) )
{
	*check = PAPI_rate_stop( );
}

static void *sysdetect_fort_handle;

/** @class PAPIF_enum_dev_type
 * @ingroup PAPIF
 * @brief returns handle of next device type
 *
 * @par Fortran Interface:
 * \#include "fpapi.h" @n
 * PAPIF_enum_dev_type( C_INT modifier, C_INT handle_index, C_INT check )
 *
 * @see PAPI_enum_dev_type
 */
PAPI_FCALL( papif_enum_dev_type, PAPIF_ENUM_DEV_TYPE, ( int *modifier, int *handle_index, int *check ))
{
    *check = PAPI_enum_dev_type(*modifier, &sysdetect_fort_handle);
    *handle_index = 0;
}

/** @class PAPIF_get_dev_type_attr
 * @ingroup PAPIF
 * @brief returns device type attributes
 *
 * @par Fortran Interface:
 * \#include "fpapi.h" @n
 * PAPIF_get_dev_type_attr( C_INT handle_index, C_INT attribute, @n
 *                          C_INT value, C_STRING string, C_INT check )
 *
 * @see PAPIF_get_dev_type_attr
 */
#if defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL( papif_get_dev_type_attr, PAPIF_GET_DEV_TYPE_ATTR, (int *handle_index,
                                                               int *attribute,
                                                               int *value,
                                                               char *string,
                                                               int *check,
                                                               int string_len) )
#else
PAPI_FCALL( papif_get_dev_type_attr, PAPIF_GET_DEV_TYPE_ATTR, (int *handle_index,
                                                               int *attribute,
                                                               int *value,
                                                               char *string,
                                                               int *check) )
#endif
{
    const char *string_ptr;
    int i;
    *handle_index = 0;
    *check = PAPI_OK;

    assert(sysdetect_fort_handle);

    switch(*attribute) {
        case PAPI_DEV_TYPE_ATTR__INT_PAPI_ID:
        case PAPI_DEV_TYPE_ATTR__INT_VENDOR_ID:
        case PAPI_DEV_TYPE_ATTR__INT_COUNT:
            *check = PAPI_get_dev_type_attr(sysdetect_fort_handle, *attribute,
                                            value);
            break;
        case PAPI_DEV_TYPE_ATTR__CHAR_NAME:
        case PAPI_DEV_TYPE_ATTR__CHAR_STATUS:
            *check = PAPI_get_dev_type_attr(sysdetect_fort_handle, *attribute,
                                            &string_ptr);
            if (*check != PAPI_OK) {
                break;
            }
#if defined(_FORTRAN_STRLEN_AT_END)
            strncpy(string, string_ptr, string_len);
            for ( i = ( int ) string_len; i < PAPI_MAX_STR_LEN;
                string[i++] = ' ' );
#else
            strcpy(string, string_ptr);
            for ( i = ( int ) strlen(string_ptr); i < PAPI_MAX_STR_LEN;
                string[i++] = ' ' );
#endif
            break;
        default:
            *check = PAPI_EINVAL;
    }
    *handle_index = 0;
    return;
}

/** @class PAPIF_get_dev_attr
 * @ingroup PAPIF
 * @brief returns device attributes
 *
 * @par Fortran Interface:
 * \#include "fpapi.h" @n
 * PAPIF_get_dev_attr( C_INT handle, C_INT id, C_INT attribute, @n
 *                     C_INT value, C_STRING string, C_INT check )
 *
 * @see PAPIF_get_dev_attr
 */
#if defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL( papif_get_dev_attr, PAPIF_GET_DEV_ATTR, (int *handle_index,
                                                     int *id,
                                                     int *attribute,
                                                     int *value,
                                                     char *string,
                                                     int *check,
                                                     int string_len) )
#else
PAPI_FCALL( papif_get_dev_attr, PAPIF_GET_DEV_ATTR, (int *handle_index,
                                                     int *id,
                                                     int *attribute,
                                                     int *value,
                                                     char *string,
                                                     int *check) )
#endif
{
    int i;
    const char *string_ptr;
    *handle_index = 0;
    *check = PAPI_OK;

    assert(sysdetect_fort_handle);

    switch(*attribute) {
        case PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_SIZE:
        case PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_SIZE:
        case PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_SIZE:
        case PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_SIZE:
        case PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_LINE_SIZE:
        case PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_LINE_SIZE:
        case PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_LINE_SIZE:
        case PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_LINE_SIZE:
        case PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_LINE_COUNT:
        case PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_LINE_COUNT:
        case PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_LINE_COUNT:
        case PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_LINE_COUNT:
        case PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_ASSOC:
        case PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_ASSOC:
        case PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_ASSOC:
        case PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_ASSOC:
        case PAPI_DEV_ATTR__CPU_UINT_FAMILY:
        case PAPI_DEV_ATTR__CPU_UINT_MODEL:
        case PAPI_DEV_ATTR__CPU_UINT_STEPPING:
        case PAPI_DEV_ATTR__CPU_UINT_SOCKET_COUNT:
        case PAPI_DEV_ATTR__CPU_UINT_NUMA_COUNT:
        case PAPI_DEV_ATTR__CPU_UINT_CORE_COUNT:
        case PAPI_DEV_ATTR__CPU_UINT_THREAD_COUNT:
        case PAPI_DEV_ATTR__CPU_UINT_THR_PER_NUMA:
        case PAPI_DEV_ATTR__CUDA_ULONG_UID:
        case PAPI_DEV_ATTR__CUDA_CHAR_DEVICE_NAME:
        case PAPI_DEV_ATTR__CUDA_UINT_WARP_SIZE:
        case PAPI_DEV_ATTR__CUDA_UINT_THR_PER_BLK:
        case PAPI_DEV_ATTR__CUDA_UINT_BLK_PER_SM:
        case PAPI_DEV_ATTR__CUDA_UINT_SHM_PER_BLK:
        case PAPI_DEV_ATTR__CUDA_UINT_SHM_PER_SM:
        case PAPI_DEV_ATTR__CUDA_UINT_BLK_DIM_X:
        case PAPI_DEV_ATTR__CUDA_UINT_BLK_DIM_Y:
        case PAPI_DEV_ATTR__CUDA_UINT_BLK_DIM_Z:
        case PAPI_DEV_ATTR__CUDA_UINT_GRD_DIM_X:
        case PAPI_DEV_ATTR__CUDA_UINT_GRD_DIM_Y:
        case PAPI_DEV_ATTR__CUDA_UINT_GRD_DIM_Z:
        case PAPI_DEV_ATTR__CUDA_UINT_SM_COUNT:
        case PAPI_DEV_ATTR__CUDA_UINT_MULTI_KERNEL:
        case PAPI_DEV_ATTR__CUDA_UINT_MAP_HOST_MEM:
        case PAPI_DEV_ATTR__CUDA_UINT_MEMCPY_OVERLAP:
        case PAPI_DEV_ATTR__CUDA_UINT_UNIFIED_ADDR:
        case PAPI_DEV_ATTR__CUDA_UINT_MANAGED_MEM:
        case PAPI_DEV_ATTR__CUDA_UINT_COMP_CAP_MAJOR:
        case PAPI_DEV_ATTR__CUDA_UINT_COMP_CAP_MINOR:
        case PAPI_DEV_ATTR__ROCM_ULONG_UID:
        case PAPI_DEV_ATTR__ROCM_UINT_SIMD_PER_CU:
        case PAPI_DEV_ATTR__ROCM_UINT_WORKGROUP_SIZE:
        case PAPI_DEV_ATTR__ROCM_UINT_WAVEFRONT_SIZE:
        case PAPI_DEV_ATTR__ROCM_UINT_WAVE_PER_CU:
        case PAPI_DEV_ATTR__ROCM_UINT_SHM_PER_WG:
        case PAPI_DEV_ATTR__ROCM_UINT_WG_DIM_X:
        case PAPI_DEV_ATTR__ROCM_UINT_WG_DIM_Y:
        case PAPI_DEV_ATTR__ROCM_UINT_WG_DIM_Z:
        case PAPI_DEV_ATTR__ROCM_UINT_GRD_DIM_X:
        case PAPI_DEV_ATTR__ROCM_UINT_GRD_DIM_Y:
        case PAPI_DEV_ATTR__ROCM_UINT_GRD_DIM_Z:
        case PAPI_DEV_ATTR__ROCM_UINT_CU_COUNT:
        case PAPI_DEV_ATTR__ROCM_UINT_COMP_CAP_MAJOR:
        case PAPI_DEV_ATTR__ROCM_UINT_COMP_CAP_MINOR:
            *check = PAPI_get_dev_attr(sysdetect_fort_handle, *id, *attribute,
                                       value);
            break;
        case PAPI_DEV_ATTR__CPU_CHAR_NAME:
        case PAPI_DEV_ATTR__ROCM_CHAR_DEVICE_NAME:
            *check = PAPI_get_dev_attr(sysdetect_fort_handle, *id, *attribute,
                                       &string_ptr);
            if (*check != PAPI_OK) {
                break;
            }
#if defined(_FORTRAN_STRLEN_AT_END)
            strncpy(string, string_ptr, (size_t) string_len);
            for ( i = ( int ) strlen(string_ptr); i < PAPI_MAX_STR_LEN;
                string[i++] = ' ' );
#else
            strcpy(string, string_ptr);
            for ( i = ( int ) strlen(string_ptr); i < PAPI_MAX_STR_LEN;
                string[i++] = ' ' );
#endif
            break;
        default:
            *check = PAPI_EINVAL;
    }
    return;
}

/* The High Level API Wrappers */
/** \internal @defgroup PAPIF-HL PAPI Fortran High Level API */

/** @class PAPIf_hl_region_begin
 * @ingroup PAPIF-HL
 * @brief Reads and stores hardware events at the beginning of an instrumented code region.
 *
 * @par Fortran Prototype:
 * \#include "fpapi.h" @n
 * PAPIf_hl_region_begin( C_STRING region, C_INT check )
 *
 * @retval PAPI_OK
 * @retval PAPI_ENOTRUN
 * -- EventSet is currently not running or could not determined.
 * @retval PAPI_ESYS
 * -- A system or C library call failed inside PAPI, see the errno variable.
 * @retval PAPI_EMISC
 * -- PAPI has been deactivated due to previous erros.
 * @retval PAPI_ENOMEM
 * -- Insufficient memory.
 *
 * PAPIf_hl_region_begin reads hardware events and stores them internally at the beginning
 * of an instrumented code region.
 * If not specified via environment variable PAPI_EVENTS, default events are used.
 * The first call sets all counters implicitly to zero and starts counting.
 * Note that if PAPI_EVENTS is not set or cannot be interpreted, default hardware events are
 * recorded.
 *
 * @par Example:
 *
 * @code
 * export PAPI_EVENTS="PAPI_TOT_INS,PAPI_TOT_CYC"
 * @endcode
 *
 *
 * @code
 * integer retval
 *
 * call PAPIf_hl_region_begin("computation", retval)
 * if ( retval .NE. PAPI_OK ) then
 *      write (*,*) "PAPIf_hl_region_begin failed!"
 * end if
 *
 * !do some computation here
 *
 * call PAPIf_hl_region_end("computation", retval)
 * if ( retval .NE. PAPI_OK ) then
 *     write (*,*) "PAPIf_hl_region_end failed!"
 * end if
 *
 * @endcode
 *
 * @see PAPI_hl_region_begin
 */
#if defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL( papif_hl_region_begin, PAPIF_HL_REGION_BEGIN,
			( char* name, int *check, int Event_len ) )
{
	char tmp[PAPI_MAX_STR_LEN];
	Fortran2cstring( tmp, name, PAPI_MAX_STR_LEN, Event_len );
	*check = PAPI_hl_region_begin( tmp );
}
#else
PAPI_FCALL( papif_hl_region_begin, PAPIF_HL_REGION_BEGIN,
			( char* name, int *check ) )
{
	*check = PAPI_hl_region_begin( name );
}
#endif

/** @class PAPIf_hl_read
 * @ingroup PAPIF-HL
 * @brief Reads and stores hardware events inside of an instrumented code region.
 *
 * @par Fortran Prototype:
 * \#include <fpapi.h> @n
 * int PAPIf_hl_read( C_STRING region, C_INT check )
 *
 * @param region
 * -- a unique region name corresponding to PAPIf_hl_region_begin
 *
 * @retval PAPI_OK
 * @retval PAPI_ENOTRUN
 * -- EventSet is currently not running or could not determined.
 * @retval PAPI_ESYS
 * -- A system or C library call failed inside PAPI, see the errno variable.
 * @retval PAPI_EMISC
 * -- PAPI has been deactivated due to previous erros.
 * @retval PAPI_ENOMEM
 * -- Insufficient memory.
 *
 * PAPIf_hl_read reads hardware events and stores them internally inside
 * of an instrumented code region.
 * Assumes that PAPIf_hl_region_begin was called before.
 *
 * @par Example:
 *
 * @code
 * integer retval
 *
 * call PAPIf_hl_region_begin("computation", retval)
 * if ( retval .NE. PAPI_OK ) then
 *     write (*,*) "PAPIf_hl_region_begin failed!"
 * end if
 *
 * !do some computation here
 *
 * call PAPIf_hl_read("computation", retval)
 * if ( retval .NE. PAPI_OK ) then
 *     write (*,*) "PAPIf_hl_read failed!"
 * end if
 *
 * !do some computation here
 *
 * call PAPIf_hl_region_end("computation", retval)
 * if ( retval .NE. PAPI_OK ) then
 *     write (*,*) "PAPIf_hl_region_end failed!"
 * end if
 *
 * @endcode
 *
 * @see PAPI_hl_read
 */
#if defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL( papif_hl_read, PAPIF_HL_READ,
			( char* name, int *check, int Event_len ) )
{
	char tmp[PAPI_MAX_STR_LEN];
	Fortran2cstring( tmp, name, PAPI_MAX_STR_LEN, Event_len );
	*check = PAPI_hl_read( tmp );
}
#else
PAPI_FCALL( papif_hl_read, PAPIF_HL_READ,
			( char* name, int *check ) )
{
	*check = PAPI_hl_read( name );
}
#endif

/** @class PAPIf_hl_region_end
 * @ingroup PAPIF-HL
 * @brief Reads and stores hardware events at the end of an instrumented code region.
 *
 * @par Fortran Prototype:
 * \#include "fpapi.h" @n
 * PAPIf_hl_region_end( C_STRING region, C_INT check )
 *
 * @param region
 * -- a unique region name corresponding to PAPIf_hl_region_begin
 *
 * @retval PAPI_OK
 * @retval PAPI_ENOTRUN
 * -- EventSet is currently not running or could not determined.
 * @retval PAPI_ESYS
 * -- A system or C library call failed inside PAPI, see the errno variable.
 * @retval PAPI_EMISC
 * -- PAPI has been deactivated due to previous erros.
 * @retval PAPI_ENOMEM
 * -- Insufficient memory.
 *
 * PAPIf_hl_region_end reads hardware events and stores the difference to the values from
 * PAPIf_hl_region_begin at the end of an instrumented code region.
 * Assumes that PAPIf_hl_region_begin was called before.
 * Note that an output is automatically generated when your application terminates.
 *
 * @par Example:
 *
 * @code
 * integer retval
 *
 * call PAPIf_hl_region_begin("computation", retval)
 * if ( retval .NE. PAPI_OK ) then
 *     write (*,*) "PAPIf_hl_region_begin failed!"
 * end if
 *
 * !do some computation here
 *
 * call PAPIf_hl_region_end("computation", retval)
 * if ( retval .NE. PAPI_OK ) then
 *     write (*,*) "PAPIf_hl_region_end failed!"
 * end if
 *
 * @endcode
 *
 * @see PAPI_hl_region_end
 */
#if defined(_FORTRAN_STRLEN_AT_END)
PAPI_FCALL( papif_hl_region_end, PAPIF_HL_REGION_END,
			( char* name, int *check, int Event_len ) )
{
	char tmp[PAPI_MAX_STR_LEN];
	Fortran2cstring( tmp, name, PAPI_MAX_STR_LEN, Event_len );
	*check = PAPI_hl_region_end( tmp );
}
#else
PAPI_FCALL( papif_hl_region_end, PAPIF_HL_REGION_END,
			( char* name, int *check ) )
{
	*check = PAPI_hl_region_end( name );
}
#endif

/** @class PAPIf_hl_stop
 * @ingroup PAPIF-HL
 * @brief Stop a running high-level event set.
 *
 * @par Fortran Prototype:
 * \#include "fpapi.h" @n
 * PAPIf_hl_stop( C_INT check )
 *
 * @retval PAPI_ENOEVNT
 * -- The EventSet is not started yet.
 * @retval PAPI_ENOMEM
 * -- Insufficient memory to complete the operation.
 *
 * PAPIf_hl_stop stops a running high-level event set.
 *
 * This call is optional and only necessary if the programmer wants to use the low-level API in addition
 * to the high-level API. It should be noted that PAPIf_hl_stop and low-level calls are not
 * allowed inside of a marked region. Furthermore, PAPIf_hl_stop is thread-local and therefore
 * has to be called in the same thread as the corresponding marked region.
 *
 * @par Example:
 *
 * @code
 * integer retval
 *
 * call PAPIf_hl_region_begin("computation", retval)
 * if ( retval .NE. PAPI_OK ) then
 *     write (*,*) "PAPIf_hl_region_begin failed!"
 * end if
 *
 * !do some computation here
 *
 * call PAPIf_hl_region_end("computation", retval)
 * if ( retval .NE. PAPI_OK ) then
 *     write (*,*) "PAPIf_hl_region_end failed!"
 * end if
 *
 * call PAPIf_hl_stop(retval)
 * if ( retval .NE. PAPI_OK ) then
 *     write (*,*) "PAPIf_hl_stop failed!"
 * end if
 *
 * @endcode
 *
 * @see PAPI_hl_stop
 */
PAPI_FCALL( papif_hl_stop, PAPIF_HL_STOP,
			( int *check ) )
{
	*check = PAPI_hl_stop( );
}

#pragma GCC visibility pop
