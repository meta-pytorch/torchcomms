/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/*
* File:    papi_internal.c
*
* Author:  Philip Mucci
*          mucci@cs.utk.edu
* Mods:    dan terpstra
*          terpstra@cs.utk.edu
* Mods:    Min Zhou
*          min@cs.utk.edu
* Mods:    Kevin London
*	   london@cs.utk.edu
* Mods:    Per Ekman
*          pek@pdc.kth.se
* Mods:    Haihang You
*          you@cs.utk.edu
* Mods:    Maynard Johnson
*          maynardj@us.ibm.com
* Mods:    Brian Sheely
*          bsheely@eecs.utk.edu
* Mods:    <Gary Mohr>
*          <gary.mohr@bull.com>
* Mods:    <your name here>
*          <your email address>
*/

#include <stdarg.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "sw_multiplex.h"
#include "extras.h"
#include "papi_preset.h"
#include "cpus.h"

#include "papi_common_strings.h"

/* Advanced definitons */
static int default_debug_handler( int errorCode );
static long long handle_derived( EventInfo_t * evi, long long *from );

/* Global definitions used by other files */
int init_level = PAPI_NOT_INITED;
int _papi_hwi_error_level = PAPI_QUIET;
PAPI_debug_handler_t _papi_hwi_debug_handler = default_debug_handler;
papi_mdi_t _papi_hwi_system_info;
int _papi_hwi_errno = PAPI_OK;
int _papi_hwi_num_errors = 0;
hwi_presets_t user_defined_events[PAPI_MAX_USER_EVENTS];
int user_defined_events_count = 0;

THREAD_LOCAL_STORAGE_KEYWORD int _papi_rate_events_running = 0;
THREAD_LOCAL_STORAGE_KEYWORD int _papi_hl_events_running = 0;

/*****************************/
/* Native Event Mapping Code */
/*****************************/

#define NATIVE_EVENT_CHUNKSIZE 1024

struct native_event_info {
  int cidx;
  int component_event;
  int ntv_idx;
  char *evt_name;
};


// The following array is indexed by the papi event code (after the native bit has been removed)
static struct native_event_info *_papi_native_events=NULL;
static int num_native_events=0;
static int num_native_chunks=0;

char **_papi_errlist= NULL;
static int num_error_chunks = 0;


// pointer to event:mask string associated with last enum call to a components
// will be NULL for non libpfm4 components
// this is needed because libpfm4 event codes and papi event codes do not contain mask information
char *papi_event_string = NULL;
void
_papi_hwi_set_papi_event_string (const char *event_string) {
	INTDBG("event_string: %s\n", event_string);
	if (papi_event_string != NULL) {
		free (papi_event_string);
		papi_event_string = NULL;
	}
	if (event_string != NULL) {
		papi_event_string = strdup(event_string);
	}
	return;
}
char *
_papi_hwi_get_papi_event_string () {
	INTDBG("papi_event_string: %s\n", papi_event_string);
	return papi_event_string;
}
void
_papi_hwi_free_papi_event_string() {
	if (papi_event_string != NULL) {
		free(papi_event_string);
		papi_event_string = NULL;
	}
	return;
}

void
_papi_hwi_set_papi_event_code (unsigned int event_code, int update_flag) {
	INTDBG("new event_code: %#x, update_flag: %d, previous event_code: %#x\n", event_code, update_flag, _papi_hwi_my_thread->tls_papi_event_code);

	// if call is just to reset and start over, set both flags to show nothing saved yet
	if (update_flag < 0) {
		_papi_hwi_my_thread->tls_papi_event_code_changed = -1;
		_papi_hwi_my_thread->tls_papi_event_code = -1;
		return;
	}

	// if 0, it is being set prior to calling a component, if >0 it is being changed by the component
	_papi_hwi_my_thread->tls_papi_event_code_changed = update_flag;
	// save the event code passed in
	_papi_hwi_my_thread->tls_papi_event_code = event_code;
	return;
}
unsigned int
_papi_hwi_get_papi_event_code () {
	INTDBG("papi_event_code: %#x\n", _papi_hwi_my_thread->tls_papi_event_code);
	return _papi_hwi_my_thread->tls_papi_event_code;
}
/* Get the index into the ESI->NativeInfoArray for the current PAPI event code */
int
_papi_hwi_get_ntv_idx (unsigned int papi_evt_code) {
	INTDBG("ENTER: papi_evt_code: %#x\n", papi_evt_code);

	int result;
	int event_index;

	if (papi_evt_code == 0) {
		INTDBG("EXIT: PAPI_ENOEVNT, invalid papi event code\n");
		return PAPI_ENOEVNT;
	}

	event_index=papi_evt_code&PAPI_NATIVE_AND_MASK;
	if ((event_index<0) || (event_index>=num_native_events)) {
		INTDBG("EXIT: PAPI_ENOEVNT, invalid index into native event array\n");
		return PAPI_ENOEVNT;
	}

	result=_papi_native_events[event_index].ntv_idx;

	INTDBG("EXIT: result: %d\n", result);
	return result;
}

//
// Check for the presence of a component name or pmu name in the event string.
// If found check if it matches this component or one of the pmu's supported by this component.
//
// returns true if the event could be for this component and false if it is not for this component.
//    if there is no component or pmu name then it could be for this component and returns true.
//
static int
is_supported_by_component(int cidx, char *event_name) {
	INTDBG("ENTER: cidx: %d, event_name: %s\n", cidx, event_name);
	int i;
	int component_name = 0;
	int pmu_name = 0;
	char *wptr = NULL;

	// if event does not have a component name or pmu name, return to show it could be supported by this component
	// when component and pmu names are not provided, we just have to call the components to see if they recognize the event
	//

	// look for component names first
	if ((wptr = strstr(event_name, ":::")) != NULL) {
		component_name = 1;
	} else if ((wptr = strstr(event_name, "::")) != NULL) {
		pmu_name = 1;
	} else {
		INTDBG("EXIT: No Component or PMU name in event string, try this component\n");
		// need to force all components to be called to find owner of this event
		// ????  can we assume the default pmu when no component or pmu name is provided ????
		return 1;
	}

	// get a temporary copy of the component or pmu name
	int name_len = wptr - event_name;
	wptr = strdup(event_name);
	wptr[name_len] = '\0';

	// if a component name was found, compare it to the component name in the component info structure
	if (component_name) {
//		INTDBG("component_name: %s\n", _papi_hwd[cidx]->cmp_info.name);
		if (strcmp (wptr, _papi_hwd[cidx]->cmp_info.name) == 0) {
			free (wptr);
			INTDBG("EXIT: Component %s supports this event\n", _papi_hwd[cidx]->cmp_info.name);
			return 1;
		}
	}

	// if a pmu name was found, compare it to the pmu name list if the component info structure (if there is one)
	if (pmu_name) {
		for ( i=0 ; i<PAPI_PMU_MAX ; i++) {
			if (_papi_hwd[cidx]->cmp_info.pmu_names[i] == NULL) {
				continue;
			}
//			INTDBG("pmu_name[%d]: %p (%s)\n", i, _papi_hwd[cidx]->cmp_info.pmu_names[i], _papi_hwd[cidx]->cmp_info.pmu_names[i]);
			if (strcmp (wptr, _papi_hwd[cidx]->cmp_info.pmu_names[i]) == 0) {
				INTDBG("EXIT: Component %s supports PMU %s and this event\n", _papi_hwd[cidx]->cmp_info.name, wptr);
				free (wptr);
				return 1;
			}
		}
	}

	free (wptr);
	INTDBG("EXIT: Component does not support this event\n");
	return 0;
}

/** @internal
 * @class _papi_hwi_prefix_component_name
 * @brief Prefixes a component's name to each of its events.
 * @param *component_name
 * @param *event_name
 * @param *out
 * @param *out_len
 *
 * Given sane component_name and event_name it returns component_name:::event_name.
 * It is safe in the case that event_name == out and it checks against the
 * traditional PAPI 'cpu' components, opting to not prepend those.
 */
int
_papi_hwi_prefix_component_name( char *component_name, char *event_name, char *out, int out_len) 
{
	int size1, size2;
	char temp[out_len];

	size1 = strlen(event_name);
	size2 = strlen(component_name);

/* sanity checks */
	if ( size1 == 0 ) {
		return (PAPI_EBUG); /* hopefully event_name always has length?! */
	}

	if ( size1 >= out_len )
		return (PAPI_ENOMEM);

/* Guard against event_name == out */
	memcpy( temp, event_name, out_len );

/* no component name to prefix */
	if ( size2 == 0 ) {
		sprintf(out, "%s%c", temp, '\0' );
		return (PAPI_OK);
	}

/* Don't prefix 'cpu' component names for now */
	if ( strstr(component_name, "pe") ||
		 strstr(component_name, "bgq") ||
		 strstr(component_name, "bgp") ) {
		sprintf( out, "%s%c", temp, '\0');
		return (PAPI_OK);
	}

/* strlen(component_name) + ::: + strlen(event_name) + NULL */
	if ( size1+size2+3+1 > out_len )
		return (PAPI_ENOMEM);

	sprintf( out, "%s:::%s%c" , component_name, temp, '\0');
	return (PAPI_OK);
}

/** @internal
 *  @class _papi_hwi_strip_component_prefix
 *  @brief Strip off cmp_name::: from an event name.
 *
 *  @param *event_name
 *  @return Start of the component consumable portion of the name.
 *
 *  This function checks specifically for ':::' and will return the start of
 *  event_name if it doesn't find the ::: .
 */
const char *_papi_hwi_strip_component_prefix(const char *event_name)
{
	const char *start = NULL;
/* We assume ::: is the seperator
 * eg:
 * 		papi_component:::event_name
 */

	start = strstr( event_name, ":::" );
	if ( start != NULL )
		start+= 3; /* return the actual start of event_name */
	else
		start = event_name;

	return (start);
}

/* find the papi event code (4000xxx) associated with the specified component, native event, and event name */
static int
_papi_hwi_find_native_event(int cidx, int event, const char *event_name) {
  INTDBG("ENTER: cidx: %x, event: %#x, event_name: %s\n", cidx, event, event_name);

  int i;

  // if no event name passed in, it can not be found
  if (event_name == NULL) {
		INTDBG("EXIT: PAPI_ENOEVNT\n");
		return PAPI_ENOEVNT;
  }

  for(i=0;i<num_native_events;i++) {
  	// if we have have not set up this event name yet, look at next
  	if (_papi_native_events[i].evt_name == NULL) {
  		continue;
  	}

  	// is this entry for the correct component and event code
  	if ((_papi_native_events[i].cidx==cidx) &&
	(_papi_native_events[i].component_event==event)) {
		// if this event name matches what we want, return its papi event code
		if (strcmp(event_name, _papi_native_events[i].evt_name) == 0) {
			INTDBG("EXIT: event: %#x, component_event: %#x, ntv_idx: %d, event_name: %s\n",
				i|PAPI_NATIVE_MASK, _papi_native_events[i].component_event, _papi_native_events[i].ntv_idx, _papi_native_events[i].evt_name);
			return i|PAPI_NATIVE_MASK;
		}
    }
  }

	INTDBG("EXIT: PAPI_ENOEVNT\n");
	return PAPI_ENOEVNT;
}

static int
_papi_hwi_add_native_event(int cidx, int ntv_event, int ntv_idx, const char *event_name) {
	INTDBG("ENTER: cidx: %d, ntv_event: %#x, ntv_idx: %d, event_name: %s\n", cidx, ntv_event, ntv_idx, event_name);

  int new_native_event;

  _papi_hwi_lock( INTERNAL_LOCK );

  if (num_native_events>=num_native_chunks*NATIVE_EVENT_CHUNKSIZE) {
     num_native_chunks++;
     _papi_native_events=(struct native_event_info *) realloc(_papi_native_events,
				 num_native_chunks*NATIVE_EVENT_CHUNKSIZE*
				 sizeof(struct native_event_info));
     if (_papi_native_events==NULL) {
        new_native_event=PAPI_ENOMEM;
	goto native_alloc_early_out;
     }
  }

  _papi_native_events[num_native_events].cidx=cidx;
  _papi_native_events[num_native_events].component_event=ntv_event;
  _papi_native_events[num_native_events].ntv_idx=ntv_idx;
  if (event_name != NULL) {
	  _papi_native_events[num_native_events].evt_name=strdup(event_name);
  } else {
	  _papi_native_events[num_native_events].evt_name=NULL;
  }
  new_native_event=num_native_events|PAPI_NATIVE_MASK;

  num_native_events++;

native_alloc_early_out:

  _papi_hwi_unlock( INTERNAL_LOCK );

  INTDBG("EXIT: new_native_event: %#x, num_native_events: %d\n", new_native_event, num_native_events);
  return new_native_event;
}

/** @internal
 * @class _papi_hwi_add_error
 *
 * Adds a new error string to PAPI's internal store.
 * MAKE SURE you are not holding INTERNAL_LOCK when you call me!
 */
static int
_papi_hwi_add_error( char *error )
{
	INTDBG("Adding a new Error message |%s|\n", error);
	_papi_hwi_lock(INTERNAL_LOCK);

	if (_papi_hwi_num_errors >= num_error_chunks*NATIVE_EVENT_CHUNKSIZE) {
		num_error_chunks++;
		_papi_errlist= (char **) realloc(_papi_errlist,
						num_error_chunks*NATIVE_EVENT_CHUNKSIZE*sizeof(char *));
		if (_papi_errlist==NULL) {
			_papi_hwi_num_errors = -2;
			goto bail;
		}

	}

	_papi_errlist[_papi_hwi_num_errors] = strdup( error );
	if ( _papi_errlist[_papi_hwi_num_errors] == NULL )
		_papi_hwi_num_errors = -2;

bail:
	_papi_hwi_unlock(INTERNAL_LOCK);

	return _papi_hwi_num_errors++;
}

static void
_papi_hwi_cleanup_errors()
{
	int i;

	if ( _papi_errlist == NULL ||
			_papi_hwi_num_errors == 0 )
		return;


	_papi_hwi_lock( INTERNAL_LOCK );
	for (i=0; i < _papi_hwi_num_errors; i++ ) {
		free( _papi_errlist[i]);
		_papi_errlist[i] = NULL;
	}

	free( _papi_errlist );
	_papi_errlist = NULL;
	_papi_hwi_num_errors = 0;
	num_error_chunks=0;

	_papi_hwi_unlock( INTERNAL_LOCK );
}

static int
_papi_hwi_lookup_error( char *error )
{
	int i;

	for (i=0; i<_papi_hwi_num_errors; i++) {
		if ( !strncasecmp( _papi_errlist[i], error, strlen( error ) ) )
			return i;
	}

	return (-1);
}

/** @internal
 *  @class _papi_hwi_publish_error
 *
 *  @return
 *  	<= 0 : Code for the error.
 *  	< 0  : We couldn't get memory to allocate for your error.
 *
 * 	An internal interface for adding an error code to the library.
 * 	The returned code is suitable for returning to users.
 *  */
int _papi_hwi_publish_error( char *error )
{
	int error_code = -1;

	if ( (error_code = _papi_hwi_lookup_error( error )) < 0 )
		error_code = _papi_hwi_add_error(error);

	return (-error_code); /* internally error_code is an index, externally, it should be <= 0 */
}


/* Why are the errors done this way?
   Should they not be auto-generated the same way the Fortran ones are?
   --vmw
*/
void
_papi_hwi_init_errors(void) {
/* we use add error to avoid the cost of lookups, we know the errors are not there yet */
	/*  0 PAPI_OK */	_papi_hwi_add_error("No error");
	/*  1 PAPI_EINVAL */	_papi_hwi_add_error("Invalid argument");
	/*  2 PAPI_ENOMEM */	_papi_hwi_add_error("Insufficient memory");
	/*  3 PAPI_ESYS */	_papi_hwi_add_error("A System/C library call failed");
	/*  4 PAPI_ECMP */	_papi_hwi_add_error("Not supported by component");
	/*  5 PAPI_ECLOST */	_papi_hwi_add_error("Access to the counters was lost or interrupted");
	/*  6 PAPI_EBUG */	_papi_hwi_add_error("Internal error, please send mail to the developers");
	/*  7 PAPI_ENOEVNT */	_papi_hwi_add_error("Event does not exist");
	/*  8 PAPI_ECNFLCT */	_papi_hwi_add_error("Event exists, but cannot be counted due to hardware resource limits");
	/*  9 PAPI_ENOTRUN */	_papi_hwi_add_error("EventSet is currently not running");
	/* 10 PAPI_EISRUN */	_papi_hwi_add_error("EventSet is currently counting");
	/* 11 PAPI_ENOEVST */	_papi_hwi_add_error("No such EventSet available");
	/* 12 PAPI_ENOTPRESET */_papi_hwi_add_error("Event in argument is not a valid preset");
	/* 13 PAPI_ENOCNTR */	_papi_hwi_add_error("Hardware does not support performance counters");
	/* 14 PAPI_EMISC */	_papi_hwi_add_error("Unknown error code");
	/* 15 PAPI_EPERM */	_papi_hwi_add_error("Permission level does not permit operation");
	/* 16 PAPI_ENOINIT */	_papi_hwi_add_error("PAPI hasn't been initialized yet");
	/* 17 PAPI_ENOCMP */	_papi_hwi_add_error("Component Index isn't set");
	/* 18 PAPI_ENOSUPP */	_papi_hwi_add_error("Not supported");
	/* 19 PAPI_ENOIMPL */	_papi_hwi_add_error("Not implemented");
	/* 20 PAPI_EBUF */	_papi_hwi_add_error("Buffer size exceeded");
	/* 21 PAPI_EINVAL_DOM */_papi_hwi_add_error("EventSet domain is not supported for the operation");
	/* 22 PAPI_EATTR */	_papi_hwi_add_error("Invalid or missing event attributes");
	/* 23 PAPI_ECOUNT */	_papi_hwi_add_error("Too many events or attributes");
	/* 24 PAPI_ECOMBO */	_papi_hwi_add_error("Bad combination of features");
	/* 25 PAPI_ECMP_DISABLED */_papi_hwi_add_error("Component containing event is disabled");
    /* 26 PAPI_EDELAY_INIT */ _papi_hwi_add_error("Delayed initialization component");
    /* 27 PAPI_EMULPASS */ _papi_hwi_add_error("Event exists, but cannot be counted due to multiple passes required by hardware");
}

int
_papi_hwi_invalid_cmp( int cidx )
{
  return ( cidx < 0 || cidx >= papi_num_components );
}


int
_papi_hwi_component_index( int event_code ) {
	INTDBG("ENTER: event_code: %#x\n", event_code);

  int cidx;
  int event_index;

  /* currently assume presets are for component 0 only */
  if (IS_PRESET(event_code)) {
     INTDBG("EXIT: Event %#x is a PRESET, assigning component %d\n", event_code,0);
     return 0;
  }

  /* user defined events are treated like preset events (component 0 only) */
  if (IS_USER_DEFINED(event_code)) {
     INTDBG("EXIT: Event %#x is USER DEFINED, assigning component %d\n", event_code,0);
     return 0;
  }

  event_index=event_code&PAPI_NATIVE_AND_MASK;

  if ( (event_index < 0) || (event_index>=num_native_events)) {
     INTDBG("EXIT: Event index %#x is out of range, num_native_events: %d\n", event_index, num_native_events);
     return PAPI_ENOEVNT;
  }

  cidx=_papi_native_events[event_index].cidx;

  if ((cidx<0) || (cidx >= papi_num_components)) {
	  INTDBG("EXIT: Component index %#x is out of range, papi_num_components: %d\n", cidx, papi_num_components);
	  return PAPI_ENOCMP;
  }

  INTDBG("EXIT: Found cidx: %d event_index: %d, event_code: %#x\n", cidx, event_index, event_code);
  return cidx;
}

/* Convert an internal component event to a papi event code */
int
_papi_hwi_native_to_eventcode(int cidx, int event_code, int ntv_idx, const char *event_name) {
  INTDBG("Entry: cidx: %d, event: %#x, ntv_idx: %d, event_name: %s\n", cidx, event_code, ntv_idx, event_name);

  int result;

  if (_papi_hwi_my_thread->tls_papi_event_code_changed > 0) {
	  result = _papi_hwi_get_papi_event_code();
	  INTDBG("EXIT: papi_event_code: %#x set by the component\n", result);
	  return result;
  }

  result=_papi_hwi_find_native_event(cidx, event_code, event_name);
  if (result==PAPI_ENOEVNT) {
     // Need to create one
     result=_papi_hwi_add_native_event(cidx, event_code, ntv_idx, event_name);
  }

  INTDBG("EXIT: result: %#x\n", result);
  return result;
}

/* Convert a native_event code to an internal event code */
int
_papi_hwi_eventcode_to_native(int event_code) {
	INTDBG("ENTER: event_code: %#x\n", event_code);

  int result;
  int event_index;

  event_index=event_code&PAPI_NATIVE_AND_MASK;
 if ((event_index < 0)  ||  (event_index>=num_native_events)) {
    INTDBG("EXIT: PAPI_ENOEVNT\n");
    return PAPI_ENOEVNT;
  }

  result=_papi_native_events[event_index].component_event;

  INTDBG("EXIT: result: %#x\n", result);
  return result;

}


/*********************/
/* Utility functions */
/*********************/

void
PAPIERROR( char *format, ... )
{
	va_list args;
	if ( ( _papi_hwi_error_level != PAPI_QUIET ) ||
		 ( getenv( "PAPI_VERBOSE" ) ) ) {
		va_start( args, format );
		fprintf( stderr, "PAPI Error: " );
		vfprintf( stderr, format, args );
		fprintf( stderr, "\n" );
		va_end( args );
	}
}

void
PAPIWARN( char *format, ... )
{
	va_list args;
	if ( ( _papi_hwi_error_level != PAPI_QUIET ) ||
		 ( getenv( "PAPI_VERBOSE" ) ) ) {
		va_start( args, format );
		fprintf( stderr, "PAPI Warning: " );
		vfprintf( stderr, format, args );
		fprintf( stderr, "\n" );
		va_end( args );
	}
}

static int
default_debug_handler( int errorCode )
{
	char str[PAPI_HUGE_STR_LEN];

	if ( errorCode == PAPI_OK )
		return ( errorCode );
	if ( ( errorCode > 0 ) || ( -errorCode > _papi_hwi_num_errors ) ) {
		PAPIERROR( "%s %d,%s,Bug! Unknown error code", PAPI_ERROR_CODE_str,
				   errorCode, "" );
		return ( PAPI_EBUG );
	}

	switch ( _papi_hwi_error_level ) {
	case PAPI_VERB_ECONT:
	case PAPI_VERB_ESTOP:
		/* gcc 2.96 bug fix, do not change */
		/* fprintf(stderr,"%s %d: %s: %s\n",PAPI_ERROR_CODE_str,errorCode,_papi_hwi_err[-errorCode].name,_papi_hwi_err[-errorCode].descr); */

		sprintf( str, "%s %d,%s", PAPI_ERROR_CODE_str, errorCode,
				 _papi_errlist[-errorCode] );
		if ( errorCode == PAPI_ESYS )
			sprintf( str + strlen( str ), ": %s", strerror( errno ) );

		PAPIERROR( str );

		if ( _papi_hwi_error_level == PAPI_VERB_ESTOP )
			abort(  );		 /* patch provided by will cohen of redhat */
		else
			return errorCode;
		break;

	case PAPI_QUIET:
	default:
		return errorCode;
	}
	return ( PAPI_EBUG );	 /* Never get here */
}

static int
allocate_eventset_map( DynamicArray_t * map )
{
	/* Allocate and clear the Dynamic Array structure */
	if ( map->dataSlotArray != NULL )
		papi_free( map->dataSlotArray );
	memset( map, 0x00, sizeof ( DynamicArray_t ) );

	/* Allocate space for the EventSetInfo_t pointers */

	map->dataSlotArray =
		( EventSetInfo_t ** ) papi_malloc( PAPI_INIT_SLOTS *
										   sizeof ( EventSetInfo_t * ) );
	if ( map->dataSlotArray == NULL ) {
		return ( PAPI_ENOMEM );
	}
	memset( map->dataSlotArray, 0x00,
			PAPI_INIT_SLOTS * sizeof ( EventSetInfo_t * ) );
	map->totalSlots = PAPI_INIT_SLOTS;
	map->availSlots = PAPI_INIT_SLOTS;
	map->fullSlots = 0;

	return ( PAPI_OK );
}

static int
expand_dynamic_array( DynamicArray_t * DA )
{
	int number;
	EventSetInfo_t **n;

	/*realloc existing PAPI_EVENTSET_MAP.dataSlotArray */

	number = DA->totalSlots * 2;
	n = ( EventSetInfo_t ** ) papi_realloc( DA->dataSlotArray,
											( size_t ) number *
											sizeof ( EventSetInfo_t * ) );
	if ( n == NULL )
		return ( PAPI_ENOMEM );

	/* Need to assign this value, what if realloc moved it? */

	DA->dataSlotArray = n;

	memset( DA->dataSlotArray + DA->totalSlots, 0x00,
			( size_t ) DA->totalSlots * sizeof ( EventSetInfo_t * ) );

	DA->totalSlots = number;
	DA->availSlots = number - DA->fullSlots;

	return ( PAPI_OK );
}

static int
EventInfoArrayLength( const EventSetInfo_t * ESI )
{
   return ( _papi_hwd[ESI->CmpIdx]->cmp_info.num_mpx_cntrs );
}





/*========================================================================*/
/* This function allocates space for one EventSetInfo_t structure and for */
/* all of the pointers in this structure.  If any malloc in this function */
/* fails, all memory malloced to the point of failure is freed, and NULL  */
/* is returned.  Upon success, a pointer to the EventSetInfo_t data       */
/* structure is returned.                                                 */
/*========================================================================*/


static int
create_EventSet( EventSetInfo_t ** here )
{
   EventSetInfo_t *ESI;

   ESI = ( EventSetInfo_t * ) papi_calloc( 1, sizeof ( EventSetInfo_t ) );
   if ( ESI == NULL ) {
      return PAPI_ENOMEM;
   }

   *here = ESI;

   return PAPI_OK;
}

int
_papi_hwi_assign_eventset( EventSetInfo_t *ESI, int cidx )
{
   INTDBG("ENTER: ESI: %p (%d), cidx: %d\n", ESI, ESI->EventSetIndex, cidx);
   int retval;
   size_t max_counters;
   char *ptr;
   unsigned int i, j;

   /* If component doesn't exist... */
   if (_papi_hwi_invalid_cmp(cidx)) return PAPI_ECMP;

   /* Assigned at create time */
   ESI->domain.domain = _papi_hwd[cidx]->cmp_info.default_domain;
   ESI->granularity.granularity =
	                         _papi_hwd[cidx]->cmp_info.default_granularity;
   ESI->CmpIdx = cidx;

   /* ??? */
   max_counters = ( size_t ) _papi_hwd[cidx]->cmp_info.num_mpx_cntrs;

   ESI->ctl_state = (hwd_control_state_t *) papi_calloc( 1, (size_t)
				   _papi_hwd[cidx]->size.control_state );
   ESI->sw_stop = (long long *) papi_calloc( ( size_t ) max_counters,
						      sizeof ( long long ) );
   ESI->hw_start = ( long long * ) papi_calloc( ( size_t ) max_counters,
                                                      sizeof ( long long ) );
   ESI->EventInfoArray = ( EventInfo_t * ) papi_calloc( (size_t) max_counters,
                                                      sizeof ( EventInfo_t ) );

   /* allocate room for the native events and for the component-private */
   /* register structures */
   /* ugh is there a cleaner way to allocate this?  vmw */
   ESI->NativeInfoArray = ( NativeInfo_t * )
     papi_calloc( ( size_t ) max_counters, sizeof ( NativeInfo_t ));

   ESI->NativeBits = papi_calloc(( size_t ) max_counters,
                                 ( size_t ) _papi_hwd[cidx]->size.reg_value );

   /* NOTE: the next two malloc allocate blocks of memory that are later */
   /* parcelled into overflow and profile arrays                         */
   ESI->overflow.deadline = ( long long * )
		papi_malloc( ( sizeof ( long long ) +
					   sizeof ( int ) * 3 ) * ( size_t ) max_counters );

   ESI->profile.prof = ( PAPI_sprofil_t ** )
		papi_malloc( ( sizeof ( PAPI_sprofil_t * ) * ( size_t ) max_counters +
					   ( size_t ) max_counters * sizeof ( int ) * 4 ) );

   /* If any of these allocations failed, free things up and fail */

   if ( ( ESI->ctl_state == NULL ) ||
	( ESI->sw_stop == NULL )   ||
        ( ESI->hw_start == NULL )  ||
	( ESI->NativeInfoArray == NULL ) ||
	( ESI->NativeBits == NULL ) ||
        ( ESI->EventInfoArray == NULL )  ||
	( ESI->profile.prof == NULL ) ||
        ( ESI->overflow.deadline == NULL ) ) {

      if ( ESI->sw_stop ) papi_free( ESI->sw_stop );
      if ( ESI->hw_start ) papi_free( ESI->hw_start );
      if ( ESI->EventInfoArray ) papi_free( ESI->EventInfoArray );
      if ( ESI->NativeInfoArray ) papi_free( ESI->NativeInfoArray );
      if ( ESI->NativeBits ) papi_free( ESI->NativeBits );
      if ( ESI->ctl_state ) papi_free( ESI->ctl_state );
      if ( ESI->overflow.deadline ) papi_free( ESI->overflow.deadline );
      if ( ESI->profile.prof ) papi_free( ESI->profile.prof );
      papi_free( ESI );
      return PAPI_ENOMEM;
   }


   /* Carve up the overflow block into separate arrays */
   ptr = ( char * ) ESI->overflow.deadline;
   ptr += sizeof ( long long ) * max_counters;
   ESI->overflow.threshold = ( int * ) ptr;
   ptr += sizeof ( int ) * max_counters;
   ESI->overflow.EventIndex = ( int * ) ptr;
   ptr += sizeof ( int ) * max_counters;
   ESI->overflow.EventCode = ( int * ) ptr;

   /* Carve up the profile block into separate arrays */
   ptr = ( char * ) ESI->profile.prof +
		( sizeof ( PAPI_sprofil_t * ) * max_counters );
   ESI->profile.count = ( int * ) ptr;
   ptr += sizeof ( int ) * max_counters;
   ESI->profile.threshold = ( int * ) ptr;
   ptr += sizeof ( int ) * max_counters;
   ESI->profile.EventIndex = ( int * ) ptr;
   ptr += sizeof ( int ) * max_counters;
   ESI->profile.EventCode = ( int * ) ptr;

   /* initialize_EventInfoArray */

   for ( i = 0; i < max_counters; i++ ) {
       ESI->EventInfoArray[i].event_code=( unsigned int ) PAPI_NULL;
       ESI->EventInfoArray[i].ops = NULL;
       ESI->EventInfoArray[i].derived=NOT_DERIVED;
       for ( j = 0; j < PAPI_EVENTS_IN_DERIVED_EVENT; j++ ) {
	   ESI->EventInfoArray[i].pos[j] = PAPI_NULL;
       }
   }

   /* initialize_NativeInfoArray */
   for( i = 0; i < max_counters; i++ ) {
      ESI->NativeInfoArray[i].ni_event = -1;
      ESI->NativeInfoArray[i].ni_position = -1;
      ESI->NativeInfoArray[i].ni_papi_code = -1;
      ESI->NativeInfoArray[i].ni_owners = 0;
      ESI->NativeInfoArray[i].ni_bits = ((unsigned char*)ESI->NativeBits) +
                                          (i*_papi_hwd[cidx]->size.reg_value);
   }

   ESI->NativeCount = 0;

   ESI->state = PAPI_STOPPED;

   /* these used to be init_config */
   retval = _papi_hwd[cidx]->init_control_state( ESI->ctl_state );
   retval |= _papi_hwd[cidx]->set_domain( ESI->ctl_state, ESI->domain.domain);

   return retval;
}

/*========================================================================*/
/* This function should free memory for one EventSetInfo_t structure.     */
/* The argument list consists of a pointer to the EventSetInfo_t          */
/* structure, *ESI.                                                       */
/* The calling function should check  for ESI==NULL.                      */
/*========================================================================*/

void
_papi_hwi_free_EventSet( EventSetInfo_t * ESI )
{
	_papi_hwi_cleanup_eventset( ESI );

#ifdef DEBUG
	memset( ESI, 0x00, sizeof ( EventSetInfo_t ) );
#endif
	papi_free( ESI );

}

static int
add_EventSet( EventSetInfo_t * ESI, ThreadInfo_t * master )
{
	DynamicArray_t *map = &_papi_hwi_system_info.global_eventset_map;
	int i, errorCode;

	_papi_hwi_lock( INTERNAL_LOCK );

	if ( map->availSlots == 0 ) {
		errorCode = expand_dynamic_array( map );
		if ( errorCode < PAPI_OK ) {
			_papi_hwi_unlock( INTERNAL_LOCK );
			return ( errorCode );
		}
	}

	i = 0;
	for ( i = 0; i < map->totalSlots; i++ ) {
		if ( map->dataSlotArray[i] == NULL ) {
			ESI->master = master;
			ESI->EventSetIndex = i;
			map->fullSlots++;
			map->availSlots--;
			map->dataSlotArray[i] = ESI;
			_papi_hwi_unlock( INTERNAL_LOCK );
			return ( PAPI_OK );
		}
	}

	_papi_hwi_unlock( INTERNAL_LOCK );
	return ( PAPI_EBUG );
}

int
_papi_hwi_create_eventset( int *EventSet, ThreadInfo_t * handle )
{
	EventSetInfo_t *ESI;
	int retval;

	/* Is the EventSet already in existence? */

	if ( ( EventSet == NULL ) || ( handle == NULL ) )
		return PAPI_EINVAL;

	if ( *EventSet != PAPI_NULL )
		return PAPI_EINVAL;

	/* Well, then allocate a new one. Use n to keep track of a NEW EventSet */

	retval = create_EventSet( &ESI );
	if ( retval != PAPI_OK )
		return retval;

	ESI->CmpIdx = -1;		 /* when eventset is created, it is not decided yet which component it belongs to, until first event is added */
	ESI->state = PAPI_STOPPED;

	/* Add it to the global table */

	retval = add_EventSet( ESI, handle );
	if ( retval < PAPI_OK ) {
		_papi_hwi_free_EventSet( ESI );
		return retval ;
	}

	*EventSet = ESI->EventSetIndex;

	INTDBG( "(%p,%p): new EventSet in slot %d\n",
			( void * ) EventSet, handle, *EventSet );

	return retval;
}

/* This function returns the index of the the next free slot
   in the EventInfoArray. If EventCode is already in the list,
   it returns PAPI_ECNFLCT. */

static int
get_free_EventCodeIndex( const EventSetInfo_t * ESI, unsigned int EventCode )
{
	int k;
	int lowslot = PAPI_ECNFLCT;
	int limit = EventInfoArrayLength( ESI );

	/* Check for duplicate events and get the lowest empty slot */

	for ( k = 0; k < limit; k++ ) {
		if ( ESI->EventInfoArray[k].event_code == EventCode )
			return ( PAPI_ECNFLCT );
		/*if ((ESI->EventInfoArray[k].event_code == PAPI_NULL) && (lowslot == PAPI_ECNFLCT)) */
		if ( ESI->EventInfoArray[k].event_code == ( unsigned int ) PAPI_NULL ) {
			lowslot = k;
			break;
		}
	}
	return ( lowslot );
}

/* This function returns the index of the EventCode or error */
/* Index to what? The index to everything stored EventCode in the */
/* EventSet. */

int
_papi_hwi_lookup_EventCodeIndex( const EventSetInfo_t * ESI,
				 unsigned int EventCode )
{
	int i;
	int limit = EventInfoArrayLength( ESI );

	for ( i = 0; i < limit; i++ ) {
	   if ( ESI->EventInfoArray[i].event_code == EventCode ) {
	      return i;
	   }
	}

	return PAPI_EINVAL;
}

/* This function only removes empty EventSets */

int
_papi_hwi_remove_EventSet( EventSetInfo_t * ESI )
{
	DynamicArray_t *map = &_papi_hwi_system_info.global_eventset_map;
	int i;

	i = ESI->EventSetIndex;

	_papi_hwi_lock( INTERNAL_LOCK );

	_papi_hwi_free_EventSet( ESI );

	/* do bookkeeping for PAPI_EVENTSET_MAP */

	map->dataSlotArray[i] = NULL;
	map->availSlots++;
	map->fullSlots--;

	_papi_hwi_unlock( INTERNAL_LOCK );

	return PAPI_OK;
}


/* this function checks if an event is already in an EventSet
     Success, return ESI->NativeInfoArray[] index
     Fail,    return PAPI_ENOEVNT;
*/
static int
event_already_in_eventset( EventSetInfo_t * ESI, int papi_event )
{
   INTDBG( "ENTER: ESI: %p, papi_event: %#x\n", ESI, papi_event);
   int i;

   int nevt = _papi_hwi_eventcode_to_native(papi_event);

   /* to find the native event from the native events list */
   for( i = 0; i < ESI->NativeCount; i++ ) {
      if ( nevt == ESI->NativeInfoArray[i].ni_event ) {
         // Also need to check papi event code if set because the same event with different masks
         // will generate the same libpfm4 event code (what was checked above).  But there will be
         // different papi events created for it and they need to be handled separately.
         if (papi_event == ESI->NativeInfoArray[i].ni_papi_code) {
            INTDBG( "EXIT: event: %#x already mapped at index: %d\n", papi_event, i);
            return i;
         }
      }
   }
   INTDBG( "EXIT: PAPI_ENOEVNT\n");
   return PAPI_ENOEVNT;
}

/* This function goes through the events in an EventSet's EventInfoArray */
/* And maps each event (whether native or part of a preset) to           */
/* an event in the EventSets NativeInfoArray.                            */

/* We need to do this every time a native event is added to or removed   */
/* from an eventset.                                                     */

/* It is also called after a update controlstate as the components are   */
/* allowed to re-arrange the native events to fit hardware constraints.  */

void
_papi_hwi_map_events_to_native( EventSetInfo_t *ESI)
{
	INTDBG("ENTER: ESI: %p, ESI->EventInfoArray: %p, ESI->NativeInfoArray: %p, ESI->NumberOfEvents: %d, ESI->NativeCount: %d\n", ESI, ESI->EventInfoArray, ESI->NativeInfoArray, ESI->NumberOfEvents, ESI->NativeCount);

	int i, event, k, n, preset_index = 0, nevt;
	int total_events = ESI->NumberOfEvents;

	event = 0;
	for( i = 0; i < total_events; i++ ) {

		/* find the first event that isn't PAPI_NULL */
		/* Is this really necessary? --vmw           */
		while ( ESI->EventInfoArray[event].event_code == ( unsigned int ) PAPI_NULL ) {
			event++;
		}

		/* If it's a preset */
		if ( IS_PRESET(ESI->EventInfoArray[event].event_code) ) {
			preset_index = ( int ) ESI->EventInfoArray[event].event_code & PAPI_PRESET_AND_MASK;

			/* walk all sub-events in the preset */
			for( k = 0; k < PAPI_EVENTS_IN_DERIVED_EVENT; k++ ) {
				nevt = _papi_hwi_presets[preset_index].code[k];
				if ( nevt == PAPI_NULL ) {
					break;
				}

				INTDBG("Looking for subevent %#x\n",nevt);

				/* Match each sub-event to something in the Native List */
				for( n = 0; n < ESI->NativeCount; n++ ) {
					if ( nevt == ESI->NativeInfoArray[n].ni_papi_code ) {
						INTDBG("Found papi event: %#x, &ESI->NativeInfoArray[%d]: %p, ni_event: %#x, ni_position %d\n",
								nevt, n, &(ESI->NativeInfoArray[n]), ESI->NativeInfoArray[n].ni_event, ESI->NativeInfoArray[n].ni_position);
						ESI->EventInfoArray[event].pos[k] = ESI->NativeInfoArray[n].ni_position;
						break;
					}
				}
			}
		}
		/* If it's a native event */
		else if( IS_NATIVE(ESI->EventInfoArray[event].event_code) ) {
			nevt = ( int ) ESI->EventInfoArray[event].event_code;

			// get index into native info array for this event
			int nidx = event_already_in_eventset( ESI, nevt );
			// if not found, then we need to return an error
			if (nidx == PAPI_ENOEVNT) {
				INTDBG("EXIT: needed event not found\n");
				return;
			}
			ESI->EventInfoArray[event].pos[0] = ESI->NativeInfoArray[nidx].ni_position;
			INTDBG("nidx: %d, ni_position: %d\n", nidx, ESI->NativeInfoArray[nidx].ni_position);

		}
		/* If it's a user-defined event */
		else if ( IS_USER_DEFINED(ESI->EventInfoArray[event].event_code) ) {
			preset_index = ( int ) ESI->EventInfoArray[event].event_code & PAPI_UE_AND_MASK;
			for ( k = 0; k < PAPI_EVENTS_IN_DERIVED_EVENT; k++ ) {
				nevt = user_defined_events[preset_index].code[k];
				INTDBG("nevt: %#x, user_defined_events[%d].code[%d]: %#x, code[%d]: %#x\n",
						nevt, preset_index, k, user_defined_events[preset_index].code[k], k+1, user_defined_events[preset_index].code[k+1]);

				if ( nevt == PAPI_NULL ) break;

				/* Match each sub-event to something in the Native List */
				for ( n = 0; n < ESI->NativeCount; n++ ) {
					// if this is the event we are looking for, set its position and exit inner loop to look for next sub-event
					if ( _papi_hwi_eventcode_to_native(nevt) == ESI->NativeInfoArray[n].ni_event ) {
						ESI->EventInfoArray[event].pos[k] = ESI->NativeInfoArray[n].ni_position;
						break;
					}
				}
			}
		}
		event++;
	}
	INTDBG("EXIT: \n");
	return;
}


static int
add_native_fail_clean( EventSetInfo_t *ESI, int nevt )
{
	INTDBG("ENTER: ESI: %p, nevt: %#x\n", ESI, nevt);

   int i, max_counters;
   int cidx;

   cidx = _papi_hwi_component_index( nevt );
   if (cidx<0) return PAPI_ENOCMP;

   max_counters = _papi_hwd[cidx]->cmp_info.num_mpx_cntrs;

   /* to find the native event from the native events list */
   for( i = 0; i < max_counters; i++ ) {
//	   INTDBG("ESI->NativeInfoArray[%d]: %p, ni_event: %#x, ni_papi_event_code: %#x, ni_position: %d, ni_owners: %d\n",
//			   i, &(ESI->NativeInfoArray[i]), ESI->NativeInfoArray[i].ni_event, ESI->NativeInfoArray[i].ni_papi_code, ESI->NativeInfoArray[i].ni_position, ESI->NativeInfoArray[i].ni_owners);
     if ( nevt == ESI->NativeInfoArray[i].ni_papi_code ) {
	 ESI->NativeInfoArray[i].ni_owners--;
	 /* to clean the entry in the nativeInfo array */
	 if ( ESI->NativeInfoArray[i].ni_owners == 0 ) {
	    ESI->NativeInfoArray[i].ni_event = -1;
	    ESI->NativeInfoArray[i].ni_position = -1;
	    ESI->NativeInfoArray[i].ni_papi_code = -1;
	    ESI->NativeCount--;
	 }
	 INTDBG( "EXIT: nevt: %#x, returned: %d\n", nevt, i);
	 return i;
      }
   }
	INTDBG( "EXIT: returned: -1\n");
   return -1;
}

/* since update_control_state trashes overflow settings, this puts things
   back into balance. */
static int
update_overflow( EventSetInfo_t * ESI )
{
   int i, retval = PAPI_OK;

   if ( ESI->overflow.flags & PAPI_OVERFLOW_HARDWARE ) {
      for( i = 0; i < ESI->overflow.event_counter; i++ ) {
	 retval = _papi_hwd[ESI->CmpIdx]->set_overflow( ESI,
							ESI->overflow.EventIndex[i],
							ESI->overflow.threshold[i] );
	 if ( retval != PAPI_OK ) {
	    break;
	 }
      }
   }
   return retval;
}

/* this function is called by _papi_hwi_add_event when adding native events
   ESI:   event set to add the events to
   nevnt: pointer to array of native event table indexes to add
   size:  number of native events to add
   out:   ???

   return:  < 0 = error
              0 = no new events added
              1 = new events added
*/
static int
add_native_events( EventSetInfo_t *ESI, unsigned int *nevt,
                   int size, EventInfo_t *out )
{
	INTDBG ("ENTER: ESI: %p, nevt: %p, size: %d, out: %p\n", ESI, nevt, size, out);
   int nidx, i, j, added_events = 0;
   int retval, retval2;
   int max_counters;
   hwd_context_t *context;

   max_counters = _papi_hwd[ESI->CmpIdx]->cmp_info.num_mpx_cntrs;

   /* Walk through the list of native events, adding them */
   for( i = 0; i < size; i++ ) {

      /* Check to see if event is already in EventSet */
      nidx = event_already_in_eventset( ESI, nevt[i] );

      if ( nidx >= 0 ) {
	 /* Event is already there.  Set position */
	 out->pos[i] = ESI->NativeInfoArray[nidx].ni_position;
	 ESI->NativeInfoArray[nidx].ni_owners++;
	 continue;
      }

	 /* Event wasn't already there */

	 if ( ESI->NativeCount == max_counters ) {

	    /* No more room in counters! */
	    for( j = 0; j < i; j++ ) {
	       if ( ( nidx = add_native_fail_clean( ESI, nevt[j] ) ) >= 0 ) {
		  out->pos[j] = -1;
		  continue;
	       }
	       INTDBG( "should not happen!\n" );
	    }
	    INTDBG( "EXIT: counters are full!\n" );
	    return PAPI_ECOUNT;
	 }

	    /* there is an empty slot for the native event; */
	    /* initialize the native index for the new added event */
	    INTDBG( "Adding nevt[%d]: %#x, ESI->NativeInfoArray[%d]: %p, Component: %d\n",
		    i, nevt[i], ESI->NativeCount, &ESI->NativeInfoArray[ESI->NativeCount], ESI->CmpIdx );
	    ESI->NativeInfoArray[ESI->NativeCount].ni_event =
			  _papi_hwi_eventcode_to_native(nevt[i]);
	    ESI->NativeInfoArray[ESI->NativeCount].ni_papi_code = nevt[i];

	    ESI->NativeInfoArray[ESI->NativeCount].ni_owners = 1;
	    ESI->NativeCount++;
	    added_events++;
   }

   INTDBG("added_events: %d\n", added_events);

   /* if we added events we need to tell the component so it */
   /* can add them too.                                      */
   if ( added_events ) {
      /* get the context we should use for this event set */
      context = _papi_hwi_get_context( ESI, NULL );

      if ( _papi_hwd[ESI->CmpIdx]->allocate_registers( ESI ) == PAPI_OK ) {

	 retval = _papi_hwd[ESI->CmpIdx]->update_control_state( ESI->ctl_state,
		  ESI->NativeInfoArray,
		  ESI->NativeCount,
		  context);
	 if ( retval != PAPI_OK ) {
clean:
	    for( i = 0; i < size; i++ ) {
	       if ( ( nidx = add_native_fail_clean( ESI, nevt[i] ) ) >= 0 ) {
		  out->pos[i] = -1;
		  continue;
	       }
	       INTDBG( "should not happen!\n" );
	    }
	    /* re-establish the control state after the previous error */
	    retval2 = _papi_hwd[ESI->CmpIdx]->update_control_state(
                       ESI->ctl_state,
		       ESI->NativeInfoArray,
		       ESI->NativeCount,
		       context);
	    if ( retval2 != PAPI_OK ) {
	       PAPIERROR("update_control_state failed to re-establish working events!" );
	       INTDBG( "EXIT: update_control_state returned: %d\n", retval2);
	       return retval2;
	    }
	    INTDBG( "EXIT: update_control_state returned: %d\n", retval);
	    return retval;
	 }
     INTDBG( "EXIT: update_control_state returned: %d, we return: 1 (need remap)\n", retval);
	 return 1; /* need remap */
      } else {
	 retval = PAPI_EMISC;
	 goto clean;
      }
   }
   INTDBG( "EXIT: PAPI_OK\n");
   return PAPI_OK;
}


int
_papi_hwi_add_event( EventSetInfo_t * ESI, int EventCode )
{
    INTDBG("ENTER: ESI: %p (%d), EventCode: %#x\n", ESI, ESI->EventSetIndex, EventCode);

    int i, j, thisindex, remap, retval = PAPI_OK;
    int cidx;

	/* Sanity check the component */
	cidx=_papi_hwi_component_index( EventCode );
	if (cidx<0) {
		return PAPI_ENOCMP;
	}
	if (_papi_hwd[cidx]->cmp_info.disabled &&
        _papi_hwd[cidx]->cmp_info.disabled != PAPI_EDELAY_INIT) {
		return PAPI_ECMP_DISABLED;
	}

    /* Sanity check that the new EventCode is from the same component */
    /* as previous events.                                            */

    if ( ESI->CmpIdx < 0 ) {
       if ( ( retval = _papi_hwi_assign_eventset( ESI, cidx)) != PAPI_OK ) {
   	      INTDBG("EXIT: Error assigning eventset to component index %d\n", cidx);
          return retval;
       }
    } else {
       if ( ESI->CmpIdx != cidx ) {
    	    INTDBG("EXIT: Event is not valid for component index %d\n", cidx);
    	    return PAPI_EINVAL;
       }
    }

    /* Make sure the event is not present and get the next free slot. */
    thisindex = get_free_EventCodeIndex( ESI, ( unsigned int ) EventCode );
    if ( thisindex < PAPI_OK ) {
       return thisindex;
    }

    INTDBG("Adding event to slot %d of EventSet %d\n",thisindex,ESI->EventSetIndex);

    /* If it is a software MPX EventSet, add it to the multiplex data structure */
    /* and this thread's multiplex list                                         */

    if ( !_papi_hwi_is_sw_multiplex( ESI ) ) {

       /* Handle preset case */
       if ( IS_PRESET(EventCode) ) {
	  int count;
	  int preset_index = EventCode & ( int ) PAPI_PRESET_AND_MASK;

	  /* Check if it's within the valid range */
	  if ( ( preset_index < 0 ) || ( preset_index >= PAPI_MAX_PRESET_EVENTS ) ) {
	     return PAPI_EINVAL;
	  }

	  /* count the number of native events in this preset */
	  count = ( int ) _papi_hwi_presets[preset_index].count;

	  /* Check if event exists */
	  if ( !count ) {
	     return PAPI_ENOEVNT;
	  }

	  /* check if the native events have been used as overflow events */
	  /* this is not allowed                                          */
	  if ( ESI->state & PAPI_OVERFLOWING ) {
	     for( i = 0; i < count; i++ ) {
		for( j = 0; j < ESI->overflow.event_counter; j++ ) {
		  if ( ESI->overflow.EventCode[j] ==(int)
			( _papi_hwi_presets[preset_index].code[i] ) ) {
		      return PAPI_ECNFLCT;
		   }
		}
	     }
	  }

	  /* Try to add the preset. */

	  remap = add_native_events( ESI,
				     _papi_hwi_presets[preset_index].code,
				     count, &ESI->EventInfoArray[thisindex] );
	  if ( remap < 0 ) {
	     return remap;
	  }
          else {
	     /* Fill in the EventCode (machine independent) information */
	     ESI->EventInfoArray[thisindex].event_code =
                                  ( unsigned int ) EventCode;
	     ESI->EventInfoArray[thisindex].derived =
				  _papi_hwi_presets[preset_index].derived_int;
	     ESI->EventInfoArray[thisindex].ops =
				  _papi_hwi_presets[preset_index].postfix;
             ESI->NumberOfEvents++;
	     _papi_hwi_map_events_to_native( ESI );

	  }
       }
       /* Handle adding Native events */
       else if ( IS_NATIVE(EventCode) ) {

	  /* Check if native event exists */
	  if ( _papi_hwi_query_native_event( ( unsigned int ) EventCode ) != PAPI_OK ) {
	     return PAPI_ENOEVNT;
	  }

	  /* check if the native events have been used as overflow events */
	  /* This is not allowed                                          */
	  if ( ESI->state & PAPI_OVERFLOWING ) {
	     for( j = 0; j < ESI->overflow.event_counter; j++ ) {
	        if ( EventCode == ESI->overflow.EventCode[j] ) {
		   return PAPI_ECNFLCT;
		}
	     }
	  }

	  /* Try to add the native event. */

	  remap = add_native_events( ESI, (unsigned int *)&EventCode, 1,
				     &ESI->EventInfoArray[thisindex] );

	  if ( remap < 0 ) {
	     return remap;
	  } else {

	     /* Fill in the EventCode (machine independent) information */
	     ESI->EventInfoArray[thisindex].event_code = 
	                                   ( unsigned int ) EventCode;
             ESI->NumberOfEvents++;
	     _papi_hwi_map_events_to_native( ESI );

	  }
       } else if ( IS_USER_DEFINED( EventCode ) ) {
		 int count;
		 int index = EventCode & PAPI_UE_AND_MASK;

		 if ( index < 0 || index >= user_defined_events_count )
		   return ( PAPI_EINVAL );

		 count = ( int ) user_defined_events[index].count;

		 for ( i = 0; i < count; i++ ) {
		   for ( j = 0; j < ESI->overflow.event_counter; j++ ) {
			 if ( ESI->overflow.EventCode[j] ==
				 (int)(user_defined_events[index].code[i]) ) {
			   return ( PAPI_EBUG );
			 }
		   }
		 }

		 remap = add_native_events( ESI,
			 user_defined_events[index].code,
			 count, &ESI->EventInfoArray[thisindex] );

		 if ( remap < 0 ) {
		   return remap;
		 } else {
		   ESI->EventInfoArray[thisindex].event_code = (unsigned int) EventCode;
		   ESI->EventInfoArray[thisindex].derived = user_defined_events[index].derived_int;
		   ESI->EventInfoArray[thisindex].ops = user_defined_events[index].postfix;
           ESI->NumberOfEvents++;
		   _papi_hwi_map_events_to_native( ESI );
		 }
       } else {

	  /* not Native, Preset, or User events */

	  return PAPI_EBUG;
       }
    }
    else {

       /* Multiplexing is special. See multiplex.c */

       retval = mpx_add_event( &ESI->multiplex.mpx_evset, EventCode,
			       ESI->domain.domain,
			       ESI->granularity.granularity );


       if ( retval < PAPI_OK ) {
	  return retval;
       }

       /* Relevant (???) */
       ESI->EventInfoArray[thisindex].event_code = ( unsigned int ) EventCode;
       ESI->EventInfoArray[thisindex].derived = NOT_DERIVED;

       ESI->NumberOfEvents++;

       /* event is in the EventInfoArray but not mapped to the NativeEvents */
       /* this causes issues if you try to set overflow on the event.       */
       /* in theory this wouldn't matter anyway.                            */
    }

    /* reinstate the overflows if any */
    retval=update_overflow( ESI );

    return retval;
}

static int
remove_native_events( EventSetInfo_t *ESI, int *nevt, int size )
{
	INTDBG( "Entry: ESI: %p, nevt: %p, size: %d\n", ESI, nevt, size);
   NativeInfo_t *native = ESI->NativeInfoArray;
   hwd_context_t *context;
   int i, j, zero = 0, retval;

   /* Remove the references to this event from the native events:
      for all the metrics in this event,
      compare to each native event in this event set,
      and decrement owners if they match  */
   for( i = 0; i < size; i++ ) {
	int cevt = _papi_hwi_eventcode_to_native(nevt[i]);
//	INTDBG( "nevt[%d]: %#x, cevt: %#x\n", i, nevt[i], cevt);
      for( j = 0; j < ESI->NativeCount; j++ ) {
	 if ((native[j].ni_event == cevt)  &&  (native[j].ni_papi_code == nevt[i]) ) {
//		INTDBG( "native[%d]: %p, ni_papi_code: %#x, ni_event: %#x, ni_position: %d, ni_owners: %d\n", 
//			j, &(native[j]), native[j].ni_papi_code, native[j].ni_event, native[j].ni_position, native[j].ni_owners);
	    native[j].ni_owners--;
	    if ( native[j].ni_owners == 0 ) {
	       zero++;
	    }
	    break;
	 }
      }
   }

   /* Remove any native events from the array if owners dropped to zero.
      The NativeInfoArray must be dense, with no empty slots, so if we
      remove an element, we must compact the list */
   for( i = 0; i < ESI->NativeCount; i++ ) {

      if ( native[i].ni_event == -1 ) continue;

      if ( native[i].ni_owners == 0 ) {
	 int copy = 0;
	 int sz = _papi_hwd[ESI->CmpIdx]->size.reg_value;
	 for( j = ESI->NativeCount - 1; j > i; j-- ) {
	    if ( native[j].ni_event == -1 || native[j].ni_owners == 0 ) continue;
	    else {
	       /* copy j into i */
	       native[i].ni_event = native[j].ni_event;
	       native[i].ni_position = native[j].ni_position;
	       native[i].ni_owners = native[j].ni_owners;
	       /* copy opaque [j].ni_bits to [i].ni_bits */
	       memcpy( native[i].ni_bits, native[j].ni_bits, ( size_t ) sz );
	       /* reset j to initialized state */
	       native[j].ni_event = -1;
	       native[j].ni_position = -1;
	       native[j].ni_owners = 0;
	       copy++;
	       break;
	    }
	 }

	 if ( copy == 0 ) {
	    /* set this structure back to empty state */
	    /* ni_owners is already 0 and contents of ni_bits doesn't matter */
	    native[i].ni_event = -1;
	    native[i].ni_position = -1;
	 }
      }
   }

	INTDBG( "ESI->NativeCount: %d, zero: %d\n", ESI->NativeCount, zero);

   /* to reset hwd_control_state values */
   ESI->NativeCount -= zero;

   /* If we removed any elements,
      clear the now empty slots, reinitialize the index, and update the count.
      Then send the info down to the component to update the hwd control structure. */
	retval = PAPI_OK;
	if ( zero ) {
      /* get the context we should use for this event set */
      context = _papi_hwi_get_context( ESI, NULL );
		retval = _papi_hwd[ESI->CmpIdx]->update_control_state( ESI->ctl_state,
														  native, ESI->NativeCount, context);
		if ( retval == PAPI_OK )
			retval = update_overflow( ESI );
	}
	return ( retval );
}

int
_papi_hwi_remove_event( EventSetInfo_t * ESI, int EventCode )
{
	int j = 0, retval, thisindex;
	EventInfo_t *array;

	thisindex =
		_papi_hwi_lookup_EventCodeIndex( ESI, ( unsigned int ) EventCode );
	if ( thisindex < PAPI_OK )
		return ( thisindex );

	/* If it is a MPX EventSet, remove it from the multiplex data structure and
	   this threads multiplex list */

	if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
		retval = mpx_remove_event( &ESI->multiplex.mpx_evset, EventCode );
		if ( retval < PAPI_OK )
			return ( retval );
	} else
		/* Remove the events hardware dependent stuff from the EventSet */
	{
		if ( IS_PRESET(EventCode) ) {
			int preset_index = EventCode & PAPI_PRESET_AND_MASK;

			/* Check if it's within the valid range */
			if ( ( preset_index < 0 ) ||
				 ( preset_index >= PAPI_MAX_PRESET_EVENTS ) )
				return PAPI_EINVAL;

			/* Check if event exists */
			if ( !_papi_hwi_presets[preset_index].count )
				return PAPI_ENOEVNT;

			/* Remove the preset event. */
			for ( j = 0; _papi_hwi_presets[preset_index].code[j] != (unsigned int)PAPI_NULL;
				  j++ );
			retval = remove_native_events( ESI, ( int * )_papi_hwi_presets[preset_index].code, j );
			if ( retval != PAPI_OK )
				return ( retval );
		} else if ( IS_NATIVE(EventCode) ) {
			/* Check if native event exists */
			if ( _papi_hwi_query_native_event( ( unsigned int ) EventCode ) !=
				 PAPI_OK )
				return PAPI_ENOEVNT;

			/* Remove the native event. */
			retval = remove_native_events( ESI, &EventCode, 1 );
			if ( retval != PAPI_OK )
				return ( retval );
		} else if ( IS_USER_DEFINED( EventCode ) ) {
		  int index = EventCode & PAPI_UE_AND_MASK;

		  if ( (index < 0) || (index >= user_defined_events_count) )
			return ( PAPI_EINVAL );

		  for( j = 0; j < PAPI_EVENTS_IN_DERIVED_EVENT &&
			  user_defined_events[index].code[j] != 0; j++ ) {
			retval = remove_native_events( ESI, ( int * )user_defined_events[index].code, j);

			if ( retval != PAPI_OK )
			  return ( retval );
		  }
		} else
			return ( PAPI_ENOEVNT );
	}
	array = ESI->EventInfoArray;

	/* Compact the Event Info Array list if it's not the last event */
	/* clear the newly empty slot in the array */
	for ( ; thisindex < ESI->NumberOfEvents - 1; thisindex++ )
		array[thisindex] = array[thisindex + 1];


	array[thisindex].event_code = ( unsigned int ) PAPI_NULL;
	for ( j = 0; j < PAPI_EVENTS_IN_DERIVED_EVENT; j++ )
		array[thisindex].pos[j] = PAPI_NULL;
	array[thisindex].ops = NULL;
	array[thisindex].derived = NOT_DERIVED;
	ESI->NumberOfEvents--;

	return ( PAPI_OK );
}

int
_papi_hwi_read( hwd_context_t * context, EventSetInfo_t * ESI,
				long long *values )
{
	INTDBG("ENTER: context: %p, ESI: %p, values: %p\n", context, ESI, values);
	int retval;
	long long *dp = NULL;
	int i, index;

	retval = _papi_hwd[ESI->CmpIdx]->read( context, ESI->ctl_state,
					       &dp, ESI->state );
	if ( retval != PAPI_OK ) {
		INTDBG("EXIT: retval: %d\n", retval);
	   return retval;
	}

	/* This routine distributes hardware counters to software counters in the
	   order that they were added. Note that the higher level
	   EventInfoArray[i] entries may not be contiguous because the user
	   has the right to remove an event.
	   But if we do compaction after remove event, this function can be
	   changed.
	 */

	for ( i = 0; i != ESI->NumberOfEvents; i++ ) {

		index = ESI->EventInfoArray[i].pos[0];

		if ( index == -1 )
			continue;

		INTDBG( "ESI->EventInfoArray: %p, pos[%d]: %d, dp[%d]: %lld, derived[%d]: %#x\n", ESI->EventInfoArray, i, index, index, dp[index], i, ESI->EventInfoArray[i].derived );

		/* If this is not a derived event */

		if ( ESI->EventInfoArray[i].derived == NOT_DERIVED ) {
			values[i] = dp[index];
			INTDBG( "value: %#llx\n", values[i] );
		} else {			 /* If this is a derived event */
			values[i] = handle_derived( &ESI->EventInfoArray[i], dp );
#ifdef DEBUG
			if ( values[i] < ( long long ) 0 ) {
				INTDBG( "Derived Event is negative!!: %lld\n", values[i] );
			}
			INTDBG( "derived value: %#llx \n", values[i] );
#endif
		}
	}

	INTDBG("EXIT: PAPI_OK\n");
	return PAPI_OK;
}

int
_papi_hwi_cleanup_eventset( EventSetInfo_t * ESI )
{
   int i, j, num_cntrs, retval;
   hwd_context_t *context;
   int EventCode;
   NativeInfo_t *native;
   if ( !_papi_hwi_invalid_cmp( ESI->CmpIdx ) ) {
   num_cntrs = _papi_hwd[ESI->CmpIdx]->cmp_info.num_mpx_cntrs;

   for(i=0;i<num_cntrs;i++) {

      EventCode=ESI->EventInfoArray[i].event_code;

      /* skip if event not there */
      if ( EventCode == PAPI_NULL ) continue;

      /* If it is a MPX EventSet, remove it from the multiplex */
      /* data structure and this thread's multiplex list */

      if ( _papi_hwi_is_sw_multiplex( ESI ) ) {
	 retval = mpx_remove_event( &ESI->multiplex.mpx_evset, EventCode );
	 if ( retval < PAPI_OK )
	    return retval;
      } else {

	  native = ESI->NativeInfoArray;

	  /* clear out ESI->NativeInfoArray */
	  /* do we really need to do this, seeing as we free() it later? */

	  for( j = 0; j < ESI->NativeCount; j++ ) {
	     native[j].ni_event = -1;
	     native[j].ni_position = -1;
	     native[j].ni_owners = 0;
	     /* native[j].ni_bits?? */
	  }
      }

      /* do we really need to do this, seeing as we free() it later? */
      ESI->EventInfoArray[i].event_code= ( unsigned int ) PAPI_NULL;
      for( j = 0; j < PAPI_EVENTS_IN_DERIVED_EVENT; j++ ) {
	  ESI->EventInfoArray[i].pos[j] = PAPI_NULL;
      }
      ESI->EventInfoArray[i].ops = NULL;
      ESI->EventInfoArray[i].derived = NOT_DERIVED;
   }

   context = _papi_hwi_get_context( ESI, NULL );
   /* calling with count of 0 equals a close? */
   retval = _papi_hwd[ESI->CmpIdx]->update_control_state( ESI->ctl_state,
			       NULL, 0, context);
   if (retval!=PAPI_OK) {
     return retval;
   }
   }

   ESI->CmpIdx = -1;
   ESI->NumberOfEvents = 0;
   ESI->NativeCount = 0;

   if ( ( ESI->state & PAPI_MULTIPLEXING ) && ESI->multiplex.mpx_evset )
		   papi_free( ESI->multiplex.mpx_evset );

   if ( ( ESI->state & PAPI_CPU_ATTACH ) && ESI->CpuInfo )
		   _papi_hwi_shutdown_cpu( ESI->CpuInfo );

   if ( ESI->ctl_state )
      papi_free( ESI->ctl_state );

   if ( ESI->sw_stop )
      papi_free( ESI->sw_stop );

   if ( ESI->hw_start )
      papi_free( ESI->hw_start );

   if ( ESI->EventInfoArray )
      papi_free( ESI->EventInfoArray );

   if ( ESI->NativeInfoArray )
      papi_free( ESI->NativeInfoArray );

   if ( ESI->NativeBits )
      papi_free( ESI->NativeBits );

   if ( ESI->overflow.deadline )
      papi_free( ESI->overflow.deadline );

   if ( ESI->profile.prof )
      papi_free( ESI->profile.prof );

   ESI->ctl_state = NULL;
   ESI->sw_stop = NULL;
   ESI->hw_start = NULL;
   ESI->EventInfoArray = NULL;
   ESI->NativeInfoArray = NULL;
   ESI->NativeBits = NULL;

   memset( &ESI->domain, 0x0, sizeof(EventSetDomainInfo_t) );
   memset( &ESI->granularity, 0x0, sizeof(EventSetGranularityInfo_t) );
   memset( &ESI->overflow, 0x0, sizeof(EventSetOverflowInfo_t) );
   memset( &ESI->multiplex, 0x0, sizeof(EventSetMultiplexInfo_t) );
   memset( &ESI->attach, 0x0, sizeof(EventSetAttachInfo_t) );
   memset( &ESI->cpu, 0x0, sizeof(EventSetCpuInfo_t) );
   memset( &ESI->profile, 0x0, sizeof(EventSetProfileInfo_t) );
   memset( &ESI->inherit, 0x0, sizeof(EventSetInheritInfo_t) );

   ESI->CpuInfo = NULL;

   return PAPI_OK;
}

int
_papi_hwi_convert_eventset_to_multiplex( _papi_int_multiplex_t * mpx )
{
	int retval, i, j = 0, *mpxlist = NULL;
	EventSetInfo_t *ESI = mpx->ESI;
	int flags = mpx->flags;

	/* If there are any events in the EventSet,
	   convert them to multiplex events */

	if ( ESI->NumberOfEvents ) {

		mpxlist =
			( int * ) papi_malloc( sizeof ( int ) *
								   ( size_t ) ESI->NumberOfEvents );
		if ( mpxlist == NULL )
			return ( PAPI_ENOMEM );

		/* Build the args to MPX_add_events(). */

		/* Remember the EventInfoArray can be sparse
		   and the data can be non-contiguous */

		for ( i = 0; i < EventInfoArrayLength( ESI ); i++ )
			if ( ESI->EventInfoArray[i].event_code !=
				 ( unsigned int ) PAPI_NULL )
				mpxlist[j++] = ( int ) ESI->EventInfoArray[i].event_code;

		/* Resize the EventInfo_t array */

		if ( ( _papi_hwd[ESI->CmpIdx]->cmp_info.kernel_multiplex == 0 ) ||
			 ( ( _papi_hwd[ESI->CmpIdx]->cmp_info.kernel_multiplex ) &&
			   ( flags & PAPI_MULTIPLEX_FORCE_SW ) ) ) {
			retval =
				MPX_add_events( &ESI->multiplex.mpx_evset, mpxlist, j,
								ESI->domain.domain,
								ESI->granularity.granularity );
			if ( retval != PAPI_OK ) {
				papi_free( mpxlist );
				return ( retval );
			}
		}

		papi_free( mpxlist );
	}

	/* Update the state before initialization! */

	ESI->state |= PAPI_MULTIPLEXING;
	if ( _papi_hwd[ESI->CmpIdx]->cmp_info.kernel_multiplex &&
		 ( flags & PAPI_MULTIPLEX_FORCE_SW ) )
		ESI->multiplex.flags = PAPI_MULTIPLEX_FORCE_SW;
	ESI->multiplex.ns = ( int ) mpx->ns;

	return ( PAPI_OK );
}

#include "components_config.h"

int papi_num_components = ( sizeof ( _papi_hwd ) / sizeof ( *_papi_hwd ) ) - 1;

/*
 * Routine that initializes all available components.
 * A component is available if a pointer to its info vector
 * appears in the NULL terminated_papi_hwd table.
 * Modified to accept an arg: 0=do not init perf_event or 
 * perf_event_uncore. 1=init ONLY perf_event or perf_event_uncore.
 */
int
_papi_hwi_init_global( int PE_OR_PEU )
{
        int retval, is_pe_peu, i = 0;

	retval = _papi_hwi_innoculate_os_vector( &_papi_os_vector );
	if ( retval != PAPI_OK ) {
	   return retval;
	}

	while ( _papi_hwd[i] ) {
      is_pe_peu = 0;
      if (strcmp(_papi_hwd[i]->cmp_info.name, "perf_event") == 0) is_pe_peu=1;
      if (strcmp(_papi_hwd[i]->cmp_info.name, "perf_event_uncore") == 0) is_pe_peu=1;
	   retval = _papi_hwi_innoculate_vector( _papi_hwd[i] );
	   if ( retval != PAPI_OK ) {
	      return retval;
	   }

	   /* We can be disabled by user before init */
	   if (!_papi_hwd[i]->cmp_info.disabled && (PE_OR_PEU == is_pe_peu)) {
	      retval = _papi_hwd[i]->init_component( i );

	      /* Do some sanity checking */
	      if (retval==PAPI_OK) {
		if (_papi_hwd[i]->cmp_info.num_cntrs >
		    _papi_hwd[i]->cmp_info.num_mpx_cntrs) {
		  fprintf(stderr,"Warning!  num_cntrs %d is more than num_mpx_cntrs %d for component %s\n",
                        _papi_hwd[i]->cmp_info.num_cntrs,
                        _papi_hwd[i]->cmp_info.num_mpx_cntrs,
                        _papi_hwd[i]->cmp_info.name);
		}

	      }
	   }

	   i++;
	}
	return PAPI_OK;
}

/* Machine info struct initialization using defaults */
/* See _papi_mdi definition in papi_internal.h       */

int
_papi_hwi_init_global_internal( void )
{

	int retval;

	memset(&_papi_hwi_system_info,0x0,sizeof( _papi_hwi_system_info ));

	memset( _papi_hwi_using_signal,0x0,sizeof( _papi_hwi_using_signal ));

	/* Global struct to maintain EventSet mapping */
	retval = allocate_eventset_map( &_papi_hwi_system_info.global_eventset_map );
	if ( retval != PAPI_OK ) {
		return retval;
	}

	_papi_hwi_system_info.pid = 0;	/* Process identifier */

	/* PAPI_hw_info_t struct */
	memset(&(_papi_hwi_system_info.hw_info),0x0,sizeof(PAPI_hw_info_t));
	return PAPI_OK;
}

void
_papi_hwi_shutdown_global_internal( void )
{
    int i = 0;
	_papi_hwi_cleanup_all_presets(  );

	_papi_hwi_cleanup_errors( );

	_papi_hwi_lock( INTERNAL_LOCK );

    for( i = 0; i < num_native_events; i++){
        free(_papi_native_events[i].evt_name);
    }

    free(_papi_native_events);
    _papi_native_events = NULL;        // In case a new library init is done.
    num_native_events=0;               // .. 
    num_native_chunks=0;               // .. 

    _papi_hwi_free_papi_event_string();

	papi_free(  _papi_hwi_system_info.global_eventset_map.dataSlotArray );
	memset(  &_papi_hwi_system_info.global_eventset_map,
		 0x00, sizeof ( DynamicArray_t ) );

	_papi_hwi_unlock( INTERNAL_LOCK );

	if ( _papi_hwi_system_info.shlib_info.map ) {
		papi_free( _papi_hwi_system_info.shlib_info.map );
	}
	memset( &_papi_hwi_system_info, 0x0, sizeof ( _papi_hwi_system_info ) );

}



void
_papi_hwi_dummy_handler( int EventSet, void *address, long long overflow_vector,
						 void *context )
{
	/* This function is not used and shouldn't be called. */
	( void ) EventSet;		 /*unused */
	( void ) address;		 /*unused */
	( void ) overflow_vector;	/*unused */
	( void ) context;		 /*unused */
	return;
}

static long long
handle_derived_add( int *position, long long *from )
{
	int pos, i;
	long long retval = 0;

	i = 0;
	while ( i < PAPI_EVENTS_IN_DERIVED_EVENT ) {
		pos = position[i++];
		if ( pos == PAPI_NULL )
			break;
		INTDBG( "Compound event, adding %lld to %lld\n", from[pos], retval );
		retval += from[pos];
	}
	return ( retval );
}

static long long
handle_derived_subtract( int *position, long long *from )
{
	int pos, i;
	long long retval = from[position[0]];

	i = 1;
	while ( i < PAPI_EVENTS_IN_DERIVED_EVENT ) {
		pos = position[i++];
		if ( pos == PAPI_NULL )
			break;
		INTDBG( "Compound event, subtracting pos=%d  %lld from %lld\n", pos,
				from[pos], retval );
		retval -= from[pos];
	}
	return ( retval );
}

static long long
units_per_second( long long units, long long cycles )
{
   return ( ( units * (long long) _papi_hwi_system_info.hw_info.cpu_max_mhz *
		      (long long) 1000000 ) / cycles );
}

static long long
handle_derived_ps( int *position, long long *from )
{
	return ( units_per_second( from[position[1]], from[position[0]] ) );
}
static long long
handle_derived_add_ps( int *position, long long *from )
{
	long long tmp = handle_derived_add( position + 1, from );
	return ( units_per_second( tmp, from[position[0]] ) );
}

/* this function implement postfix calculation, it reads in a string where I use:
      |      as delimiter
      N2     indicate No. 2 native event in the derived preset
      +, -, *, /  as operator
      #      as MHZ(million hz) got from  _papi_hwi_system_info.hw_info.cpu_max_mhz*1000000.0

  Haihang (you@cs.utk.edu)
*/ 
 static long long
 _papi_hwi_postfix_calc( EventInfo_t * evi, long long *hw_counter )
 {
        char *point = evi->ops, operand[16];
        double stack[PAPI_EVENTS_IN_DERIVED_EVENT];
       int i, val, top = 0;

       INTDBG("ENTER: evi: %p, evi->ops: %p (%s), evi->pos[0]: %d, evi->pos[1]: %d, hw_counter: %p (%lld %lld)\n",
              evi, evi->ops, evi->ops, evi->pos[0], evi->pos[1], hw_counter, hw_counter[0], hw_counter[1]);

        memset(&stack,0,PAPI_EVENTS_IN_DERIVED_EVENT*sizeof(double));

        while ( *point != '\0' ) {
               if ( *point == '|' ) {  /* consume '|' characters */
                        point++;
                } else if ( *point == 'N' ) {   /* to get count for each native event */
                        point++;
                       i = 0;
                       while ( isdigit(*point) ) {
                               assert(i<16);
                                operand[i] = *point;
                                point++;
                                i++;
                       }
                       assert(0<i && i<16);
                        operand[i] = '\0';
                       val = atoi( operand );
                       assert( top < PAPI_EVENTS_IN_DERIVED_EVENT );
                       assert( 0 <= val && val < PAPI_EVENTS_IN_DERIVED_EVENT );
                       stack[top] = ( double ) hw_counter[evi->pos[val]];
                        top++;
               } else if ( *point == '#' ) {   /* to get mhz */
                        point++;
                       assert( top < PAPI_EVENTS_IN_DERIVED_EVENT );
                        stack[top] = _papi_hwi_system_info.hw_info.cpu_max_mhz * 1000000.0;
                        top++;
               } else if ( isdigit( *point ) ) {
                        i = 0;
                       while ( isdigit(*point) ) {
                               assert(i<16);
                                operand[i] = *point;
                                point++;
                                i++;
                       }
                       assert(0<i && i<16);
                        operand[i] = '\0';
                       assert( top < PAPI_EVENTS_IN_DERIVED_EVENT );
                        stack[top] = atoi( operand );
                        top++;
                } else if ( *point == '+' ) {   /* + calculation */
                       point++;
                       assert(top >= 2);
                        stack[top - 2] += stack[top - 1];
                        top--;
                } else if ( *point == '-' ) {   /* - calculation */
                       point++;
                       assert(top >= 2);
                        stack[top - 2] -= stack[top - 1];
                        top--;
                } else if ( *point == '*' ) {   /* * calculation */
                       point++;
                       assert(top >= 2);
                        stack[top - 2] *= stack[top - 1];
                        top--;
                } else if ( *point == '/' ) {   /* / calculation */
                       point++;
                       assert(top >= 2);
                       /* FIXME should handle runtime divide by zero */
                        stack[top - 2] /= stack[top - 1];
                        top--;
               } else { /* flag an error parsing the preset */
                       PAPIERROR( "BUG! Unable to parse \"%s\"", evi->ops );
                       return ( long long ) stack[0];
                }
        }
        assert(top == 1);
        INTDBG("EXIT: stack[0]: %lld\n", (long long)stack[0]);
        return ( long long ) stack[0];
 }

static long long
handle_derived( EventInfo_t * evi, long long *from )
{
	INTDBG("ENTER: evi: %p, evi->derived: %d, from: %p\n", evi, evi->derived, from);
	switch ( evi->derived ) {
	case DERIVED_ADD:
		return ( handle_derived_add( evi->pos, from ) );
	case DERIVED_ADD_PS:
		return ( handle_derived_add_ps( evi->pos, from ) );
	case DERIVED_SUB:
		return ( handle_derived_subtract( evi->pos, from ) );
	case DERIVED_PS:
		return ( handle_derived_ps( evi->pos, from ) );
	case DERIVED_POSTFIX:
		return ( _papi_hwi_postfix_calc( evi, from ) );
	case DERIVED_CMPD:		 /* This type has existed for a long time, but was never implemented.
							    Probably because its a no-op. However, if it's in a header, it
							    should be supported. As I found out when I implemented it in 
							    Pentium 4 for testing...dkt */
		return ( from[evi->pos[0]] );
	default:
		PAPIERROR( "BUG! Unknown derived command %d, returning 0", evi->derived );
		INTDBG("EXIT: Unknown derived command %d\n", evi->derived);
		return ( ( long long ) 0 );
	}
}


/* table matching derived types to derived strings.
   used by get_info, encode_event, xml translator
*/
static const hwi_describe_t _papi_hwi_derived[] = {
  {NOT_DERIVED, "NOT_DERIVED", "Do nothing"},
  {DERIVED_ADD, "DERIVED_ADD", "Add counters"},
  {DERIVED_PS, "DERIVED_PS",
   "Divide by the cycle counter and convert to seconds"},
  {DERIVED_ADD_PS, "DERIVED_ADD_PS",
   "Add 2 counters then divide by the cycle counter and xl8 to secs."},
  {DERIVED_CMPD, "DERIVED_CMPD",
   "Event lives in first counter but takes 2 or more codes"},
  {DERIVED_SUB, "DERIVED_SUB", "Sub all counters from first counter"},
  {DERIVED_POSTFIX, "DERIVED_POSTFIX",
   "Process counters based on specified postfix string"},
  {DERIVED_INFIX, "DERIVED_INFIX",
   "Process counters based on specified infix string"},
  {-1, NULL, NULL}
};

/* _papi_hwi_derived_type:
   Helper routine to extract a derived type from a derived string
   returns type value if found, otherwise returns -1
*/
int
_papi_hwi_derived_type( char *tmp, int *code )
{
  int i = 0;
  while ( _papi_hwi_derived[i].name != NULL ) {
    if ( strcasecmp( tmp, _papi_hwi_derived[i].name ) == 0 ) {
      *code = _papi_hwi_derived[i].value;
      return PAPI_OK;
    }
    i++;
  }
  INTDBG( "Invalid derived string %s\n", tmp );
  return PAPI_EINVAL;
}


/* _papi_hwi_derived_string:
   Helper routine to extract a derived string from a derived type
   copies derived type string into derived if found,
   otherwise returns PAPI_EINVAL
*/
static int
_papi_hwi_derived_string( int type, char *derived, int len )
{
  int j;

  for ( j = 0; _papi_hwi_derived[j].value != -1; j++ ) {
    if ( _papi_hwi_derived[j].value == type ) {
      strncpy( derived, _papi_hwi_derived[j].name, ( size_t )\
	       len );
      return PAPI_OK;
    }
  }
  INTDBG( "Invalid derived type %d\n", type );
  return PAPI_EINVAL;
}


/* _papi_hwi_get_preset_event_info:
   Assumes EventCode contains a valid preset code.
   But defensive programming says check for NULL pointers.
   Returns a filled in PAPI_event_info_t structure containing
   descriptive strings and values for the specified preset event.
*/
int
_papi_hwi_get_preset_event_info( int EventCode, PAPI_event_info_t * info )
{
	INTDBG("ENTER: EventCode: %#x, info: %p\n", EventCode, info);

	int i = EventCode & PAPI_PRESET_AND_MASK;
	unsigned int j;

	if ( _papi_hwi_presets[i].symbol ) {	/* if the event is in the preset table */
      // since we are setting the whole structure to zero the strncpy calls below will 
      // be leaving NULL terminates strings as long as they copy 1 less byte than the 
      // buffer size of the field.

	   INTDBG("ENTER: Configuring: %s\n", _papi_hwi_presets[i].symbol);

	   memset( info, 0, sizeof ( PAPI_event_info_t ) );

		/* set up eventcode and name */
	   info->event_code = ( unsigned int ) EventCode;
	   strncpy( info->symbol, _papi_hwi_presets[i].symbol,
	            sizeof(info->symbol)-1);

		/* set up short description, if available */
	   if ( _papi_hwi_presets[i].short_descr != NULL ) {
	      strncpy( info->short_descr, _papi_hwi_presets[i].short_descr,
		          sizeof ( info->short_descr )-1 );
	   }

		/* set up long description, if available */
	   if ( _papi_hwi_presets[i].long_descr != NULL ) {
	      strncpy( info->long_descr,  _papi_hwi_presets[i].long_descr,
		          sizeof ( info->long_descr )-1 );
	   }

	   info->event_type = _papi_hwi_presets[i].event_type;
	   info->count = _papi_hwi_presets[i].count;


		/* set up if derived event */
	   _papi_hwi_derived_string( _papi_hwi_presets[i].derived_int,
				     info->derived,  sizeof ( info->derived )-1 );

	   if ( _papi_hwi_presets[i].postfix != NULL ) {
	      strncpy( info->postfix, _papi_hwi_presets[i].postfix,
		          sizeof ( info->postfix )-1 );
	   }

	   for(j=0;j < info->count; j++) {

		/* make sure the name exists before trying to copy it */
		/* that can happen if an event is in the definition in */
		/* papi_events.csv but the event is unsupported on the cpu */
		/* ideally that should never happen, but also ideally */
		/* we wouldn't segfault if it does */

	      if (_papi_hwi_presets[i].name[j]==NULL) {
		INTDBG("ERROR in event definition of %s\n", _papi_hwi_presets[i].symbol);
			   return PAPI_ENOEVNT;
		}
		else {
			info->code[j]=_papi_hwi_presets[i].code[j];
			strncpy(info->name[j], _papi_hwi_presets[i].name[j],
				sizeof(info->name[j])-1);
		}
	   }

	   if ( _papi_hwi_presets[i].note != NULL ) {
	      strncpy( info->note, _papi_hwi_presets[i].note,
		          sizeof ( info->note )-1 );
	   }

	   return PAPI_OK;
	} else {
	   return PAPI_ENOEVNT;
	}
}


/* _papi_hwi_get_user_event_info:
   Assumes EventCode contains a valid user event code.
   But defensive programming says check for NULL pointers.
   Returns a filled in PAPI_event_info_t structure containing
   descriptive strings and values for the specified preset event.
*/
int
_papi_hwi_get_user_event_info( int EventCode, PAPI_event_info_t * info )
{
	INTDBG("ENTER: EventCode: %#x, info: %p\n", EventCode, info);

	unsigned int i = EventCode & PAPI_UE_AND_MASK;
	unsigned int j;

	// if event code not in valid range, return error
	if (i >= PAPI_MAX_USER_EVENTS) {
		INTDBG("EXIT: Invalid event index: %d, max value is: %d\n", i, PAPI_MAX_USER_EVENTS - 1);
		return( PAPI_ENOEVNT );
	}

	if ( user_defined_events[i].symbol == NULL) {	/* if the event is in the preset table */
		INTDBG("EXIT: Event symbol for this event is NULL\n");
		return PAPI_ENOEVNT;
	}

	/* set whole structure to 0 */
	memset( info, 0, sizeof ( PAPI_event_info_t ) );

	info->event_code = ( unsigned int ) EventCode;
	strncpy( info->symbol, user_defined_events[i].symbol,
		sizeof(info->symbol)-1);

	if ( user_defined_events[i].short_descr != NULL )
		strncpy( info->short_descr, user_defined_events[i].short_descr,
			sizeof(info->short_descr)-1);

	if ( user_defined_events[i].long_descr != NULL )
		strncpy( info->long_descr,  user_defined_events[i].long_descr,
			sizeof(info->long_descr)-1);

//	info->event_type = user_defined_events[i].event_type;
	info->count = user_defined_events[i].count;

	_papi_hwi_derived_string( user_defined_events[i].derived_int,
			info->derived,  sizeof(info->derived)-1);

	if ( user_defined_events[i].postfix != NULL )
		strncpy( info->postfix, user_defined_events[i].postfix,
			sizeof(info->postfix)-1);

	for(j=0;j < info->count; j++) {
		info->code[j]=user_defined_events[i].code[j];
		INTDBG("info->code[%d]: %#x\n", j, info->code[j]);
		strncpy(info->name[j], user_defined_events[i].name[j], sizeof(info->name[j])-1);
	}

	if ( user_defined_events[i].note != NULL ) {
		strncpy( info->note, user_defined_events[i].note, sizeof(info->note)-1);
	}

	INTDBG("EXIT: PAPI_OK: event_code: %#x, symbol: %s, short_desc: %s, long_desc: %s\n", info->event_code, info->symbol, info->short_descr, info->long_descr);
	return PAPI_OK;
}


/* Returns PAPI_OK if native EventCode found, or PAPI_ENOEVNT if not;
   Used to enumerate the entire array, e.g. for native_avail.c */
int
_papi_hwi_query_native_event( unsigned int EventCode )
{
	INTDBG("ENTER: EventCode: %#x\n", EventCode);
   char name[PAPI_HUGE_STR_LEN];      /* probably overkill, */
                                      /* but should always be big enough */
   int cidx;
   int nevt_code;

   cidx = _papi_hwi_component_index( EventCode );
   if (cidx<0) {
	   INTDBG("EXIT: PAPI_ENOCMP\n");
	   return PAPI_ENOCMP;
   }

   // save event code so components can get it with call to: _papi_hwi_get_papi_event_code()
   _papi_hwi_set_papi_event_code(EventCode, 0);

	if ((nevt_code = _papi_hwi_eventcode_to_native(EventCode)) < 0) {
		INTDBG("EXIT: nevt_code: %d\n", nevt_code);
		return nevt_code;
	}
   int ret = _papi_hwd[cidx]->ntv_code_to_name( (unsigned int)nevt_code, name, sizeof(name));

   INTDBG("EXIT: ret: %d\n", ret);
   return (ret);
}

/* Converts an ASCII name into a native event code usable by other routines
   Returns code = 0 and PAPI_OK if name not found.
   This allows for sparse native event arrays */
int
_papi_hwi_native_name_to_code( const char *in, int *out )
{
	INTDBG("ENTER: in: %s, out: %p\n", in, out);

	int retval = PAPI_ENOEVNT;
	char name[PAPI_HUGE_STR_LEN];	/* make sure it's big enough */

	unsigned int i;
	int cidx;
	char *full_event_name;

	if (in == NULL) {
		INTDBG("EXIT: PAPI_EINVAL\n");
		return PAPI_EINVAL;
	}

	full_event_name = strdup(in);

	in = _papi_hwi_strip_component_prefix(in);

	// look in each component
	for(cidx=0; cidx < papi_num_components; cidx++) {

		if (_papi_hwd[cidx]->cmp_info.disabled &&
            _papi_hwd[cidx]->cmp_info.disabled != PAPI_EDELAY_INIT)
            continue;

		// if this component does not support the pmu
		// which defines this event, no need to call it
		if (is_supported_by_component(cidx, full_event_name) == 0) {
			continue;
		}

		INTDBG("cidx: %d, name: %s, event: %s\n",
			cidx, _papi_hwd[cidx]->cmp_info.name, in);

		// show that we do not have an event code yet
		// (the component may create one and update this info)
		// this also clears any values left over from a previous call
		_papi_hwi_set_papi_event_code(-1, -1);


		// if component has a ntv_name_to_code function, use it to get event code
		if (_papi_hwd[cidx]->ntv_name_to_code != NULL) {
			// try and get this events event code
			retval = _papi_hwd[cidx]->ntv_name_to_code( in, ( unsigned * ) out );
			if (retval==PAPI_OK) {
				*out = _papi_hwi_native_to_eventcode(cidx, *out, -1, in);
				free (full_event_name);
				INTDBG("EXIT: PAPI_OK  event: %s code: %#x\n", in, *out);
				return PAPI_OK;
			}
		} else {
			// force the code through the work around
			retval = PAPI_ECMP;
		}

		/* If not implemented, work around */
		if ( retval==PAPI_ECMP) {
			i = 0;
			retval = _papi_hwd[cidx]->ntv_enum_events( &i, PAPI_ENUM_FIRST );
			if (retval != PAPI_OK) {
				free (full_event_name);
				INTDBG("EXIT: retval: %d\n", retval);
				return retval;
			}

//			_papi_hwi_lock( INTERNAL_LOCK );

			do {
				// save event code so components can get it with call to: _papi_hwi_get_papi_event_code()
				_papi_hwi_set_papi_event_code(i, 0);
				retval = _papi_hwd[cidx]->ntv_code_to_name(i, name, sizeof(name));
				/* printf("%#x\nname =|%s|\ninput=|%s|\n", i, name, in); */
				if ( retval == PAPI_OK && in != NULL) {
					if ( strcasecmp( name, in ) == 0 ) {
						*out = _papi_hwi_native_to_eventcode(cidx, i, -1, name);
						free (full_event_name);
						INTDBG("EXIT: PAPI_OK, event: %s, code: %#x\n", in, *out);
						return PAPI_OK;
					}
					retval = PAPI_ENOEVNT;
				} else {
					*out = 0;
					retval = PAPI_ENOEVNT;
					break;
				}
			} while ( ( _papi_hwd[cidx]->ntv_enum_events( &i, PAPI_ENUM_EVENTS ) == PAPI_OK ) );

//			_papi_hwi_unlock( INTERNAL_LOCK );
		}
	}

	free (full_event_name);
	INTDBG("EXIT: retval: %d\n", retval);

	return retval;
}

/* Returns event name based on native event code.
   Returns NULL if name not found */
int
_papi_hwi_native_code_to_name( unsigned int EventCode,
			       char *hwi_name, int len )
{
	INTDBG("ENTER: EventCode: %#x, hwi_name: %p, len: %d\n", EventCode, hwi_name, len);
  int cidx;
  int retval;
  int nevt_code;

  cidx = _papi_hwi_component_index( EventCode );
  if (cidx<0) return PAPI_ENOEVNT;

  if ( EventCode & PAPI_NATIVE_MASK ) {
	  // save event code so components can get it with call to: _papi_hwi_get_papi_event_code()
	  _papi_hwi_set_papi_event_code(EventCode, 0);

	if ((nevt_code = _papi_hwi_eventcode_to_native(EventCode)) < 0) {
		INTDBG("EXIT: nevt_code: %d\n", nevt_code);
		return nevt_code;
	}
	if ( (retval = _papi_hwd[cidx]->ntv_code_to_name(
						(unsigned int)nevt_code,
						hwi_name, len) ) == PAPI_OK ) {
			retval = _papi_hwi_prefix_component_name( _papi_hwd[cidx]->cmp_info.short_name, 
											 hwi_name, hwi_name, len);
			INTDBG("EXIT: retval: %d\n", retval);
			return retval;
	}
	INTDBG("EXIT: retval: %d\n", retval);
	return (retval);
  }
  INTDBG("EXIT: PAPI_ENOEVNT\n");
  return PAPI_ENOEVNT;
}



/* The native event equivalent of PAPI_get_event_info */
int
_papi_hwi_get_native_event_info( unsigned int EventCode,
				 PAPI_event_info_t *info )
{
	  INTDBG("ENTER: EventCode: %#x, info: %p\n", EventCode, info);
    int retval;
    int cidx;
    int nevt_code;

    cidx = _papi_hwi_component_index( EventCode );
    if (cidx<0) return PAPI_ENOCMP;

    if (_papi_hwd[cidx]->cmp_info.disabled &&
        _papi_hwd[cidx]->cmp_info.disabled != PAPI_EDELAY_INIT)
        return PAPI_ENOCMP;

    if ( EventCode & PAPI_NATIVE_MASK ) {
        // save event code so components can get it with call to: _papi_hwi_get_papi_event_code()
        _papi_hwi_set_papi_event_code(EventCode, 0);

       /* clear the event info */
       memset( info, 0, sizeof ( PAPI_event_info_t ) );
       info->event_code = ( unsigned int ) EventCode;
       info->component_index = (unsigned int) cidx;
       retval = _papi_hwd[cidx]->ntv_code_to_info(
			      _papi_hwi_eventcode_to_native(EventCode), info);

       /* If component error, it's missing the ntv_code_to_info vector */
       /* so we'll have to fake it.                                    */
       if ( retval == PAPI_ECMP ) {


	  INTDBG("missing NTV_CODE_TO_INFO, faking\n");
	  /* Fill in the info structure */

		if ((nevt_code = _papi_hwi_eventcode_to_native(EventCode)) < 0) {
			INTDBG("EXIT: nevt_code: %d\n", nevt_code);
			return nevt_code;
		}
	  if ( (retval = _papi_hwd[cidx]->ntv_code_to_name(
				    (unsigned int)nevt_code,
				    info->symbol,
				    sizeof(info->symbol)) ) == PAPI_OK ) {

	  } else {
	     INTDBG("EXIT: retval: %d\n", retval);
	     return retval;
	  }

		if ((nevt_code = _papi_hwi_eventcode_to_native(EventCode)) <0) {
			INTDBG("EXIT: nevt_code: %d\n", nevt_code);
			return nevt_code;
		}
	  retval = _papi_hwd[cidx]->ntv_code_to_descr(
				     (unsigned int)nevt_code,
				     info->long_descr,
				     sizeof ( info->long_descr));
	  if (retval!=PAPI_OK) {
	     INTDBG("Failed ntv_code_to_descr()\n");
	  }

       }
	   retval = _papi_hwi_prefix_component_name(
						_papi_hwd[cidx]->cmp_info.short_name, 
						info->symbol,
						info->symbol,
						sizeof(info->symbol) );

       INTDBG("EXIT: retval: %d\n", retval);
       return retval;
    }

	INTDBG("EXIT: PAPI_ENOEVNT\n");
    return PAPI_ENOEVNT;
}

EventSetInfo_t *
_papi_hwi_lookup_EventSet( int eventset )
{
	const DynamicArray_t *map = &_papi_hwi_system_info.global_eventset_map;
	EventSetInfo_t *set;

	if ( ( eventset < 0 ) || ( eventset > map->totalSlots ) )
		return ( NULL );

	set = map->dataSlotArray[eventset];
#ifdef DEBUG
	if ( ( ISLEVEL( DEBUG_THREADS ) ) && ( _papi_hwi_thread_id_fn ) &&
		 ( set->master->tid != _papi_hwi_thread_id_fn(  ) ) )
		return ( NULL );
#endif

	return ( set );
}

int
_papi_hwi_is_sw_multiplex(EventSetInfo_t *ESI)
{
   /* Are we multiplexing at all */
   if ( ( ESI->state & PAPI_MULTIPLEXING ) == 0 ) {
      return 0;
   }

   /* Does the component support kernel multiplexing */
   if ( _papi_hwd[ESI->CmpIdx]->cmp_info.kernel_multiplex ) {
      /* Have we forced software multiplexing */
      if ( ESI->multiplex.flags == PAPI_MULTIPLEX_FORCE_SW ) {
	 return 1;
      }
      /* Nope, using hardware multiplexing */
      return 0;
   }

   /* We are multiplexing but the component does not support hardware */

   return 1;

}

hwd_context_t *
_papi_hwi_get_context( EventSetInfo_t * ESI, int *is_dirty )
{
	INTDBG("Entry: ESI: %p, is_dirty: %p\n", ESI, is_dirty);
	int dirty_ctx;
	hwd_context_t *ctx=NULL;

	/* assume for now the control state is clean (last updated by this ESI) */
	dirty_ctx = 0;

	/* get a context pointer based on if we are counting for a thread or for a cpu */
	if (ESI->state & PAPI_CPU_ATTACHED) {
		/* use cpu context */
		ctx = ESI->CpuInfo->context[ESI->CmpIdx];

		/* if the user wants to know if the control state was last set by the same event set, tell him */
		if (is_dirty != NULL) {
			if (ESI->CpuInfo->from_esi != ESI) {
				dirty_ctx = 1;
			}
			*is_dirty = dirty_ctx;
		}
		ESI->CpuInfo->from_esi = ESI;

	} else {

		/* use thread context */
		ctx = ESI->master->context[ESI->CmpIdx];

		/* if the user wants to know if the control state was last set by the same event set, tell him */
		if (is_dirty != NULL) {
			if (ESI->master->from_esi != ESI) {
				dirty_ctx = 1;
			}
			*is_dirty = dirty_ctx;
		}
		ESI->master->from_esi = ESI;

	}
	return( ctx );
}

static int
get_component_index(const char *name)
{
    int cidx;

    for (cidx = 0; cidx < papi_num_components; ++cidx) {
        if (strcmp(_papi_hwd[cidx]->cmp_info.name, name) == 0) {
            break;
        }
    }

    return cidx;
}

int
_papi_hwi_enum_dev_type(int enum_modifier, void **handle)
{
    _papi_hwi_sysdetect_t args;
    args.query_type = PAPI_SYSDETECT_QUERY__DEV_TYPE_ENUM;
    args.query.enumerate.modifier = enum_modifier;

    int cidx = get_component_index("sysdetect");
    assert(cidx < papi_num_components);

    return _papi_hwd[cidx]->user(0, &args, handle);
}

int
_papi_hwi_get_dev_type_attr(void *handle, PAPI_dev_type_attr_e attr, void *value)
{
    _papi_hwi_sysdetect_t args;
    args.query_type = PAPI_SYSDETECT_QUERY__DEV_TYPE_ATTR;
    args.query.dev_type.handle = handle;
    args.query.dev_type.attr = attr;

    int cidx = get_component_index("sysdetect");
    assert(cidx < papi_num_components);

    return _papi_hwd[cidx]->user(0, &args, value);
}

int
_papi_hwi_get_dev_attr(void *handle, int id, PAPI_dev_attr_e attr, void *value)
{
    _papi_hwi_sysdetect_t args;
    args.query_type = PAPI_SYSDETECT_QUERY__DEV_ATTR;
    args.query.dev.handle = handle;
    args.query.dev.id = id;
    args.query.dev.attr = attr;

    int cidx = get_component_index("sysdetect");
    assert(cidx < papi_num_components);

    return _papi_hwd[cidx]->user(0, &args, value);
}
