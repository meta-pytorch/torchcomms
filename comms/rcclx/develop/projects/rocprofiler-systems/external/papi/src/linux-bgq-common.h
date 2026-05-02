/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-bgq-common.h
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

#include "papi.h"
/* Header required by BGPM */
#include "bgpm/include/bgpm.h"

extern int _papi_hwi_publish_error( char *error );

// Define gymnastics to create a compile time AT string.
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define _AT_ __FILE__ ":" TOSTRING(__LINE__)

/* return EXIT_FAILURE;  \*/

#define MAX_COUNTERS ( PEVT_LAST_EVENT + 1 )
//#define DEBUG_BGQ


/*************************  COMMON PROTOTYPES  *********************************
 *******************************************************************************/

/* common prototypes for BGQ sustrate and BGPM components */
int         _check_BGPM_error( int err, char* bgpmfunc );
long_long	_common_getEventValue( unsigned event_id, int EventGroup );
int 		_common_deleteRecreate( int *EventGroup_ptr );
int 		_common_rebuildEventgroup( int count, int *EventGroup_local, int *EventGroup_ptr );
int 		_common_set_overflow_BGPM( int EventGroup, 
									   int evt_idx,
									   int threshold, 
									   void (*handler)(int, uint64_t, uint64_t, const ucontext_t *) );
