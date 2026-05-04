#ifndef _LINUX_BGP_H
#define _LINUX_BGP_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/profil.h>
#include <assert.h>
#include <limits.h>
#include <signal.h>
#include <errno.h>
#include <sys/ucontext.h>
#include <spi/bgp_SPI.h>

#include <stdarg.h>
#include <ctype.h>

#define MAX_COUNTERS BGP_UPC_MAX_MONITORED_EVENTS
#define MAX_COUNTER_TERMS MAX_COUNTERS

#include "papi.h"
#include "papi_preset.h"
//#include "papi_defines.h"
#include "linux-bgp-native-events.h"

// Context structure not used...
typedef struct bgp_context
{
	int reserved;
} bgp_context_t;

// Control state structure...  Holds local copy of read counters...
typedef struct bgp_control_state
{
	long_long counters[BGP_UPC_MAX_MONITORED_EVENTS];
} bgp_control_state_t;

// Register allocation structure
typedef struct bgp_reg_alloc
{
	_papi_hwd_bgp_native_event_id_t id;
} bgp_reg_alloc_t;

// Register structure not used...
typedef struct bgp_register
{
	int reserved;
} bgp_register_t;

/* Override void* definitions from PAPI framework layer */
/* with typedefs to conform to PAPI component layer code. */
#undef  hwd_reg_alloc_t
#undef  hwd_register_t
#undef  hwd_control_state_t
#undef  hwd_context_t

typedef bgp_reg_alloc_t hwd_reg_alloc_t;
typedef bgp_register_t hwd_register_t;
typedef bgp_control_state_t hwd_control_state_t;
typedef bgp_context_t hwd_context_t;

extern void _papi_hwd_lock( int );
extern void _papi_hwd_unlock( int );

#include "linux-bgp-context.h"

extern hwi_search_t *preset_search_map;

#endif
