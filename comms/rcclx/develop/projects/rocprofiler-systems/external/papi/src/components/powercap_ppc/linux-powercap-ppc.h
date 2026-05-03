/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    linux-io.h
 * CVS:     $Id$
 *
 * @author  PAPI team UTK/ICL
 *          dgenet@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief Power capping component for PowerPC
 *  This file contains the source code for a component that enables
 *  PAPI to get and set power capping on PowerPC (Power9) architecture.
 */

#ifndef _POWERCAP_PPC_H
#define _POWERCAP_PPC_H

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#define papi_powercap_ppc_lock() _papi_hwi_lock(COMPONENT_LOCK);
#define papi_powercap_ppc_unlock() _papi_hwi_unlock(COMPONENT_LOCK);

typedef struct _powercap_ppc_register {
    unsigned int selector;
} _powercap_ppc_register_t;

typedef struct _powercap_ppc_native_event_entry {
  char name[PAPI_MAX_STR_LEN];
  char units[PAPI_MIN_STR_LEN];
  char description[PAPI_MAX_STR_LEN];
  int socket_id;
  int component_id;
  int event_id;
  int type;
  int return_type;
  _powercap_ppc_register_t resources;
} _powercap_ppc_native_event_entry_t;

typedef struct _powercap_ppc_reg_alloc {
    _powercap_ppc_register_t ra_bits;
} _powercap_ppc_reg_alloc_t;

// package events
// powercap-current  powercap-max  powercap-min
#define PKG_MIN_POWER               0
#define PKG_MAX_POWER               1
#define PKG_CUR_POWER               2

#define PKG_NUM_EVENTS              3
#define POWERCAP_MAX_COUNTERS (PKG_NUM_EVENTS)

typedef struct _powercap_ppc_control_state {
  long long count[POWERCAP_MAX_COUNTERS];
  long long which_counter[POWERCAP_MAX_COUNTERS];
  long long lastupdate;
  int active_counters;
} _powercap_ppc_control_state_t;

typedef struct _powercap_ppc_context {
  long long start_value[POWERCAP_MAX_COUNTERS];
  _powercap_ppc_control_state_t state;
} _powercap_ppc_context_t;

#endif /* _POWERCAP_PPC_H */
