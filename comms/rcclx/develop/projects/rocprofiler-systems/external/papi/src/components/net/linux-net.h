/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    linux-net.h
 * CVS:     $Id$
 *
 * @author  Haihang You
 *          you@cs.utk.edu
 *
 * @author  Jose Pedro Oliveira
 *          jpo@di.uminho.pt
 *
 * @ingroup papi_components
 *
 * @brief net component
 *  This file contains the source code for a component that enables
 *  PAPI-C to access network statistics through the /proc file system.
 *  This component will dynamically create a native events table for
 *  all the interfaces listed in /proc/net/dev (16 entries for each
 *  interface).
 */

#ifndef _PAPI_NET_H
#define _PAPI_NET_H

#include <unistd.h>

/*************************  DEFINES SECTION  ***********************************
 *******************************************************************************/
/* this number assumes that there will never be more events than indicated
 * 20 INTERFACES * 16 COUNTERS = 320 */
#define NET_MAX_COUNTERS 320

/** Structure that stores private information of each event */
typedef struct NET_register
{
    /* This is used by the framework.It likes it to be !=0 to do somehting */
    unsigned int selector;
} NET_register_t;


/*
 * The following structures mimic the ones used by other components. It is more
 * convenient to use them like that as programming with PAPI makes specific
 * assumptions for them.
 */


/** This structure is used to build the table of events */
typedef struct NET_native_event_entry
{
    NET_register_t resources;
    char name[PAPI_MAX_STR_LEN];
    char description[PAPI_MAX_STR_LEN];
} NET_native_event_entry_t;


typedef struct NET_reg_alloc
{
    NET_register_t ra_bits;
} NET_reg_alloc_t;


typedef struct NET_control_state
{
    long long values[NET_MAX_COUNTERS]; // used for caching
    long long lastupdate;
} NET_control_state_t;


typedef struct NET_context
{
    NET_control_state_t state;
} NET_context_t;


/*************************  GLOBALS SECTION  ***********************************
 *******************************************************************************/

#endif /* _PAPI_NET_H */

/* vim:set ts=4 sw=4 sts=4 et: */
