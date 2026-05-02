/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
 * @file    appio.h
 * CVS:     $Id: appio.h,v 1.1.2.4 2012/02/01 05:01:00 tmohan Exp $
 *
 * @author  Philip Mucci
 *          phil.mucci@samaratechnologygroup.com
 *
 * @author  Tushar Mohan
 *          tushar.mohan@samaratechnologygroup.com
 *
 * @ingroup papi_components
 *
 * @brief appio component
 *  This file contains the source code for a component that enables
 *  PAPI to access application level file and socket I/O information.
 *  It does this through function replacement in the first person and
 *  by trapping syscalls in the third person.
 */

#ifndef _PAPI_APPIO_H
#define _PAPI_APPIO_H

#include <unistd.h>

/*************************  DEFINES SECTION  ***********************************/

/* Set this equal to the number of elements in _appio_counter_info array */
#define APPIO_MAX_COUNTERS 45

/** Structure that stores private information of each event */
typedef struct APPIO_register
{
    /* This is used by the framework. It likes it to be !=0 to do something */
    unsigned int selector;
} APPIO_register_t;


/*
 * The following structures mimic the ones used by other components. It is more
 * convenient to use them like that as programming with PAPI makes specific
 * assumptions for them.
 */


/* This structure is used to build the table of events */

typedef struct APPIO_native_event_entry
{
    APPIO_register_t resources;
    const char* name;
    const char* description;
} APPIO_native_event_entry_t;


typedef struct APPIO_reg_alloc
{
    APPIO_register_t ra_bits;
} APPIO_reg_alloc_t;


typedef struct APPIO_control_state
{
    int num_events;
    int counter_bits[APPIO_MAX_COUNTERS];
    long long values[APPIO_MAX_COUNTERS]; // used for caching
} APPIO_control_state_t;


typedef struct APPIO_context
{
    APPIO_control_state_t state;
} APPIO_context_t;


/*************************  GLOBALS SECTION  ***********************************
 *******************************************************************************/

#endif /* _PAPI_APPIO_H */

/* vim:set ts=4 sw=4 sts=4 et: */
