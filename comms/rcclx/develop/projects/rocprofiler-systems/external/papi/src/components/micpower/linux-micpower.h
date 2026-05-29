/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-micpower.h
 * @author  James Ralph
 *			ralph@eecs.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief Mic power component
 *  This file has the source code for a component that enables PAPI-C to access
 *  hardware monitoring sensors through a sysfs interface. This code 
 *  will dynamically create a native events table for all the sensors that can 
 *  be found under /sys/class/hwmon/hwmon[0-9]+.
 *
 * Notes: 
 *  - Based heavily upon the lm-sensors component by Heike Jagode.
 */

#ifndef _PAPI_MICPOWER_H_
#define _PAPI_MICPOWER_H_

#include <unistd.h>
#include <dirent.h>



/*************************  DEFINES SECTION  ***********************************
 *******************************************************************************/
/* this number assumes that there will never be more events than indicated */
#define MICPOWER_MAX_COUNTERS 16

/** Structure that stores private information of each event */
typedef struct {
	unsigned int selector;
} MICPOWER_register_t;

/*
 * The following structures mimic the ones used by other components. It is more
 * convenient to use them like that as programming with PAPI makes specific
 * assumptions for them.
 */



/** This structure is used to build the table of events */
typedef struct MICPOWER_native_event_entry
{
  char name[PAPI_MAX_STR_LEN];
  char units[PAPI_MIN_STR_LEN];
  char description[PAPI_MAX_STR_LEN];
  MICPOWER_register_t resources;
} MICPOWER_native_event_entry_t;

typedef struct MICPOWER_reg_alloc
{
	MICPOWER_register_t ra_bits;
} MICPOWER_reg_alloc_t;


typedef struct MICPOWER_control_state
{
	long long counts[MICPOWER_MAX_COUNTERS];	// used for caching
	long long lastupdate;
} MICPOWER_control_state_t;


typedef struct MICPOWER_context
{
	MICPOWER_control_state_t state;
} MICPOWER_context_t;



/*************************  GLOBALS SECTION  ***********************************
 *******************************************************************************/


#endif /* _PAPI_MICPOWER_H_ */
