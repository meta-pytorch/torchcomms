/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
 * @file    linux-coretemp.h
 * CVS:     $Id$
 * @author  James Ralph
 *			ralph@eecs.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief coretemp component
 *  This file has the source code for a component that enables PAPI-C to access
 *  hardware monitoring sensors through the coretemp sysfs interface. This code 
 *  will dynamically create a native events table for all the sensors that can 
 *  be found under /sys/class/hwmon/hwmon[0-9]+.
 *
 * Notes: 
 *  - Based heavily upon the lm-sensors component by Heike Jagode.
 */

#ifndef _PAPI_CORETEMP_H
#define _PAPI_CORETEMP_H

#include <unistd.h>
#include <dirent.h>



/*************************  DEFINES SECTION  ***********************************
 *******************************************************************************/
/* this number assumes that there will never be more events than indicated */
#define CORETEMP_MAX_COUNTERS 512

/** Structure that stores private information of each event */
typedef struct CORETEMP_register
{
	/* This is used by the framework.It likes it to be !=0 to do somehting */
	unsigned int selector;
	/* These are the only information needed to locate a libsensors event */
	int subfeat_nr;
} CORETEMP_register_t;

/*
 * The following structures mimic the ones used by other components. It is more
 * convenient to use them like that as programming with PAPI makes specific
 * assumptions for them.
 */



/** This structure is used to build the table of events */
typedef struct CORETEMP_native_event_entry
{
  char name[PAPI_MAX_STR_LEN];
  char units[PAPI_MIN_STR_LEN];
  char description[PAPI_MAX_STR_LEN];
  char path[PATH_MAX];
  int stone; /* some counters are set in stone, a max temperature is just that... */
  long value;
  CORETEMP_register_t resources;
} CORETEMP_native_event_entry_t;

typedef struct CORETEMP_reg_alloc
{
	CORETEMP_register_t ra_bits;
} CORETEMP_reg_alloc_t;


typedef struct CORETEMP_control_state
{
	long long counts[CORETEMP_MAX_COUNTERS];	// used for caching
	long long lastupdate;
} CORETEMP_control_state_t;


typedef struct CORETEMP_context
{
	CORETEMP_control_state_t state;
} CORETEMP_context_t;



/*************************  GLOBALS SECTION  ***********************************
 *******************************************************************************/


#endif /* _PAPI_CORETEMP_H */
