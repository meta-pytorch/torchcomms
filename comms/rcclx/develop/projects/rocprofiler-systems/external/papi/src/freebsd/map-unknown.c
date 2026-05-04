/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/* 
* File:    map-unknown.c
* Author:  Harald Servat
*          redcrash@gmail.com
*/

#include "freebsd.h"
#include "papiStdEventDefs.h"
#include "map.h"

/****************************************************************************
 UNKNOWN SUBSTRATE
 UNKNOWN SUBSTRATE
 UNKNOWN SUBSTRATE
 UNKNOWN SUBSTRATE
****************************************************************************/

/*
	NativeEvent_Value_UnknownProcessor must match UnkProcessor_info 
*/

Native_Event_LabelDescription_t UnkProcessor_info[] =
{
	{ "branches", "Measure the number of branches retired." },
	{ "branch-mispredicts", "Measure the number of retired branches that were mispredicted." },
	/* { "cycles", "Measure processor cycles." }, */
	{ "dc-misses", "Measure the number of data cache misses." },
	{ "ic-misses", "Measure the number of instruction cache misses." },
	{ "instructions", "Measure the number of instructions retired." },
	{ "interrupts", "Measure the number of interrupts seen." },
	{ "unhalted-cycles", "Measure the number of cycles the processor is not in a halted or sleep state." },
	{ NULL, NULL }
};
