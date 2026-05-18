#ifndef _PAPI_DEFINES_H
#define _PAPI_DEFINES_H

/* Thread related: locks */

#define INTERNAL_LOCK           PAPI_NUM_LOCK+0 /* papi_internal.c */
#define MULTIPLEX_LOCK          PAPI_NUM_LOCK+1 /* multiplex.c */
#define THREADS_LOCK            PAPI_NUM_LOCK+2 /* threads.c */
#define HIGHLEVEL_LOCK          PAPI_NUM_LOCK+3 /* papi_hl.c */
#define MEMORY_LOCK             PAPI_NUM_LOCK+4 /* papi_memory.c */
#define COMPONENT_LOCK          PAPI_NUM_LOCK+5 /* per-component */
#define GLOBAL_LOCK             PAPI_NUM_LOCK+6 /* papi.c for global variable (static and non) initialization/shutdown */
#define CPUS_LOCK               PAPI_NUM_LOCK+7 /* cpus.c */
#define NAMELIB_LOCK            PAPI_NUM_LOCK+8 /* papi_pfm4_events.c */


#define NUM_INNER_LOCK  9
#define PAPI_MAX_LOCK   (NUM_INNER_LOCK + PAPI_NUM_LOCK + PAPI_NUM_COMP)

#include OSLOCK


#endif
