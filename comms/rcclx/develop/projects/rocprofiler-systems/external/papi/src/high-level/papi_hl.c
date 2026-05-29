/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/**
* @file     papi_hl.c
* @author   Frank Winkler
*           frank.winkler@icl.utk.edu
* @author   Philip Mucci
*           mucci@cs.utk.edu
* @brief This file contains the 'high level' interface to PAPI.
*  BASIC is a high level language. ;-) */

#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <pthread.h>
#include <search.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <stdint.h>
#include <unistd.h>
#include "papi.h"
#include "papi_internal.h"


/* For dynamic linking to libpapi */
/* Weak symbol for pthread_once to avoid additional linking
 * against libpthread when not used. */
#pragma weak pthread_once

#define verbose_fprintf \
   if (verbosity == 1) fprintf

/* defaults for number of components and events */
#define PAPIHL_NUM_OF_COMPONENTS 10
#define PAPIHL_NUM_OF_EVENTS_PER_COMPONENT 10

#define PAPIHL_ACTIVE 1
#define PAPIHL_DEACTIVATED 0

/* number of nested regions */
#define PAPIHL_MAX_STACK_SIZE 10

/* global components data begin *****************************************/
typedef struct components
{
   int component_id;
   int num_of_events;
   int max_num_of_events;
   char **event_names;
   int *event_codes;
   short *event_types;
   int EventSet; //only for testing at initialization phase
} components_t;

components_t *components = NULL;
int num_of_components = 0;
int max_num_of_components = PAPIHL_NUM_OF_COMPONENTS;
int total_num_events = 0;
int num_of_cleaned_threads = 0;

/* global components data end *******************************************/


/* thread local components data begin ***********************************/
typedef struct local_components
{
   int EventSet;
   /** Return values for the eventsets */
   long_long *values;
} local_components_t;

THREAD_LOCAL_STORAGE_KEYWORD local_components_t *_local_components = NULL;
THREAD_LOCAL_STORAGE_KEYWORD long_long _local_cycles;
THREAD_LOCAL_STORAGE_KEYWORD volatile bool _local_state = PAPIHL_ACTIVE;
THREAD_LOCAL_STORAGE_KEYWORD unsigned int _local_region_begin_cnt = 0; /**< Count each PAPI_hl_region_begin call */
THREAD_LOCAL_STORAGE_KEYWORD unsigned int _local_region_end_cnt = 0;   /**< Count each PAPI_hl_region_end call */

THREAD_LOCAL_STORAGE_KEYWORD unsigned int _local_region_id_stack[PAPIHL_MAX_STACK_SIZE];
THREAD_LOCAL_STORAGE_KEYWORD int _local_region_id_top = -1;


/* thread local components data end *************************************/


/* global event storage data begin **************************************/
typedef struct reads
{
   struct reads *next;
   struct reads *prev;
   long_long value;        /**< Event value */
} reads_t;

typedef struct
{
   long_long begin;        /**< Event value for region_begin */
   long_long region_value; /**< Delta value for region_end - region_begin */
   reads_t *read_values;   /**< List of read event values inside a region */
} value_t;

typedef struct regions
{
   unsigned int region_id; /**< Unique region ID */
   int parent_region_id;   /**< Region ID of parent region */
   char *region;           /**< Region name */
   struct regions *next;
   struct regions *prev;
   value_t values[];       /**< Array of event values based on current eventset */
} regions_t;

typedef struct
{
   unsigned long key;      /**< Thread ID */
   regions_t *value;       /**< List of regions */
} threads_t;

int compar(const void *l, const void *r)
{
   const threads_t *lm = l;
   const threads_t *lr = r;
   return lm->key - lr->key;
}

typedef struct
{
   void *root;             /**< Root of binary tree */
   threads_t *find_p;      /**< Pointer that is used for finding a thread node */ 
} binary_tree_t;

/**< Global binary tree that stores events from all threads */
binary_tree_t* binary_tree = NULL;

/* global event storage data end ****************************************/


/* global auxiliary variables begin *************************************/
enum region_type { REGION_BEGIN, REGION_READ, REGION_END };

char **requested_event_names = NULL; /**< Events from user or default */
int num_of_requested_events = 0;

bool hl_initiated = false;       /**< Check PAPI-HL has been initiated */
bool hl_finalized = false;       /**< Check PAPI-HL has been fininalized */
bool events_determined = false;  /**< Check if events are determined */
bool output_generated = false;   /**< Check if output has been already generated */
static char *absolute_output_file_path = NULL;
static int output_counter = 0;   /**< Count each output generation. Not used yet */
short verbosity = 0;             /**< Verbose output is off by default */
bool state = PAPIHL_ACTIVE;      /**< PAPIHL is active until first error or finalization */
static int region_begin_cnt = 0; /**< Count each PAPI_hl_region_begin call */
static int region_end_cnt = 0;   /**< Count each PAPI_hl_region_end call */
unsigned long master_thread_id = -1; /**< Remember id of master thread */

/* global auxiliary variables end ***************************************/

static void _internal_hl_library_init(void);
static void _internal_hl_onetime_library_init(void);

/* functions for creating eventsets for different components */
static int _internal_hl_checkCounter ( char* counter );
static int _internal_hl_determine_rank();
static char *_internal_hl_remove_spaces( char *str, int mode );
static int _internal_hl_determine_default_events();
static int _internal_hl_read_user_events();
static int _internal_hl_new_component(int component_id, components_t *component);
static int _internal_hl_add_event_to_component(char *event_name, int event,
                                        short event_type, components_t *component);
static int _internal_hl_create_components();
static int _internal_hl_read_events(const char* events);
static int _internal_hl_create_event_sets();
static int _internal_hl_start_counters();

/* functions for storing events */
static int _internal_hl_region_id_pop();
static int _internal_hl_region_id_push();
static int _internal_hl_region_id_stack_peak();

static inline reads_t* _internal_hl_insert_read_node( reads_t** head_node );
static inline int _internal_hl_add_values_to_region( regions_t *node, enum region_type reg_typ );
static inline regions_t* _internal_hl_insert_region_node( regions_t** head_node, const char *region );
static inline regions_t* _internal_hl_find_region_node( regions_t* head_node, const char *region );
static inline threads_t* _internal_hl_insert_thread_node( unsigned long tid );
static inline threads_t* _internal_hl_find_thread_node( unsigned long tid );
static int _internal_hl_store_counters( unsigned long tid, const char *region,
                                        enum region_type reg_typ );
static int _internal_hl_read_counters();
static int _internal_hl_read_and_store_counters( const char *region, enum region_type reg_typ );
static int _internal_hl_create_global_binary_tree();

/* functions for output generation */
static int _internal_hl_mkdir(const char *dir);
static int _internal_hl_determine_output_path();
static void _internal_hl_json_line_break_and_indent(FILE* f, bool b, int width);
static void _internal_hl_json_definitions(FILE* f, bool beautifier);
static void _internal_hl_json_region_events(FILE* f, bool beautifier, regions_t *regions);
static void _internal_hl_json_regions(FILE* f, bool beautifier, threads_t* thread_node);
static void _internal_hl_json_threads(FILE* f, bool beautifier, unsigned long* tids, int threads_num);
static int _internal_hl_cmpfunc(const void * a, const void * b);
static int _internal_get_sorted_thread_list(unsigned long** tids, int* threads_num);
static void _internal_hl_write_json_file(FILE* f, unsigned long* tids, int threads_num);
static void _internal_hl_read_json_file(const char* path);
static void _internal_hl_write_output();

/* functions for cleaning up heap memory */
static void _internal_hl_clean_up_local_data();
static void _internal_hl_clean_up_global_data();
static void _internal_hl_clean_up_all(bool deactivate);
static int _internal_hl_check_for_clean_thread_states();

/* internal advanced functions */
int _internal_PAPI_hl_init(); /**< intialize high level library */
int _internal_PAPI_hl_cleanup_thread(); /**< clean local-thread event sets */
int _internal_PAPI_hl_finalize(); /**< shutdown event sets and clear up everything */
int _internal_PAPI_hl_set_events(const char* events); /**< set specfic events to be recorded */
void _internal_PAPI_hl_print_output(); /**< generate output */


static void _internal_hl_library_init(void)
{
   /* This function is only called by one thread! */
   int retval;

   /* check VERBOSE level */
   if ( getenv("PAPI_HL_VERBOSE") != NULL ) {
      verbosity = 1;
   }

   if ( ( retval = PAPI_library_init(PAPI_VER_CURRENT) ) != PAPI_VER_CURRENT )
      verbose_fprintf(stdout, "PAPI-HL Error: PAPI_library_init failed!\n");
   
   /* PAPI_thread_init only suceeds if PAPI_library_init has suceeded */
   char *multi_thread = getenv("PAPI_HL_THREAD_MULTIPLE");
   if ( NULL == multi_thread || atoi(multi_thread) == 1 ) {
      retval = PAPI_thread_init(_papi_gettid);
   } else {
      retval = PAPI_thread_init(_papi_getpid);
   }

   if (retval == PAPI_OK) {

      /* determine output directory and output file */
      if ( ( retval = _internal_hl_determine_output_path() ) != PAPI_OK ) {
         verbose_fprintf(stdout, "PAPI-HL Error: _internal_hl_determine_output_path failed!\n");
         state = PAPIHL_DEACTIVATED;
         verbose_fprintf(stdout, "PAPI-HL Error: PAPI could not be initiated!\n");
      } else {

         /* register the termination function for output */
         atexit(_internal_PAPI_hl_print_output);
         verbose_fprintf(stdout, "PAPI-HL Info: PAPI has been initiated!\n");

         /* remember thread id */
         master_thread_id = PAPI_thread_id();
         HLDBG("master_thread_id=%lu\n", master_thread_id);
      }

      /* Support multiplexing if user wants to */
      if ( getenv("PAPI_MULTIPLEX") != NULL ) {
         retval = PAPI_multiplex_init();
         if ( retval == PAPI_ENOSUPP) {
            verbose_fprintf(stdout, "PAPI-HL Info: Multiplex is not supported!\n");
         } else if ( retval != PAPI_OK ) {
            verbose_fprintf(stdout, "PAPI-HL Error: PAPI_multiplex_init failed!\n");
         } else if ( retval == PAPI_OK ) {
            verbose_fprintf(stdout, "PAPI-HL Info: Multiplex has been initiated!\n");
         }
      }

   } else {
      verbose_fprintf(stdout, "PAPI-HL Error: PAPI_thread_init failed!\n");
      state = PAPIHL_DEACTIVATED;
      verbose_fprintf(stdout, "PAPI-HL Error: PAPI could not be initiated!\n");
   }

   hl_initiated = true;
}

static void _internal_hl_onetime_library_init(void)
{
   static pthread_once_t library_is_initialized = PTHREAD_ONCE_INIT;
   if ( pthread_once ) {
      /* we assume that PAPI_hl_init() is called from a parallel region */
      pthread_once(&library_is_initialized, _internal_hl_library_init);
      /* wait until first thread has finished */
      int i = 0;
      /* give it 5 seconds in case PAPI_thread_init crashes */
      while ( !hl_initiated && (i++) < 500000 )
         usleep(10);
   } else {
      /* we assume that PAPI_hl_init() is called from a serial application
       * that was not linked against libpthread */
      _internal_hl_library_init();
   }
}

static int
_internal_hl_checkCounter ( char* counter )
{
   int EventSet = PAPI_NULL;
   int eventcode;
   int retval;

   HLDBG("Counter: %s\n", counter);
   if ( ( retval = PAPI_create_eventset( &EventSet ) ) != PAPI_OK )
      return ( retval );

   if ( ( retval = PAPI_event_name_to_code( counter, &eventcode ) ) != PAPI_OK ) {
      HLDBG("Counter %s does not exist\n", counter);
      return ( retval );
   }

   if ( ( retval = PAPI_add_event (EventSet, eventcode) ) != PAPI_OK ) {
      HLDBG("Cannot add counter %s\n", counter);
      return ( retval );
   }

   if ( ( retval = PAPI_cleanup_eventset (EventSet) ) != PAPI_OK )
      return ( retval );

   if ( ( retval = PAPI_destroy_eventset (&EventSet) ) != PAPI_OK )
      return ( retval );

   return ( PAPI_OK );
}

static int _internal_hl_determine_rank()
{
   int rank = -1;
   /* check environment variables for rank identification */

   if ( getenv("OMPI_COMM_WORLD_RANK") != NULL )
      rank = atoi(getenv("OMPI_COMM_WORLD_RANK"));
   else if ( getenv("ALPS_APP_PE") != NULL )
      rank = atoi(getenv("ALPS_APP_PE"));
   else if ( getenv("PMI_RANK") != NULL )
      rank = atoi(getenv("PMI_RANK"));
   else if ( getenv("SLURM_PROCID") != NULL )
      rank = atoi(getenv("SLURM_PROCID"));

   return rank;
}

static char *_internal_hl_remove_spaces( char *str, int mode )
{
   char *out = str, *put = str;
   for(; *str != '\0'; ++str) {
      if ( mode == 0 ) {
         if(*str != ' ')
            *put++ = *str;
      } else {
         while (*str == ' ' && *(str + 1) == ' ')
            str++;
         *put++ = *str;
      }
   }
   *put = '\0';
   return out;
}

static int _internal_hl_determine_default_events()
{
   int i;
   HLDBG("Default events\n");
   char *default_events[] = {
      "PAPI_TOT_CYC",
   };
   int num_of_defaults = sizeof(default_events) / sizeof(char*);

   /* allocate memory for requested events */
   requested_event_names = (char**)malloc(num_of_defaults * sizeof(char*));
   if ( requested_event_names == NULL )
      return ( PAPI_ENOMEM );

   /* check if default events are available on the current machine */
   for ( i = 0; i < num_of_defaults; i++ ) {
      if ( _internal_hl_checkCounter( default_events[i] ) == PAPI_OK ) {
         requested_event_names[num_of_requested_events++] = strdup(default_events[i]);
         if ( requested_event_names[num_of_requested_events -1] == NULL )
            return ( PAPI_ENOMEM );
      } 
      else {
         /* if PAPI_FP_OPS is not available try PAPI_SP_OPS or PAPI_DP_OPS */
         if ( strcmp(default_events[i], "PAPI_FP_OPS") == 0 ) {
            if ( _internal_hl_checkCounter( "PAPI_SP_OPS" ) == PAPI_OK )
               requested_event_names[num_of_requested_events++] = strdup("PAPI_SP_OPS");
            else if ( _internal_hl_checkCounter( "PAPI_DP_OPS" ) == PAPI_OK )
               requested_event_names[num_of_requested_events++] = strdup("PAPI_DP_OPS");
         }

         /* if PAPI_FP_INS is not available try PAPI_VEC_SP or PAPI_VEC_DP */
         if ( strcmp(default_events[i], "PAPI_FP_INS") == 0 ) {
            if ( _internal_hl_checkCounter( "PAPI_VEC_SP" ) == PAPI_OK )
               requested_event_names[num_of_requested_events++] = strdup("PAPI_VEC_SP");
            else if ( _internal_hl_checkCounter( "PAPI_VEC_DP" ) == PAPI_OK )
               requested_event_names[num_of_requested_events++] = strdup("PAPI_VEC_DP");
         }
      }
   }

   return ( PAPI_OK );
}

static int _internal_hl_read_user_events(const char *user_events)
{
   char* user_events_copy;
   const char *separator; //separator for events
   int num_of_req_events = 1; //number of events in string
   int req_event_index = 0; //index of event
   const char *position = NULL; //current position in processed string
   char *token;
   
   HLDBG("User events: %s\n", user_events);
   user_events_copy = strdup(user_events);
   if ( user_events_copy == NULL )
      return ( PAPI_ENOMEM );

   /* check if string is not empty */
   if ( strlen( user_events_copy ) > 0 )
   {
      /* count number of separator characters */
      position = user_events_copy;
      separator=",";
      while ( *position ) {
         if ( strchr( separator, *position ) ) {
            num_of_req_events++;
         }
            position++;
      }

      /* allocate memory for requested events */
      requested_event_names = (char**)malloc(num_of_req_events * sizeof(char*));
      if ( requested_event_names == NULL ) {
         free(user_events_copy);
         return ( PAPI_ENOMEM );
      }

      /* parse list of event names */
      token = strtok( user_events_copy, separator );
      while ( token ) {
         if ( req_event_index >= num_of_req_events ){
            /* more entries as in the first run */
            free(user_events_copy);
            return PAPI_EINVAL;
         }
         requested_event_names[req_event_index] = strdup(_internal_hl_remove_spaces(token, 0));
         if ( requested_event_names[req_event_index] == NULL ) {
            free(user_events_copy);
            return ( PAPI_ENOMEM );
         }
         token = strtok( NULL, separator );
         req_event_index++;
      }
   }

   num_of_requested_events = req_event_index;
   free(user_events_copy);
   if ( num_of_requested_events == 0 )
      return PAPI_EINVAL;

   HLDBG("Number of requested events: %d\n", num_of_requested_events);
   return ( PAPI_OK );
}

static int _internal_hl_new_component(int component_id, components_t *component)
{
   int retval;

   /* create new EventSet */
   component->EventSet = PAPI_NULL;
   if ( ( retval = PAPI_create_eventset( &component->EventSet ) ) != PAPI_OK ) {
      verbose_fprintf(stdout, "PAPI-HL Error: Cannot create EventSet for component %d.\n", component_id);
      return ( retval );
   }

   /* Support multiplexing if user wants to */
   if ( getenv("PAPI_MULTIPLEX") != NULL ) {

      /* multiplex only for cpu core events */
      if ( component_id == 0 ) {
         retval = PAPI_assign_eventset_component(component->EventSet, component_id);
         if ( retval != PAPI_OK ) {
            verbose_fprintf(stdout, "PAPI-HL Error: PAPI_assign_eventset_component failed.\n");
         } else {
            if ( PAPI_get_multiplex(component->EventSet) == false ) {
               retval = PAPI_set_multiplex(component->EventSet);
               if ( retval != PAPI_OK ) {
                  verbose_fprintf(stdout, "PAPI-HL Error: PAPI_set_multiplex failed.\n");
               }
            }
         }
      }
   }

   component->component_id = component_id;
   component->num_of_events = 0;
   component->max_num_of_events = PAPIHL_NUM_OF_EVENTS_PER_COMPONENT;

   component->event_names = NULL;
   component->event_names = (char**)malloc(component->max_num_of_events * sizeof(char*));
   if ( component->event_names == NULL )
      return ( PAPI_ENOMEM );

   component->event_codes = NULL;
   component->event_codes = (int*)malloc(component->max_num_of_events * sizeof(int));
   if ( component->event_codes == NULL )
      return ( PAPI_ENOMEM );

   component->event_types = NULL;
   component->event_types = (short*)malloc(component->max_num_of_events * sizeof(short));
   if ( component->event_types == NULL )
      return ( PAPI_ENOMEM );

   num_of_components += 1;
   return ( PAPI_OK );
}

static int _internal_hl_add_event_to_component(char *event_name, int event,
                                        short event_type, components_t *component)
{
   int i, retval;

   /* check if we need to reallocate memory for event_names, event_codes and event_types */
   if ( component->num_of_events == component->max_num_of_events ) {
      component->max_num_of_events *= 2;

      component->event_names = (char**)realloc(component->event_names, component->max_num_of_events * sizeof(char*));
      if ( component->event_names == NULL )
         return ( PAPI_ENOMEM );

      component->event_codes = (int*)realloc(component->event_codes, component->max_num_of_events * sizeof(int));
      if ( component->event_codes == NULL )
         return ( PAPI_ENOMEM );

      component->event_types = (short*)realloc(component->event_types, component->max_num_of_events * sizeof(short));
      if ( component->event_types == NULL )
         return ( PAPI_ENOMEM );
   }

   retval = PAPI_add_event( component->EventSet, event );
   if ( retval != PAPI_OK ) {
      const PAPI_component_info_t* cmpinfo;
      cmpinfo = PAPI_get_component_info( component->component_id );
      verbose_fprintf(stdout, "PAPI-HL Warning: Cannot add %s to component %s.\n", event_name, cmpinfo->name);
      verbose_fprintf(stdout, "The following event combination is not supported:\n");
      for ( i = 0; i < component->num_of_events; i++ )
         verbose_fprintf(stdout, "  %s\n", component->event_names[i]);
      verbose_fprintf(stdout, "  %s\n", event_name);
      verbose_fprintf(stdout, "Advice: Use papi_event_chooser to obtain an appropriate event set for this component or set PAPI_MULTIPLEX=1.\n");

      return PAPI_EINVAL;
   }

   component->event_names[component->num_of_events] = event_name;
   component->event_codes[component->num_of_events] = event;
   component->event_types[component->num_of_events] = event_type;
   component->num_of_events += 1;

   total_num_events += 1;

   return PAPI_OK;
}

static int _internal_hl_create_components()
{
   int i, j, retval, event;
   int component_id = -1;
   int comp_index = 0;
   bool component_exists = false;
   short event_type = 0;

   HLDBG("Create components\n");
   components = (components_t*)malloc(max_num_of_components * sizeof(components_t));
   if ( components == NULL )
      return ( PAPI_ENOMEM );

   for ( i = 0; i < num_of_requested_events; i++ ) {
      /* check if requested event contains event type (instant or delta) */
      const char sep = '=';
      char *ret;
      int index;
      /* search for '=' in event name */
      ret = strchr(requested_event_names[i], sep);
      if (ret) {
         if ( strcmp(ret, "=instant") == 0 )
            event_type = 1;
         else
            event_type = 0;

         /* get index of '=' in event name */
         index = (int)(ret - requested_event_names[i]);
         /* remove event type from string if '=instant' or '=delta' */
         if ( (strcmp(ret, "=instant") == 0) || (strcmp(ret, "=delta") == 0) )
            requested_event_names[i][index] = '\0';
      }

      /* change event type to instantaneous for specific events */
      /* we consider all nvml events as instantaneous values */
      if( (strstr(requested_event_names[i], "nvml:::") != NULL) ) {
         event_type = 1;
         verbose_fprintf(stdout, "PAPI-HL Info: The event \"%s\" will be stored as instantaneous value.\n", requested_event_names[i]);
      }

      /* check if event is supported on current machine */
      retval = _internal_hl_checkCounter(requested_event_names[i]);
      if ( retval != PAPI_OK ) {
         verbose_fprintf(stdout, "PAPI-HL Warning: \"%s\" does not exist or is not supported on this machine.\n", requested_event_names[i]);
      } else {
         /* determine event code and corresponding component id */
         retval = PAPI_event_name_to_code( requested_event_names[i], &event );
         if ( retval != PAPI_OK )
            return ( retval );
         component_id = PAPI_COMPONENT_INDEX( event );

         /* check if component_id already exists in global components structure */
         for ( j = 0; j < num_of_components; j++ ) {
            if ( components[j].component_id == component_id ) {
               component_exists = true;
               comp_index = j;
               break;
            }
            else {
               component_exists = false;
            }
         }

         /* create new component */
         if ( false == component_exists ) {
            /* check if we need to reallocate memory for components */
            if ( num_of_components == max_num_of_components ) {
               max_num_of_components *= 2;
               components = (components_t*)realloc(components, max_num_of_components * sizeof(components_t));
               if ( components == NULL )
                  return ( PAPI_ENOMEM );
            }
            comp_index = num_of_components;
            retval = _internal_hl_new_component(component_id, &components[comp_index]);
            if ( retval != PAPI_OK )
               return ( retval );
         }

         /* add event to current component */
         retval = _internal_hl_add_event_to_component(requested_event_names[i], event, event_type, &components[comp_index]);
         if ( retval == PAPI_ENOMEM )
            return ( retval );
      }
   }

   HLDBG("Number of components %d\n", num_of_components);
   if ( num_of_components > 0 )
      verbose_fprintf(stdout, "PAPI-HL Info: Using the following events:\n");

   /* destroy all EventSets from global data */
   for ( i = 0; i < num_of_components; i++ ) {
      if ( ( retval = PAPI_cleanup_eventset (components[i].EventSet) ) != PAPI_OK )
         return ( retval );
      if ( ( retval = PAPI_destroy_eventset (&components[i].EventSet) ) != PAPI_OK )
         return ( retval );
      components[i].EventSet = PAPI_NULL;

      HLDBG("component_id = %d\n", components[i].component_id);
      HLDBG("num_of_events = %d\n", components[i].num_of_events);
      for ( j = 0; j < components[i].num_of_events; j++ ) {
         HLDBG(" %s type=%d\n", components[i].event_names[j], components[i].event_types[j]);
         verbose_fprintf(stdout, "  %s\n", components[i].event_names[j]);
      }
   }

   if ( num_of_components == 0 )
      return PAPI_EINVAL;

   return PAPI_OK;
}

static int _internal_hl_read_events(const char* events)
{
   int i, retval;
   HLDBG("Read events: %s\n", events);
   if ( events != NULL ) {
      if ( _internal_hl_read_user_events(events) != PAPI_OK )
         if ( ( retval = _internal_hl_determine_default_events() ) != PAPI_OK )
            return ( retval );

   /* check if user specified events via environment variable */
   } else if ( getenv("PAPI_EVENTS") != NULL ) {
      char *user_events_from_env = strdup( getenv("PAPI_EVENTS") );
      if ( user_events_from_env == NULL )
         return ( PAPI_ENOMEM );
      /* if string is emtpy use default events */
      if ( strlen( user_events_from_env ) == 0 ) {
         if ( ( retval = _internal_hl_determine_default_events() ) != PAPI_OK ) {
            free(user_events_from_env);
            return ( retval );
         }
      }
      else if ( _internal_hl_read_user_events(user_events_from_env) != PAPI_OK )
         if ( ( retval = _internal_hl_determine_default_events() ) != PAPI_OK ) {
            free(user_events_from_env);
            return ( retval );
         }
      free(user_events_from_env);
   } else {
      if ( ( retval = _internal_hl_determine_default_events() ) != PAPI_OK )
            return ( retval );
   }

   /* create components based on requested events */
   if ( _internal_hl_create_components() != PAPI_OK )
   {
      /* requested events do not work at all, use default events */
      verbose_fprintf(stdout, "PAPI-HL Warning: All requested events do not work, using default.\n");

      for ( i = 0; i < num_of_requested_events; i++ )
         free(requested_event_names[i]);
      free(requested_event_names);
      num_of_requested_events = 0;
      if ( ( retval = _internal_hl_determine_default_events() ) != PAPI_OK )
         return ( retval );
      if ( ( retval = _internal_hl_create_components() ) != PAPI_OK )
         return ( retval );
   }

   events_determined = true;
   return ( PAPI_OK );
}

static int _internal_hl_create_event_sets()
{
   int i, j, retval;

   if ( state == PAPIHL_ACTIVE ) {
      /* allocate memory for local components */
      _local_components = (local_components_t*)malloc(num_of_components * sizeof(local_components_t));
      if ( _local_components == NULL )
         return ( PAPI_ENOMEM );

      for ( i = 0; i < num_of_components; i++ ) {
         /* create EventSet */
         _local_components[i].EventSet = PAPI_NULL;
         if ( ( retval = PAPI_create_eventset( &_local_components[i].EventSet ) ) != PAPI_OK ) {
            return (retval );
         }

         /* Support multiplexing if user wants to */
         if ( getenv("PAPI_MULTIPLEX") != NULL ) {

            /* multiplex only for cpu core events */
            if ( components[i].component_id == 0 ) {
               retval = PAPI_assign_eventset_component(_local_components[i].EventSet, components[i].component_id );
               if ( retval != PAPI_OK ) {
                  verbose_fprintf(stdout, "PAPI-HL Error: PAPI_assign_eventset_component failed.\n");
               } else {
                  if ( PAPI_get_multiplex(_local_components[i].EventSet) == false ) {
                     retval = PAPI_set_multiplex(_local_components[i].EventSet);
                     if ( retval != PAPI_OK ) {
                        verbose_fprintf(stdout, "PAPI-HL Error: PAPI_set_multiplex failed.\n");
                     }
                  }
               }
            }
         }

         /* add event to current EventSet */
         for ( j = 0; j < components[i].num_of_events; j++ ) {
            retval = PAPI_add_event( _local_components[i].EventSet, components[i].event_codes[j] );
            if ( retval != PAPI_OK ) {
               return (retval );
            }
         }
         /* allocate memory for return values */
         _local_components[i].values = (long_long*)malloc(components[i].num_of_events * sizeof(long_long));
         if ( _local_components[i].values == NULL )
            return ( PAPI_ENOMEM );

      }
      return PAPI_OK;
   }
   return ( PAPI_EMISC );
}

static int _internal_hl_start_counters()
{
   int i, retval;
   long_long cycles;

   if ( state == PAPIHL_ACTIVE ) {
      for ( i = 0; i < num_of_components; i++ ) {
         if ( ( retval = PAPI_start( _local_components[i].EventSet ) ) != PAPI_OK )
            return (retval );

         /* warm up PAPI code paths and data structures */
         if ( ( retval = PAPI_read_ts( _local_components[i].EventSet, _local_components[i].values, &cycles ) != PAPI_OK ) ) {
            return (retval );
         }
      }
      _papi_hl_events_running = 1;
      return PAPI_OK;
   }
   return ( PAPI_EMISC );
}

static int _internal_hl_region_id_pop() {
   if ( _local_region_id_top == -1 ) {
      return PAPI_ENOEVNT;
   } else {
      _local_region_id_top--;
   }
   return PAPI_OK;
}

static int _internal_hl_region_id_push() {
   if ( _local_region_id_top == PAPIHL_MAX_STACK_SIZE ) {
      return PAPI_ENOMEM;
   } else {
      _local_region_id_top++;
      _local_region_id_stack[_local_region_id_top] = _local_region_begin_cnt;
   }
   return PAPI_OK;
}

static int _internal_hl_region_id_stack_peak() {
   if ( _local_region_id_top == -1 ) {
      return -1;
   } else {
      return _local_region_id_stack[_local_region_id_top];
   }
}

static inline reads_t* _internal_hl_insert_read_node(reads_t** head_node)
{
   reads_t *new_node;

   /* create new region node */
   if ( ( new_node = malloc(sizeof(reads_t)) ) == NULL )
      return ( NULL );
   new_node->next = NULL;
   new_node->prev = NULL;

   /* insert node in list */
   if ( *head_node == NULL ) {
      *head_node = new_node;
      return new_node;
   }
   (*head_node)->prev = new_node;
   new_node->next = *head_node;
   *head_node = new_node;

   return new_node;
}

static inline int _internal_hl_add_values_to_region( regions_t *node, enum region_type reg_typ )
{
   int i, j;
   long_long ts;
   int cmp_iter = 2;

   /* get timestamp */
   ts = PAPI_get_real_nsec();

   if ( reg_typ == REGION_BEGIN ) {
      /* set first fixed counters */
      node->values[0].begin = _local_cycles;
      node->values[1].begin = ts;
      /* events from components */
      for ( i = 0; i < num_of_components; i++ )
         for ( j = 0; j < components[i].num_of_events; j++ )
            node->values[cmp_iter++].begin = _local_components[i].values[j];
   } else if ( reg_typ == REGION_READ ) {
      /* create a new read node and add values*/
      reads_t* read_node;
      if ( ( read_node = _internal_hl_insert_read_node(&node->values[0].read_values) ) == NULL )
         return ( PAPI_ENOMEM );
      read_node->value = _local_cycles - node->values[0].begin;
      if ( ( read_node = _internal_hl_insert_read_node(&node->values[1].read_values) ) == NULL )
         return ( PAPI_ENOMEM );
      read_node->value = ts - node->values[1].begin;
      for ( i = 0; i < num_of_components; i++ ) {
         for ( j = 0; j < components[i].num_of_events; j++ ) {
            if ( ( read_node = _internal_hl_insert_read_node(&node->values[cmp_iter].read_values) ) == NULL )
               return ( PAPI_ENOMEM );
            if ( components[i].event_types[j] == 1 )
               read_node->value = _local_components[i].values[j];
            else
               read_node->value = _local_components[i].values[j] - node->values[cmp_iter].begin;
            cmp_iter++;
         }
      }
   } else if ( reg_typ == REGION_END ) {
      /* determine difference of current value and begin */
      node->values[0].region_value = _local_cycles - node->values[0].begin;
      node->values[1].region_value = ts - node->values[1].begin;
      /* events from components */
      for ( i = 0; i < num_of_components; i++ )
         for ( j = 0; j < components[i].num_of_events; j++ ) {
            /* if event type is instantaneous only save last value */
            if ( components[i].event_types[j] == 1 ) {
               node->values[cmp_iter].region_value = _local_components[i].values[j];
            } else {
               node->values[cmp_iter].region_value = _local_components[i].values[j] - node->values[cmp_iter].begin;
            }
            cmp_iter++;
         }
   }
   return ( PAPI_OK );
}


static inline regions_t* _internal_hl_insert_region_node(regions_t** head_node, const char *region )
{
   regions_t *new_node;
   int i;
   int extended_total_num_events;

   /* number of all events including CPU cycles and real time */
   extended_total_num_events = total_num_events + 2;

   /* create new region node */
   new_node = malloc(sizeof(regions_t) + extended_total_num_events * sizeof(value_t));
   if ( new_node == NULL )
      return ( NULL );
   new_node->region = (char *)malloc((strlen(region) + 1) * sizeof(char));
   if ( new_node->region == NULL ) {
      free(new_node);
      return ( NULL );
   }

   new_node->next = NULL;
   new_node->prev = NULL;

   new_node->region_id = _local_region_begin_cnt;
   new_node->parent_region_id = _internal_hl_region_id_stack_peak();
   strcpy(new_node->region, region);
   for ( i = 0; i < extended_total_num_events; i++ ) {
      new_node->values[i].read_values = NULL;
   }

   /* insert node in list */
   if ( *head_node == NULL ) {
      *head_node = new_node;
      return new_node;
   }
   (*head_node)->prev = new_node;
   new_node->next = *head_node;
   *head_node = new_node;

   return new_node;
}


static inline regions_t* _internal_hl_find_region_node(regions_t* head_node, const char *region )
{
   regions_t* find_node = head_node;
   while ( find_node != NULL ) {
      if ( ((int)find_node->region_id == _internal_hl_region_id_stack_peak()) && (strcmp(find_node->region, region) == 0) ) {
         return find_node;
      }
      find_node = find_node->next;
   }
   find_node = NULL;
   return find_node;
}

static inline threads_t* _internal_hl_insert_thread_node(unsigned long tid)
{
   threads_t *new_node = (threads_t*)malloc(sizeof(threads_t));
   if ( new_node == NULL )
      return ( NULL );
   new_node->key = tid;
   new_node->value = NULL; /* head node of region list */
   tsearch(new_node, &binary_tree->root, compar);
   return new_node;
}

static inline threads_t* _internal_hl_find_thread_node(unsigned long tid)
{
   threads_t *find_node = binary_tree->find_p;
   find_node->key = tid;
   void *found = tfind(find_node, &binary_tree->root, compar);
   if ( found != NULL ) {
      find_node = (*(threads_t**)found);
      return find_node;
   }
   return NULL;
}


static int _internal_hl_store_counters( unsigned long tid, const char *region,
                                        enum region_type reg_typ )
{
   int retval;

   _papi_hwi_lock( HIGHLEVEL_LOCK );
   threads_t* current_thread_node;

   /* check if current thread is already stored in tree */
   current_thread_node = _internal_hl_find_thread_node(tid);
   if ( current_thread_node == NULL ) {
      /* insert new node for current thread in tree if type is REGION_BEGIN */
      if ( reg_typ == REGION_BEGIN ) {
         if ( ( current_thread_node = _internal_hl_insert_thread_node(tid) ) == NULL ) {
            _papi_hwi_unlock( HIGHLEVEL_LOCK );
            return ( PAPI_ENOMEM );
         }
      } else {
         _papi_hwi_unlock( HIGHLEVEL_LOCK );
         return ( PAPI_EINVAL );
      }
   }

   regions_t* current_region_node;
   if ( reg_typ == REGION_READ || reg_typ == REGION_END ) {
      current_region_node = _internal_hl_find_region_node(current_thread_node->value, region);
      if ( current_region_node == NULL ) {
         if ( reg_typ == REGION_READ ) {
            /* ignore no matching REGION_READ */
            verbose_fprintf(stdout, "PAPI-HL Warning: Cannot find matching region for PAPI_hl_read(\"%s\") for thread id=%lu.\n", region, PAPI_thread_id());
            retval = PAPI_OK;
         } else {
            verbose_fprintf(stdout, "PAPI-HL Warning: Cannot find matching region for PAPI_hl_region_end(\"%s\") for thread id=%lu.\n", region, PAPI_thread_id());
            retval = PAPI_EINVAL;
         }
         _papi_hwi_unlock( HIGHLEVEL_LOCK );
         return ( retval );
      } 
   } else {
      /* create new node for current region in list if type is REGION_BEGIN */
      if ( ( current_region_node = _internal_hl_insert_region_node(&current_thread_node->value, region) ) == NULL ) {
         _papi_hwi_unlock( HIGHLEVEL_LOCK );
         return ( PAPI_ENOMEM );
      }
   }


   /* add recorded values to current region */
   if ( ( retval = _internal_hl_add_values_to_region( current_region_node, reg_typ ) ) != PAPI_OK ) {
      _papi_hwi_unlock( HIGHLEVEL_LOCK );
      return ( retval );
   }

   /* count all REGION_BEGIN and REGION_END calls */
   if ( reg_typ == REGION_BEGIN ) region_begin_cnt++;
   if ( reg_typ == REGION_END ) region_end_cnt++;

   _papi_hwi_unlock( HIGHLEVEL_LOCK );
   return ( PAPI_OK );
}


static int _internal_hl_read_counters()
{
   int i, j, retval;
   for ( i = 0; i < num_of_components; i++ ) {
      if ( i < ( num_of_components - 1 ) ) {
         retval = PAPI_read( _local_components[i].EventSet, _local_components[i].values);
      } else {
         /* get cycles for last component */
         retval = PAPI_read_ts( _local_components[i].EventSet, _local_components[i].values, &_local_cycles );
      }
      HLDBG("Thread-ID:%lu, Component-ID:%d\n", PAPI_thread_id(), components[i].component_id);
      for ( j = 0; j < components[i].num_of_events; j++ ) {
        HLDBG("Thread-ID:%lu, %s:%lld\n", PAPI_thread_id(), components[i].event_names[j], _local_components[i].values[j]);
      }

      if ( retval != PAPI_OK )
         return ( retval );
   }
   return ( PAPI_OK );
}

static int _internal_hl_read_and_store_counters( const char *region, enum region_type reg_typ )
{
   int retval;
   /* read all events */
   if ( ( retval = _internal_hl_read_counters() ) != PAPI_OK ) {
      verbose_fprintf(stdout, "PAPI-HL Error: Could not read counters for thread %lu.\n", PAPI_thread_id());
      _internal_hl_clean_up_all(true);
      return ( retval );
   }

   /* store all events */
   if ( ( retval = _internal_hl_store_counters( PAPI_thread_id(), region, reg_typ) ) != PAPI_OK ) {
      verbose_fprintf(stdout, "PAPI-HL Error: Could not store counters for thread %lu.\n", PAPI_thread_id());
      verbose_fprintf(stdout, "PAPI-HL Advice: Check if your regions are matching.\n");
      _internal_hl_clean_up_all(true);
      return ( retval );
   }
   return ( PAPI_OK );
}

static int _internal_hl_create_global_binary_tree()
{
   if ( ( binary_tree = (binary_tree_t*)malloc(sizeof(binary_tree_t)) ) == NULL )
      return ( PAPI_ENOMEM );
   binary_tree->root = NULL;
   if ( ( binary_tree->find_p = (threads_t*)malloc(sizeof(threads_t)) ) == NULL )
      return ( PAPI_ENOMEM );
   return ( PAPI_OK );
}


static int _internal_hl_mkdir(const char *dir)
{
   int retval;
   int errno;
   char *tmp = NULL;
   char *p = NULL;
   size_t len;

   if ( ( tmp = strdup(dir) ) == NULL )
      return ( PAPI_ENOMEM );
   len = strlen(tmp);

   /* check if there is a file with the same name as the ouptut directory */
   struct stat buf;
   if ( stat(dir, &buf) == 0 && S_ISREG(buf.st_mode) ) {
      verbose_fprintf(stdout, "PAPI-HL Error: Name conflict with measurement directory and existing file.\n");
      return ( PAPI_ESYS );
   }

   if(tmp[len - 1] == '/')
      tmp[len - 1] = 0;
   for(p = tmp + 1; *p; p++)
   {
      if(*p == '/')
      {
         *p = 0;
         errno = 0;
         retval = mkdir(tmp, S_IRWXU);
         *p = '/';
         if ( retval != 0 && errno != EEXIST ) {
            free(tmp);
            return ( PAPI_ESYS );
         }
      }
   }
   retval = mkdir(tmp, S_IRWXU);
   free(tmp);
   if ( retval != 0 && errno != EEXIST )
      return ( PAPI_ESYS );

   return ( PAPI_OK );
}

static int _internal_hl_determine_output_path()
{
   /* check if PAPI_OUTPUT_DIRECTORY is set */
   char *output_prefix = NULL;
   if ( getenv("PAPI_OUTPUT_DIRECTORY") != NULL ) {
      if ( ( output_prefix = strdup( getenv("PAPI_OUTPUT_DIRECTORY") ) ) == NULL )
         return ( PAPI_ENOMEM );
   } else {
      if ( ( output_prefix = strdup( getcwd(NULL,0) ) ) == NULL )
         return ( PAPI_ENOMEM );
   }
   
   /* generate absolute path for measurement directory */
   if ( ( absolute_output_file_path = (char *)malloc((strlen(output_prefix) + 64) * sizeof(char)) ) == NULL ) {
      free(output_prefix);
      return ( PAPI_ENOMEM );
   }
   if ( output_counter > 0 )
      sprintf(absolute_output_file_path, "%s/papi_hl_output_%d", output_prefix, output_counter);
   else
      sprintf(absolute_output_file_path, "%s/papi_hl_output", output_prefix);

   /* check if directory already exists */
   struct stat buf;
   if ( stat(absolute_output_file_path, &buf) == 0 && S_ISDIR(buf.st_mode) ) {

      /* rename old directory by adding a timestamp */
      char *new_absolute_output_file_path = NULL;
      if ( ( new_absolute_output_file_path = (char *)malloc((strlen(absolute_output_file_path) + 64) * sizeof(char)) ) == NULL ) {
         free(output_prefix);
         free(absolute_output_file_path);
         return ( PAPI_ENOMEM );
      }

      /* create timestamp */
      time_t t = time(NULL);
      struct tm tm = *localtime(&t);
      char m_time[32];
      sprintf(m_time, "%d%02d%02d-%02d%02d%02d", tm.tm_year+1900, tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec);
      /* add timestamp to existing folder string */
      sprintf(new_absolute_output_file_path, "%s-%s", absolute_output_file_path, m_time);

      uintmax_t current_unix_time = (uintmax_t)t;
      uintmax_t unix_time_from_old_directory = buf.st_mtime;

      /* This is a workaround for MPI applications!!!
       * Only rename existing measurement directory when it is older than
       * current timestamp. If it's not, we assume that another MPI process already created a 
       * new measurement directory. */
      if ( unix_time_from_old_directory < current_unix_time ) {

         if ( rename(absolute_output_file_path, new_absolute_output_file_path) != 0 ) {
            verbose_fprintf(stdout, "PAPI-HL Warning: Cannot rename old measurement directory.\n");
            verbose_fprintf(stdout, "If you use MPI, another process may have already renamed the directory.\n");
         }
      }

      free(new_absolute_output_file_path);
   }
   free(output_prefix);
   output_counter++;

   return ( PAPI_OK );
}

static void _internal_hl_json_line_break_and_indent( FILE* f, bool b, int width )
{
   int i;
   if ( b ) {
      fprintf(f, "\n");
      for ( i = 0; i < width; ++i )
         fprintf(f, "  ");
   }
}

static void _internal_hl_json_definitions(FILE* f, bool beautifier)
{
   int num_events, i, j;

   _internal_hl_json_line_break_and_indent(f, beautifier, 1);
   fprintf(f, "\"event_definitions\":{");

   /* get all events + types */
   num_events = 1;
   for ( i = 0; i < num_of_components; i++ ) {
      for ( j = 0; j < components[i].num_of_events; j++ ) {
         _internal_hl_json_line_break_and_indent(f, beautifier, 2);

         const char *event_type = "delta";
         if ( components[i].event_types[j] == 1 )
            event_type = "instant";
         const PAPI_component_info_t* cmpinfo;
         cmpinfo = PAPI_get_component_info( components[i].component_id );

         fprintf(f, "\"%s\":{", components[i].event_names[j]);
         _internal_hl_json_line_break_and_indent(f, beautifier, 3);
         fprintf(f, "\"component\":\"%s\",", cmpinfo->name);
         _internal_hl_json_line_break_and_indent(f, beautifier, 3);
         fprintf(f, "\"type\":\"%s\"", event_type);
         _internal_hl_json_line_break_and_indent(f, beautifier, 2);
         fprintf(f, "}");
         if ( num_events < total_num_events )
            fprintf(f, ",");
         num_events++;
      }
   }

   _internal_hl_json_line_break_and_indent(f, beautifier, 1);
   fprintf(f, "},");
}

static void _internal_hl_json_region_events(FILE* f, bool beautifier, regions_t *regions)
{
   char **all_event_names = NULL;
   int *all_event_types = NULL;
   int extended_total_num_events;
   int i, j, cmp_iter;

   /* generate array of all events including CPU cycles and real time for output */
   extended_total_num_events = total_num_events + 2;
   all_event_names = (char**)malloc(extended_total_num_events * sizeof(char*));
   all_event_names[0] = "cycles";
   all_event_names[1] = "real_time_nsec";

   all_event_types = (int*)malloc(extended_total_num_events * sizeof(int));
   all_event_types[0] = 0;
   all_event_types[1] = 0;


   cmp_iter = 2;
   for ( i = 0; i < num_of_components; i++ ) {
      for ( j = 0; j < components[i].num_of_events; j++ ) {
         all_event_names[cmp_iter] = components[i].event_names[j];
         if ( components[i].event_types[j] == 0 )
            all_event_types[cmp_iter] = 0;
         else
            all_event_types[cmp_iter] = 1;
         cmp_iter++;
      }
   }

   for ( j = 0; j < extended_total_num_events; j++ ) {

      _internal_hl_json_line_break_and_indent(f, beautifier, 5);

      /* print read values if available */
      if ( regions->values[j].read_values != NULL) {
         reads_t* read_node = regions->values[j].read_values;
         /* going to last node */
         while ( read_node->next != NULL ) {
            read_node = read_node->next;
         }
         /* read values in reverse order */
         int read_cnt = 1;
         fprintf(f, "\"%s\":{", all_event_names[j]);

         _internal_hl_json_line_break_and_indent(f, beautifier, 6);
         fprintf(f, "\"region_value\":\"%lld\",", regions->values[j].region_value);

         while ( read_node != NULL ) {
            _internal_hl_json_line_break_and_indent(f, beautifier, 6);
            fprintf(f, "\"read_%d\":\"%lld\"", read_cnt,read_node->value);

            read_node = read_node->prev;

            if ( read_node == NULL ) {
               _internal_hl_json_line_break_and_indent(f, beautifier, 5);
               fprintf(f, "}");
               if ( j < extended_total_num_events - 1 )
                  fprintf(f, ",");
            } else {
               fprintf(f, ",");
            }

            read_cnt++;
         }
      } else {
         HLDBG("  %s:%lld\n", all_event_names[j], regions->values[j].region_value);
         fprintf(f, "\"%s\":\"%lld\"", all_event_names[j], regions->values[j].region_value);
         if ( j < ( extended_total_num_events - 1 ) )
            fprintf(f, ",");
      }
   }

   free(all_event_names);
   free(all_event_types);
}

static void _internal_hl_json_regions(FILE* f, bool beautifier, threads_t* thread_node)
{
   /* iterate over regions list */
   regions_t *regions = thread_node->value;

   /* going to last node */
   while ( regions->next != NULL ) {
      regions = regions->next;
   }

   /* read regions in reverse order */
   while (regions != NULL) {
      HLDBG("  Region:%u\n", regions->region_id);

      _internal_hl_json_line_break_and_indent(f, beautifier, 4);
      fprintf(f, "\"%u\":{", regions->region_id);

      _internal_hl_json_line_break_and_indent(f, beautifier, 5);
      fprintf(f, "\"name\":\"%s\",", regions->region);
      _internal_hl_json_line_break_and_indent(f, beautifier, 5);
      fprintf(f, "\"parent_region_id\":\"%d\",", regions->parent_region_id);

      _internal_hl_json_region_events(f, beautifier, regions);

      regions = regions->prev;
      _internal_hl_json_line_break_and_indent(f, beautifier, 4);
      if (regions == NULL ) {
         fprintf(f, "}");
      } else {
         fprintf(f, "},");
      }
   }
}

static void _internal_hl_json_threads(FILE* f, bool beautifier, unsigned long* tids, int threads_num)
{
   int i;

   _internal_hl_json_line_break_and_indent(f, beautifier, 1);
   fprintf(f, "\"threads\":{");

   /* get regions of all threads */
   for ( i = 0; i < threads_num; i++ )
   {
      HLDBG("Thread ID:%lu\n", tids[i]);
      /* find values of current thread in global binary tree */
      threads_t* thread_node = _internal_hl_find_thread_node(tids[i]);
      if ( thread_node != NULL ) {
         /* do we really need the exact thread id? */
         /* we only store iterator id as thread id, not tids[i] */
         _internal_hl_json_line_break_and_indent(f, beautifier, 2);
         fprintf(f, "\"%d\":{", i);

         _internal_hl_json_line_break_and_indent(f, beautifier, 3);
         fprintf(f, "\"regions\":{");

         _internal_hl_json_regions(f, beautifier, thread_node);

         _internal_hl_json_line_break_and_indent(f, beautifier, 3);
         fprintf(f, "}");

         _internal_hl_json_line_break_and_indent(f, beautifier, 2);
         if ( i < threads_num - 1 ) {
            fprintf(f, "},");
         } else {
            fprintf(f, "}");
         }
      }
   }

   _internal_hl_json_line_break_and_indent(f, beautifier, 1);
   fprintf(f, "}");

}

static int _internal_hl_cmpfunc(const void * a, const void * b) {
   return ( *(int*)a - *(int*)b );
}

static int _internal_get_sorted_thread_list(unsigned long** tids, int* threads_num)
{
   if ( PAPI_list_threads( *tids, threads_num ) != PAPI_OK ) {
      verbose_fprintf(stdout, "PAPI-HL Error: PAPI_list_threads call failed!\n");
      return -1;
   }
   if ( ( *tids = malloc( *(threads_num) * sizeof(unsigned long) ) ) == NULL ) {
      verbose_fprintf(stdout, "PAPI-HL Error: OOM!\n");
      return -1;
   }
   if ( PAPI_list_threads( *tids, threads_num ) != PAPI_OK ) {
      verbose_fprintf(stdout, "PAPI-HL Error: PAPI_list_threads call failed!\n");
      return -1;
   }

   /* sort thread ids in ascending order */
   qsort(*tids, *(threads_num), sizeof(unsigned long), _internal_hl_cmpfunc);
   return PAPI_OK;
}

static void _internal_hl_write_json_file(FILE* f, unsigned long* tids, int threads_num)
{
   /* JSON beautifier (line break and indent) */
   bool beautifier = true;

   /* start of JSON file */
   fprintf(f, "{");
   _internal_hl_json_line_break_and_indent(f, beautifier, 1);
   fprintf(f, "\"papi_version\":\"%d.%d.%d.%d\",", PAPI_VERSION_MAJOR( PAPI_VERSION ),
      PAPI_VERSION_MINOR( PAPI_VERSION ),
      PAPI_VERSION_REVISION( PAPI_VERSION ),
      PAPI_VERSION_INCREMENT( PAPI_VERSION ) );

   /* add some hardware info */
   const PAPI_hw_info_t *hwinfo;
   if ( ( hwinfo = PAPI_get_hardware_info(  ) ) != NULL ) {
      _internal_hl_json_line_break_and_indent(f, beautifier, 1);
      char* cpu_info = _internal_hl_remove_spaces(strdup(hwinfo->model_string), 1);
      fprintf(f, "\"cpu_info\":\"%s\",", cpu_info);
      free(cpu_info);
      _internal_hl_json_line_break_and_indent(f, beautifier, 1);
      fprintf(f, "\"max_cpu_rate_mhz\":\"%d\",", hwinfo->cpu_max_mhz);
      _internal_hl_json_line_break_and_indent(f, beautifier, 1);
      fprintf(f, "\"min_cpu_rate_mhz\":\"%d\",", hwinfo->cpu_min_mhz);
   }

   /* write definitions */
   _internal_hl_json_definitions(f, beautifier);

   /* write all regions with events per thread */
   _internal_hl_json_threads(f, beautifier, tids, threads_num);

   /* end of JSON file */
   _internal_hl_json_line_break_and_indent(f, beautifier, 0);
   fprintf(f, "}");
   fprintf(f, "\n");
}

static void _internal_hl_read_json_file(const char* path)
{
   /* print output to stdout */
   printf("\n\nPAPI-HL Output:\n");
   FILE* output_file = fopen(path, "r");
   int c = fgetc(output_file);
   while (c != EOF)
   {
      printf("%c", c);
      c = fgetc(output_file);
   }
   printf("\n");
   fclose(output_file);
}

static void _internal_hl_write_output()
{
   if ( output_generated == false )
   {
      _papi_hwi_lock( HIGHLEVEL_LOCK );
      if ( output_generated == false ) {
         /* check if events were recorded */
         if ( binary_tree == NULL ) {
            verbose_fprintf(stdout, "PAPI-HL Info: No events were recorded.\n");
            free(absolute_output_file_path);
            return;
         }

         if ( region_begin_cnt == region_end_cnt ) {
            verbose_fprintf(stdout, "PAPI-HL Info: Print results...\n");
         } else {
            verbose_fprintf(stdout, "PAPI-HL Warning: Cannot generate output due to not matching regions.\n");
            output_generated = true;
            HLDBG("region_begin_cnt=%d, region_end_cnt=%d\n", region_begin_cnt, region_end_cnt);
            _papi_hwi_unlock( HIGHLEVEL_LOCK );
            free(absolute_output_file_path);
            return;
         }

         /* create new measurement directory */
         if ( ( _internal_hl_mkdir(absolute_output_file_path) ) != PAPI_OK ) {
            verbose_fprintf(stdout, "PAPI-HL Error: Cannot create measurement directory %s.\n", absolute_output_file_path);
            free(absolute_output_file_path);
            return;
         }

         /* determine rank for output file */
         int rank = _internal_hl_determine_rank();

         /* if system does not provide rank id, create a random id */
         if ( rank < 0 ) {
            srandom( time(NULL) + getpid() );
            rank = random() % 1000000;
         }

         int unique_output_file_created = 0;
         char *final_absolute_output_file_path = NULL;
         int fd;
         int random_cnt = 0;

         /* allocate memory for final output file path */
         if ( ( final_absolute_output_file_path = (char *)malloc((strlen(absolute_output_file_path) + 20) * sizeof(char)) ) == NULL ) {
            verbose_fprintf(stdout, "PAPI-HL Error: Cannot create output file.\n");
            free(absolute_output_file_path);
            free(final_absolute_output_file_path);
            return;
         }

         /* create unique output file per process based on rank variable */
         while ( unique_output_file_created == 0 ) {
            rank += random_cnt;
            sprintf(final_absolute_output_file_path, "%s/rank_%06d.json", absolute_output_file_path, rank);

            fd = open(final_absolute_output_file_path, O_WRONLY|O_APPEND|O_CREAT|O_NONBLOCK, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
            if ( fd == -1 ) {
               verbose_fprintf(stdout, "PAPI-HL Error: Cannot create output file.\n");
               free(absolute_output_file_path);
               free(final_absolute_output_file_path);
               return;
            }

            struct flock filelock;
            filelock.l_type   = F_WRLCK; /* Test for any lock on any part of file. */
            filelock.l_start  = 0;
            filelock.l_whence = SEEK_SET;
            filelock.l_len    = 0;

            if ( fcntl(fd, F_SETLK, &filelock) == 0 ) {
               unique_output_file_created = 1;
               free(absolute_output_file_path);

               /* write into file */
               FILE *fp = fdopen(fd, "w");
               if ( fp != NULL ) {
                  
                  /* list all threads */
                  unsigned long *tids = NULL;
                  int threads_num;
                  if ( _internal_get_sorted_thread_list(&tids, &threads_num) != PAPI_OK ) {
                     fclose(fp);
                     free(final_absolute_output_file_path);
                     return;
                  }

                  /* start writing json output */
                  _internal_hl_write_json_file(fp, tids, threads_num);
                  free(tids);
                  fclose(fp);

                  if ( getenv("PAPI_REPORT") != NULL ) {
                     _internal_hl_read_json_file(final_absolute_output_file_path);
                  }

               } else {
                  verbose_fprintf(stdout, "PAPI-HL Error: Cannot create output file: %s\n", strerror( errno ));
                  free(final_absolute_output_file_path);
                  fcntl(fd, F_UNLCK, &filelock);
                  return;
               }
               fcntl(fd, F_UNLCK, &filelock);
            } else {
               /* try another file name */
               close(fd);
               random_cnt++;
            }
         }

         output_generated = true;
         free(final_absolute_output_file_path);
      }
      _papi_hwi_unlock( HIGHLEVEL_LOCK );
   }
}

static void _internal_hl_clean_up_local_data()
{
   int i, retval;
   /* destroy all EventSets from local data */
   if ( _local_components != NULL ) {
      HLDBG("Thread-ID:%lu\n", PAPI_thread_id());
      for ( i = 0; i < num_of_components; i++ ) {
         if ( ( retval = PAPI_stop( _local_components[i].EventSet, _local_components[i].values ) ) != PAPI_OK )
            /* only print error when event set is running */
            if ( retval != -9 )
              verbose_fprintf(stdout, "PAPI-HL Error: PAPI_stop failed: %d.\n", retval);
         if ( ( retval = PAPI_cleanup_eventset (_local_components[i].EventSet) ) != PAPI_OK )
            verbose_fprintf(stdout, "PAPI-HL Error: PAPI_cleanup_eventset failed: %d.\n", retval);
         if ( ( retval = PAPI_destroy_eventset (&_local_components[i].EventSet) ) != PAPI_OK )
            verbose_fprintf(stdout, "PAPI-HL Error: PAPI_destroy_eventset failed: %d.\n", retval);
         free(_local_components[i].values);
      }
      free(_local_components);
      _local_components = NULL;

      /* count global thread variable */
      _papi_hwi_lock( HIGHLEVEL_LOCK );
      num_of_cleaned_threads++;
      _papi_hwi_unlock( HIGHLEVEL_LOCK );
   }
   _papi_hl_events_running = 0;
   _local_state = PAPIHL_DEACTIVATED;
}

static void _internal_hl_clean_up_global_data()
{
   int i;
   int extended_total_num_events;

   /* clean up binary tree of recorded events */
   threads_t *thread_node;
   if ( binary_tree != NULL ) {
      while ( binary_tree->root != NULL ) {
         thread_node = *(threads_t **)binary_tree->root;

         /* clean up double linked list of region data */
         regions_t *region = thread_node->value;
         regions_t *tmp;
         while ( region != NULL ) {

            /* clean up read node list */
            extended_total_num_events = total_num_events + 2;
            for ( i = 0; i < extended_total_num_events; i++ ) {
               reads_t *read_node = region->values[i].read_values;
               reads_t *read_node_tmp;
               while ( read_node != NULL ) {
                  read_node_tmp = read_node;
                  read_node = read_node->next;
                  free(read_node_tmp);
               }
            }

            tmp = region;
            region = region->next;

            free(tmp->region);
            free(tmp);
         }
         free(region);

         tdelete(thread_node, &binary_tree->root, compar);
         free(thread_node);
      }
   }

   /* we cannot free components here since other threads could still use them */

   /* clean up requested event names */
   for ( i = 0; i < num_of_requested_events; i++ )
      free(requested_event_names[i]);
   free(requested_event_names);

   free(absolute_output_file_path);
}

static void _internal_hl_clean_up_all(bool deactivate)
{
   int i, num_of_threads;

   /* we assume that output has been already generated or
    * cannot be generated due to previous errors */
   output_generated = true;

   /* clean up thread local data */
   if ( _local_state == PAPIHL_ACTIVE ) {
     HLDBG("Clean up thread local data for thread %lu\n", PAPI_thread_id());
     _internal_hl_clean_up_local_data();
   }

   /* clean up global data */
   if ( state == PAPIHL_ACTIVE ) {
      _papi_hwi_lock( HIGHLEVEL_LOCK );
      if ( state == PAPIHL_ACTIVE ) {

         verbose_fprintf(stdout, "PAPI-HL Info: Output generation is deactivated!\n");

         HLDBG("Clean up global data for thread %lu\n", PAPI_thread_id());
         _internal_hl_clean_up_global_data();

         /* check if all other registered threads have cleaned up */
         PAPI_list_threads(NULL, &num_of_threads);

         HLDBG("Number of registered threads: %d.\n", num_of_threads);
         HLDBG("Number of cleaned threads: %d.\n", num_of_cleaned_threads);

         if ( _internal_hl_check_for_clean_thread_states() == PAPI_OK &&
               num_of_threads == num_of_cleaned_threads ) {
            PAPI_shutdown();
            /* clean up components */
            for ( i = 0; i < num_of_components; i++ ) {
               free(components[i].event_names);
               free(components[i].event_codes);
               free(components[i].event_types);
            }
            free(components);
            HLDBG("PAPI-HL shutdown!\n");
         } else {
            verbose_fprintf(stdout, "PAPI-HL Warning: Could not call PAPI_shutdown() since some threads still have running event sets.\n");
         }

         /* deactivate PAPI-HL */
         if ( deactivate )
            state = PAPIHL_DEACTIVATED;
      }
      _papi_hwi_unlock( HIGHLEVEL_LOCK );
   }
}

static int _internal_hl_check_for_clean_thread_states()
{
   EventSetInfo_t *ESI;
   DynamicArray_t *map = &_papi_hwi_system_info.global_eventset_map;
   int i;

   for( i = 0; i < map->totalSlots; i++ ) {
      ESI = map->dataSlotArray[i];
      if ( ESI ) {
         if ( ESI->state & PAPI_RUNNING ) 
            return ( PAPI_EISRUN );
      }
   }
   return ( PAPI_OK );
}

int
_internal_PAPI_hl_init()
{
   if ( state == PAPIHL_ACTIVE ) {
      if ( hl_initiated == false && hl_finalized == false ) {
         _internal_hl_onetime_library_init();
         /* check if the library has been initialized successfully */
         if ( state == PAPIHL_DEACTIVATED )
            return ( PAPI_EMISC );
         return ( PAPI_OK );
      }
      return ( PAPI_ENOINIT );
   }
   return ( PAPI_EMISC );
}

int _internal_PAPI_hl_cleanup_thread()
{
   if ( state == PAPIHL_ACTIVE && 
        hl_initiated == true && 
        _local_state == PAPIHL_ACTIVE ) {
         /* do not clean local data from master thread */
         if ( master_thread_id != PAPI_thread_id() )
           _internal_hl_clean_up_local_data();
         return ( PAPI_OK );
      }
   return ( PAPI_EMISC );
}

int _internal_PAPI_hl_finalize()
{
   if ( state == PAPIHL_ACTIVE && hl_initiated == true ) {
      _internal_hl_clean_up_all(true);
      return ( PAPI_OK );
   }
   return ( PAPI_EMISC );
}

int
_internal_PAPI_hl_set_events(const char* events)
{
   int retval;
   if ( state == PAPIHL_ACTIVE ) {

      /* This may only be called once after the high-level API was successfully
       * initiated. Any second call just returns PAPI_OK without doing an
       * expensive lock. */
      if ( hl_initiated == true ) {
         if ( events_determined == false )
         {
            _papi_hwi_lock( HIGHLEVEL_LOCK );
            if ( events_determined == false && state == PAPIHL_ACTIVE )
            {
               HLDBG("Set events: %s\n", events);
               if ( ( retval = _internal_hl_read_events(events) ) != PAPI_OK ) {
                  state = PAPIHL_DEACTIVATED;
                  _internal_hl_clean_up_global_data();
                  _papi_hwi_unlock( HIGHLEVEL_LOCK );
                  return ( retval );
               }
               if ( ( retval = _internal_hl_create_global_binary_tree() ) != PAPI_OK ) {
                  state = PAPIHL_DEACTIVATED;
                  _internal_hl_clean_up_global_data();
                  _papi_hwi_unlock( HIGHLEVEL_LOCK );
                  return ( retval );
               }
            }
            _papi_hwi_unlock( HIGHLEVEL_LOCK );
         }
      }
      /* in case the first locked thread ran into problems */
      if ( state == PAPIHL_DEACTIVATED)
         return ( PAPI_EMISC );
      return ( PAPI_OK );
   }
   return ( PAPI_EMISC );
}

void
_internal_PAPI_hl_print_output()
{
   if ( state == PAPIHL_ACTIVE && 
        hl_initiated == true && 
        output_generated == false ) {
      _internal_hl_write_output();
   }
}

/** @class PAPI_hl_region_begin
 * @brief Read performance events at the beginning of a region.
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_hl_region_begin( const char* region );
 *
 * @param region
 * -- a unique region name
 *
 * @retval PAPI_OK
 * @retval PAPI_ENOTRUN
 * -- EventSet is currently not running or could not determined.
 * @retval PAPI_ESYS
 * -- A system or C library call failed inside PAPI, see the errno variable.
 * @retval PAPI_EMISC
 * -- PAPI has been deactivated due to previous errors.
 * @retval PAPI_ENOMEM
 * -- Insufficient memory.
 *
 * PAPI_hl_region_begin reads performance events and stores them internally at the beginning
 * of an instrumented code region.
 * If not specified via the environment variable PAPI_EVENTS, default events are used.
 * The first call sets all counters implicitly to zero and starts counting.
 * Note that if PAPI_EVENTS is not set or cannot be interpreted, default performance events are
 * recorded.
 *
 * @par Example:
 *
 * @code
 * export PAPI_EVENTS="PAPI_TOT_INS,PAPI_TOT_CYC"
 *
 * @endcode
 *
 *
 * @code
 * int retval;
 *
 * retval = PAPI_hl_region_begin("computation");
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 *  //Do some computation here
 *
 * retval = PAPI_hl_region_end("computation");
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 * @endcode
 *
 * @see PAPI_hl_read
 * @see PAPI_hl_region_end
 * @see PAPI_hl_stop
 */
int
PAPI_hl_region_begin( const char* region )
{
   int retval;
   /* if a rate event set is running stop it */
   if ( _papi_rate_events_running == 1 ) {
      if ( ( retval = PAPI_rate_stop() ) != PAPI_OK )
         return ( retval );
   }

   if ( state == PAPIHL_DEACTIVATED ) {
      /* check if we have to clean up local stuff */
      if ( _local_state == PAPIHL_ACTIVE )
         _internal_hl_clean_up_local_data();
      return ( PAPI_EMISC );
   }

   if ( hl_finalized == true )
      return ( PAPI_ENOTRUN );

   if ( hl_initiated == false ) {
      if ( ( retval = _internal_PAPI_hl_init() ) != PAPI_OK )
         return ( retval );
   }

   if ( events_determined == false ) {
      if ( ( retval = _internal_PAPI_hl_set_events(NULL) ) != PAPI_OK )
         return ( retval );
   }

   if ( _local_components == NULL ) {
      if ( ( retval = _internal_hl_create_event_sets() ) != PAPI_OK ) {
         HLDBG("Could not create local events sets for thread %lu.\n", PAPI_thread_id());
         _internal_hl_clean_up_all(true);
         return ( retval );
      }
   }

   if ( _papi_hl_events_running == 0 ) {
      if ( ( retval = _internal_hl_start_counters() ) != PAPI_OK ) {
         HLDBG("Could not start counters for thread %lu.\n", PAPI_thread_id());
         _internal_hl_clean_up_all(true);
         return ( retval );
      }
   }

   /* read and store all events */
   HLDBG("Thread ID:%lu, Region:%s\n", PAPI_thread_id(), region);
   if ( ( retval = _internal_hl_read_and_store_counters(region, REGION_BEGIN) ) != PAPI_OK )
      return ( retval );

   if ( ( retval = _internal_hl_region_id_push() ) != PAPI_OK ) {
      verbose_fprintf(stdout, "PAPI-HL Warning: Number of nested regions exceeded for thread %lu.\n", PAPI_thread_id());
      _internal_hl_clean_up_all(true);
      return ( retval );
   }
   _local_region_begin_cnt++;
   return ( PAPI_OK );
}

/** @class PAPI_hl_read
 * @brief Read performance events inside of a region and store the difference to the corresponding
 * beginning of the region.
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_hl_read( const char* region );
 *
 * @param region
 * -- a unique region name corresponding to PAPI_hl_region_begin
 *
 * @retval PAPI_OK
 * @retval PAPI_ENOTRUN
 * -- EventSet is currently not running or could not determined.
 * @retval PAPI_ESYS
 * -- A system or C library call failed inside PAPI, see the errno variable.
 * @retval PAPI_EMISC
 * -- PAPI has been deactivated due to previous errors.
 * @retval PAPI_ENOMEM
 * -- Insufficient memory.
 *
 * PAPI_hl_read reads performance events inside of a region and stores the difference to the 
 * corresponding beginning of the region.
 *
 * Assumes that PAPI_hl_region_begin was called before.
 *
 * @par Example:
 *
 * @code
 * int retval;
 *
 * retval = PAPI_hl_region_begin("computation");
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 *  //Do some computation here
 *
 * retval = PAPI_hl_read("computation");
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 *  //Do some computation here
 *
 * retval = PAPI_hl_region_end("computation");
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 * @endcode
 *
 * @see PAPI_hl_region_begin
 * @see PAPI_hl_region_end
 * @see PAPI_hl_stop
 */
int
PAPI_hl_read(const char* region)
{
   int retval;

   if ( state == PAPIHL_DEACTIVATED ) {
      /* check if we have to clean up local stuff */
      if ( _local_state == PAPIHL_ACTIVE )
         _internal_hl_clean_up_local_data();
      return ( PAPI_EMISC );
   }

   if ( _local_region_begin_cnt == 0 ) {
      verbose_fprintf(stdout, "PAPI-HL Warning: Cannot find matching region for PAPI_hl_read(\"%s\") for thread %lu.\n", region, PAPI_thread_id());
      return ( PAPI_EMISC );
   }

   if ( _local_components == NULL )
      return ( PAPI_ENOTRUN );

   /* read and store all events */
   HLDBG("Thread ID:%lu, Region:%s\n", PAPI_thread_id(), region);
   if ( ( retval = _internal_hl_read_and_store_counters(region, REGION_READ) ) != PAPI_OK )
      return ( retval );

   return ( PAPI_OK );
}

/** @class PAPI_hl_region_end
 * @brief Read performance events at the end of a region and store the difference to the
 * corresponding beginning of the region.
 *
 * @par C Interface:
 * \#include <papi.h> @n
 * int PAPI_hl_region_end( const char* region );
 *
 * @param region
 * -- a unique region name corresponding to PAPI_hl_region_begin
 *
 * @retval PAPI_OK
 * @retval PAPI_ENOTRUN
 * -- EventSet is currently not running or could not determined.
 * @retval PAPI_ESYS
 * -- A system or C library call failed inside PAPI, see the errno variable.
 * @retval PAPI_EMISC
 * -- PAPI has been deactivated due to previous errors.
 * @retval PAPI_ENOMEM
 * -- Insufficient memory.
 *
 * PAPI_hl_region_end reads performance events at the end of a region and stores the
 * difference to the corresponding beginning of the region.
 * 
 * Assumes that PAPI_hl_region_begin was called before.
 * 
 * Note that PAPI_hl_region_end does not stop counting the performance events. Counting
 * continues until the application terminates. Therefore, the programmer can also create
 * nested regions if required. To stop a running high-level event set, the programmer must call
 * PAPI_hl_stop(). It should also be noted, that a marked region is thread-local and therefore
 * has to be in the same thread.
 * 
 * An output of the measured events is created automatically after the application exits.
 * In the case of a serial, or a thread-parallel application there is only one output file.
 * MPI applications would be saved in multiple files, one per MPI rank.
 * The output is generated in the current directory by default. However, it is recommended to
 * specify an output directory for larger measurements, especially for MPI applications via
 * the environment variable PAPI_OUTPUT_DIRECTORY. In the case where measurements are performed,
 * while there are old measurements in the same directory, PAPI will not overwrite or delete the
 * old measurement directories. Instead, timestamps are added to the old directories.
 * 
 * For more convenience, the output can also be printed to stdout by setting PAPI_REPORT=1. This
 * is not recommended for MPI applications as each MPI rank tries to print the output concurrently.
 *
 * The generated measurement output can also be converted in a better readable output. The python
 * script papi_hl_output_writer.py enhances the output by creating some derived metrics, like IPC,
 * MFlops/s, and MFlips/s as well as real and processor time in case the corresponding PAPI events
 * have been recorded. The python script can also summarize performance events over all threads and
 * MPI ranks when using the option "accumulate" as seen below.
 * 
 * @par Example:
 *
 * @code
 * int retval;
 *
 * retval = PAPI_hl_region_begin("computation");
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 *  //Do some computation here
 *
 * retval = PAPI_hl_region_end("computation");
 * if ( retval != PAPI_OK )
 *     handle_error(1);
 *
 * @endcode
 *
 * @code
 * python papi_hl_output_writer.py --type=accumulate
 *
 * {
 *    "computation": {
 *       "Region count": 1,
 *       "Real time in s": 0.97 ,
 *       "CPU time in s": 0.98 ,
 *       "IPC": 1.41 ,
 *       "MFLIPS /s": 386.28 ,
 *       "MFLOPS /s": 386.28 ,
 *       "Number of ranks ": 1,
 *       "Number of threads ": 1,
 *       "Number of processes ": 1
 *    }
 * }
 *
 * @endcode
 * 
 * @see PAPI_hl_region_begin
 * @see PAPI_hl_read
 * @see PAPI_hl_stop
 */
int
PAPI_hl_region_end( const char* region )
{
   int retval;

   if ( state == PAPIHL_DEACTIVATED ) {
      /* check if we have to clean up local stuff */
      if ( _local_state == PAPIHL_ACTIVE )
         _internal_hl_clean_up_local_data();
      return ( PAPI_EMISC );
   }

   if ( _local_region_begin_cnt == 0 ) {
      verbose_fprintf(stdout, "PAPI-HL Warning: Cannot find matching region for PAPI_hl_region_end(\"%s\") for thread %lu.\n", region, PAPI_thread_id());
      return ( PAPI_EMISC );
   }

   if ( _local_components == NULL )
      return ( PAPI_ENOTRUN );

   /* read and store all events */
   HLDBG("Thread ID:%lu, Region:%s\n", PAPI_thread_id(), region);
   if ( ( retval = _internal_hl_read_and_store_counters(region, REGION_END) ) != PAPI_OK )
      return ( retval );

   _internal_hl_region_id_pop();
   _local_region_end_cnt++;
   return ( PAPI_OK );
}

/** @class PAPI_hl_stop
  * @brief Stop a running high-level event set.
  *
  * @par C Interface: 
  * \#include <papi.h> @n
  * int PAPI_hl_stop();
  * 
  * @retval PAPI_ENOEVNT 
  * -- The EventSet is not started yet.
  * @retval PAPI_ENOMEM 
  * -- Insufficient memory to complete the operation. 
  *
  * PAPI_hl_stop stops a running high-level event set.
  * 
  * This call is optional and only necessary if the programmer wants to use the low-level API in addition
  * to the high-level API. It should be noted that PAPI_hl_stop and low-level calls are not
  * allowed inside of a marked region. Furthermore, PAPI_hl_stop is thread-local and therefore
  * has to be called in the same thread as the corresponding marked region.
  *
 * @see PAPI_hl_region_begin
 * @see PAPI_hl_read
 * @see PAPI_hl_region_end
 */
int
PAPI_hl_stop()
{
   int retval, i;

   if ( _papi_hl_events_running == 1 ) {
      if ( _local_components != NULL ) {
         for ( i = 0; i < num_of_components; i++ ) {
            if ( ( retval = PAPI_stop( _local_components[i].EventSet, _local_components[i].values ) ) != PAPI_OK )
               return ( retval );
         }
      }
      _papi_hl_events_running = 0;
      return ( PAPI_OK );
   }
   return ( PAPI_ENOEVNT );
}

