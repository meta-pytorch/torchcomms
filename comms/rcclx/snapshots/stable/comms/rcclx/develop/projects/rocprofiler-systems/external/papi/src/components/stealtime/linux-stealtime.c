/**
* @file    linux-stealtime.c
* @author  Vince Weaver
*          vweaver1@eecs.utk.edu
* @brief A component that gather info on VM stealtime
*/

#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <dirent.h>
#include <stdint.h>
#include <ctype.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

struct counter_info
{
	char *name;
	char *description;
        char *units;
	unsigned long long value;
};

typedef struct counter_info STEALTIME_register_t;
typedef struct counter_info STEALTIME_native_event_entry_t;
typedef struct counter_info STEALTIME_reg_alloc_t;


struct STEALTIME_control_state
{
  long long *values;
  int *which_counter;
  int num_events;
};


struct STEALTIME_context
{
  long long *start_count;
  long long *current_count;
  long long *value;
};


static int num_events = 0;

static struct counter_info *event_info=NULL;

/* Advance declaration of buffer */
papi_vector_t _stealtime_vector;

/******************************************************************************
 ********  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT ********
 *****************************************************************************/

struct statinfo {
  long long user;
  long long nice;
  long long system;
  long long idle;
  long long iowait;
  long long irq;
  long long softirq;
  long long steal;
  long long guest;
};

static int
read_stealtime( struct STEALTIME_context *context, int starting) {

  FILE *fff;
  char buffer[BUFSIZ],*result;
  int i,count;
  struct statinfo our_stat;

  int hz=sysconf(_SC_CLK_TCK);


  fff=fopen("/proc/stat","r");
  if (fff==NULL) {
     return PAPI_ESYS; 
  }

  for(i=0;i<num_events;i++) {
    result=fgets(buffer,BUFSIZ,fff);
    if (result==NULL) break;

    count=sscanf(buffer,"%*s %lld %lld %lld %lld %lld %lld %lld %lld %lld",
		 &our_stat.user,
		 &our_stat.nice,
		 &our_stat.system,
		 &our_stat.idle,
		 &our_stat.iowait,
		 &our_stat.irq,
		 &our_stat.softirq,
		 &our_stat.steal,
		 &our_stat.guest);
    if (count<=7) {
       fclose(fff);
       return PAPI_ESYS;
    }

    if (starting) {
       context->start_count[i]=our_stat.steal;
    }
    context->current_count[i]=our_stat.steal;

    /* convert to us */
    context->value[i]=(context->current_count[i]-context->start_count[i])*
      (1000000/hz);
  }
  

  fclose(fff);

  return PAPI_OK;

}



/*****************************************************************************
 *******************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *************
 *****************************************************************************/

/*
 * Component setup and shutdown
 */
static int _stealtime_shutdown_component( void );  // prototype for later routine.

static int
_stealtime_init_component( int cidx )
{
    int retval = PAPI_OK;
  (void)cidx;

	FILE *fff;
	char buffer[BUFSIZ],*result,string[BUFSIZ];
	int i;

	/* Make sure /proc/stat exists */
	fff=fopen("/proc/stat","r");
	if (fff==NULL) {
	   strncpy(_stealtime_vector.cmp_info.disabled_reason,
		   "Cannot open /proc/stat",PAPI_MAX_STR_LEN);
       _stealtime_shutdown_component();
       retval = PAPI_ESYS;
       goto fn_fail;
	}

	num_events=0;
	while(1) {
	  result=fgets(buffer,BUFSIZ,fff);
	  if (result==NULL) break;

	  /* /proc/stat line with cpu stats always starts with "cpu" */

	  if (!strncmp(buffer,"cpu",3)) {
	     num_events++;
	  }
	  else {
	    break;
	  }

	}

	fclose(fff);

	if (num_events<1) {
	   strncpy(_stealtime_vector.cmp_info.disabled_reason,
		   "Cannot find enough CPU lines in /proc/stat",
		   PAPI_MAX_STR_LEN);
       _stealtime_shutdown_component();
       retval = PAPI_ESYS;
       goto fn_fail;
	}

	event_info=calloc(num_events,sizeof(struct counter_info));
	if (event_info==NULL) {
        _stealtime_shutdown_component();
        retval = PAPI_ENOMEM;
        goto fn_fail;
	}

	
	sysconf(_SC_CLK_TCK);
	event_info[0].name=strdup("TOTAL");
	event_info[0].description=strdup("Total amount of steal time");
	event_info[0].units=strdup("us");

	for(i=1;i<num_events;i++) {
	   sprintf(string,"CPU%d",i);
	   event_info[i].name=strdup(string);
	   sprintf(string,"Steal time for CPU %d",i);
	   event_info[i].description=strdup(string);
	   event_info[i].units=strdup("us");
        }

	//	printf("Found %d CPUs\n",num_events-1);

	_stealtime_vector.cmp_info.num_native_events=num_events;
	_stealtime_vector.cmp_info.num_cntrs=num_events;
	_stealtime_vector.cmp_info.num_mpx_cntrs=num_events;

  fn_exit:
    _papi_hwd[cidx]->cmp_info.disabled = retval;
    return retval;
  fn_fail:
    goto fn_exit;
}





/*
 * This is called whenever a thread is initialized
 */
static int
_stealtime_init_thread( hwd_context_t * ctx )
{
  struct STEALTIME_context *context=(struct STEALTIME_context *)ctx;

  context->start_count=calloc(num_events,sizeof(long long));
  if (context->start_count==NULL) return PAPI_ENOMEM;

  context->current_count=calloc(num_events,sizeof(long long));
  if (context->current_count==NULL) return PAPI_ENOMEM;

  context->value=calloc(num_events,sizeof(long long));
  if (context->value==NULL) return PAPI_ENOMEM;

  return PAPI_OK;
}


/*
 *
 */
static int
_stealtime_shutdown_component( void )
{
       int i;
       int num_events = _stealtime_vector.cmp_info.num_native_events;
       if (event_info!=NULL) {
               for (i=0; i<num_events; i++){
                       free(event_info[i].name);
                       free(event_info[i].description);
                       free(event_info[i].units);
               }
               free(event_info);
       }

   return PAPI_OK;
}

/*
 *
 */
static int
_stealtime_shutdown_thread( hwd_context_t * ctx )
{

  struct STEALTIME_context *context=(struct STEALTIME_context *)ctx;

  if (context->start_count!=NULL) free(context->start_count);
  if (context->current_count!=NULL) free(context->current_count);
  if (context->value!=NULL) free(context->value);

  return PAPI_OK;
}



/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup) functions
 */
static int
_stealtime_init_control_state( hwd_control_state_t *ctl )
{

    struct STEALTIME_control_state *control = 
      (struct STEALTIME_control_state *)ctl;

    control->values=NULL;
    control->which_counter=NULL;
    control->num_events=0;

    return PAPI_OK;
}


/*
 *
 */
static int
_stealtime_update_control_state( hwd_control_state_t *ctl, 
			      NativeInfo_t *native,
			      int count, 
			      hwd_context_t *ctx )
{

    struct STEALTIME_control_state *control;

    ( void ) ctx;
    int i, index;

    control= (struct STEALTIME_control_state *)ctl;

    if (count!=control->num_events) {
      //       printf("Resizing %d to %d\n",control->num_events,count);
       control->which_counter=realloc(control->which_counter,
				      count*sizeof(int));
       control->values=realloc(control->values,
			       count*sizeof(long long));
       
    }


    for ( i = 0; i < count; i++ ) {
       index = native[i].ni_event;
       control->which_counter[i]=index;
       native[i].ni_position = i;
    }

    control->num_events=count;

    return PAPI_OK;
}


/*
 *
 */
static int
_stealtime_start( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

  (void)ctl;

  //    struct STEALTIME_control_state *control;
    struct STEALTIME_context *context;
    
    //control = (struct STEALTIME_control_state *)ctl;
    context = (struct STEALTIME_context *)ctx;

    read_stealtime( context, 1 );

    /* no need to update control, as we assume only one EventSet  */
    /* is active at once, so starting things at the context level */
    /* is fine, since stealtime is system-wide                    */

    return PAPI_OK;
}


/*
 *
 */
static int
_stealtime_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{

  (void) ctl;

  //    struct STEALTIME_control_state *control;
    struct STEALTIME_context *context;
    
    //control = (struct STEALTIME_control_state *)ctl;
    context = (struct STEALTIME_context *)ctx;

    read_stealtime( context, 0 );

    return PAPI_OK;

}



/*
 *
 */
static int
_stealtime_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
			 long long **events, int flags )
{
    ( void ) flags;

    struct STEALTIME_control_state *control;
    struct STEALTIME_context *context;

    int i;
    
    control = (struct STEALTIME_control_state *)ctl;
    context = (struct STEALTIME_context *)ctx;

    read_stealtime( context, 0 );

    for(i=0;i<control->num_events;i++) {
       control->values[i]=
                 context->value[control->which_counter[i]];
    }

    *events = control->values;

    return PAPI_OK;

}




/*
 *
 */
static int
_stealtime_reset( hwd_context_t * ctx, hwd_control_state_t * ctrl )
{

  /* re-initializes counter_start values to current */

  _stealtime_start(ctx,ctrl);

  return PAPI_OK;
}


/*
 * Unused stealtime write function
 */
/* static int */
/* _stealtime_write( hwd_context_t * ctx, hwd_control_state_t * ctrl, long long *from ) */
/* { */
/* 	( void ) ctx; */
/* 	( void ) ctrl; */
/* 	( void ) from; */

/* 	return PAPI_OK; */
/* } */


/*
 * Functions for setting up various options
 */

/* This function sets various options in the component
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
static int
_stealtime_ctl( hwd_context_t * ctx, int code, _papi_int_option_t * option )
{
	( void ) ctx;
	( void ) code;
	( void ) option;

	return PAPI_OK;
}


/*
 * This function has to set the bits needed to count different domains
 * In particular: PAPI_DOM_USER, PAPI_DOM_KERNEL PAPI_DOM_OTHER
 * By default return PAPI_EINVAL if none of those are specified
 * and PAPI_OK with success
 * PAPI_DOM_USER is only user context is counted
 * PAPI_DOM_KERNEL is only the Kernel/OS context is counted
 * PAPI_DOM_OTHER  is Exception/transient mode (like user TLB misses)
 * PAPI_DOM_ALL   is all of the domains
 */
static int
_stealtime_set_domain( hwd_control_state_t * cntrl, int domain )
{
	( void ) cntrl;
	int found = 0;
	if ( PAPI_DOM_USER & domain ) {
		found = 1;
	}
	if ( PAPI_DOM_KERNEL & domain ) {
		found = 1;
	}
	if ( PAPI_DOM_OTHER & domain ) {
		found = 1;
	}
	if ( !found )
		return PAPI_EINVAL;

	return PAPI_OK;
}


/*
 *
 */
static int
_stealtime_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{

  int event=EventCode;

  if (event >=0 && event < num_events) {
     strncpy( name, event_info[event].name, len );
     return PAPI_OK;
  }

  return PAPI_ENOEVNT;
}


/*
 *
 */
static int
_stealtime_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{

  int event=EventCode;

  if (event >=0 && event < num_events) {
	strncpy( name, event_info[event].description, len );
	return PAPI_OK;
  }

  return PAPI_ENOEVNT;
}



static int
_stealtime_ntv_code_to_info(unsigned int EventCode, PAPI_event_info_t *info)
{

  int index = EventCode;

  if ( ( index < 0) || (index >= num_events )) return PAPI_ENOEVNT;

  strncpy( info->symbol, event_info[index].name,sizeof(info->symbol));
  info->symbol[sizeof(info->symbol)-1] = '\0';

  strncpy( info->long_descr, event_info[index].description,sizeof(info->symbol));
  info->long_descr[sizeof(info->symbol)-1] = '\0';

  strncpy( info->units, event_info[index].units,sizeof(info->units));
  info->units[sizeof(info->units)-1] = '\0';

  return PAPI_OK;

}




/*
 *
 */
static int
_stealtime_ntv_enum_events( unsigned int *EventCode, int modifier )
{

     if ( modifier == PAPI_ENUM_FIRST ) {
	if (num_events==0) return PAPI_ENOEVNT;
	*EventCode = 0;
	return PAPI_OK;
     }

     if ( modifier == PAPI_ENUM_EVENTS ) {
        int index;

        index = *EventCode;

	if ( (index+1) < num_events ) {
	   *EventCode = *EventCode + 1;
	   return PAPI_OK;
	} else {
	   return PAPI_ENOEVNT;
	}
     } 
		
     return PAPI_EINVAL;
}


/*
 *
 */
papi_vector_t _stealtime_vector = {
   .cmp_info = {
        /* component information (unspecified values initialized to 0) */
       .name = "stealtime",
	   .short_name="stealtime",
       .version = "5.0",
       .description = "Stealtime filesystem statistics",
       .default_domain = PAPI_DOM_USER,
       .default_granularity = PAPI_GRN_THR,
       .available_granularities = PAPI_GRN_THR,
       .hardware_intr_sig = PAPI_INT_SIGNAL,

       /* component specific cmp_info initializations */
       .fast_real_timer = 0,
       .fast_virtual_timer = 0,
       .attach = 0,
       .attach_must_ptrace = 0,
       .available_domains = PAPI_DOM_USER | PAPI_DOM_KERNEL,
  },

     /* sizes of framework-opaque component-private structures */
  .size = {
       .context = sizeof ( struct STEALTIME_context ),
       .control_state = sizeof ( struct STEALTIME_control_state ),
       .reg_value = sizeof ( STEALTIME_register_t ),
       .reg_alloc = sizeof ( STEALTIME_reg_alloc_t ),
  },

     /* function pointers in this component */
  .init_thread =           _stealtime_init_thread,
  .init_component =        _stealtime_init_component,
  .init_control_state =    _stealtime_init_control_state,
  .start =                 _stealtime_start,
  .stop =                  _stealtime_stop,
  .read =                  _stealtime_read,
  .shutdown_thread =       _stealtime_shutdown_thread,
  .shutdown_component =    _stealtime_shutdown_component,
  .ctl =                   _stealtime_ctl,
  .update_control_state =  _stealtime_update_control_state,
  .set_domain =            _stealtime_set_domain,
  .reset =                 _stealtime_reset,

  .ntv_enum_events =   _stealtime_ntv_enum_events,
  .ntv_code_to_name =  _stealtime_ntv_code_to_name,
  .ntv_code_to_descr = _stealtime_ntv_code_to_descr,
  .ntv_code_to_info = _stealtime_ntv_code_to_info,
};




