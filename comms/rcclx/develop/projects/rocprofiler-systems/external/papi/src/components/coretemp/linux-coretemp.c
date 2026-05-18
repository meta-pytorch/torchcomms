#include <string.h>

/* Headers required by PAPI */
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"

#include "linux-coretemp.h"

/* this is what I found on my core2 machine 
 * but I have not explored this widely yet*/
#define REFRESH_LAT 4000

#define INVALID_RESULT -1000000L

papi_vector_t _coretemp_vector;

/* temporary event */
struct temp_event {
  char name[PAPI_MAX_STR_LEN];
  char units[PAPI_MIN_STR_LEN];
  char description[PAPI_MAX_STR_LEN];
  char location[PAPI_MAX_STR_LEN];
  char path[PATH_MAX];
  int stone;
  long count;
  struct temp_event *next;
};

// The following macro follows if a string function has an error. It should 
// never happen; but it is necessary to prevent compiler warnings. We print 
// something just in case there is programmer error in invoking the function.
#define HANDLE_STRING_ERROR {fprintf(stderr,"%s:%i unexpected string function error.\n",__FILE__,__LINE__); exit(-1);}

static CORETEMP_native_event_entry_t * _coretemp_native_events;
static int num_events		= 0;
static int is_initialized	= 0;

/***************************************************************************/
/******  BEGIN FUNCTIONS  USED INTERNALLY SPECIFIC TO THIS COMPONENT *******/
/***************************************************************************/

static struct temp_event* root = NULL;
static struct temp_event *last = NULL;

static int
insert_in_list(char *name, char *units,
	       char *description, char *filename) {


    struct temp_event *temp;


    /* new_event   path, events->d_name */
    temp = (struct temp_event *) papi_calloc(1, sizeof(struct temp_event));
    if (temp==NULL) {
       PAPIERROR("out of memory!");
       /* We should also free any previously allocated data */
       return PAPI_ENOMEM;
    }

    temp->next = NULL;

    if (root == NULL) {
       root = temp;
    }
    else if (last) {
       last->next = temp;
    }
    else {
		   /* Because this is a function, it is possible */
		   /* we are called with root!=NULL but no last  */
		   /* so add this to keep coverity happy         */
		   free(temp);
		   PAPIERROR("This shouldn't be possible\n");

		   return PAPI_ECMP;
    }

    last = temp;

    snprintf(temp->name, PAPI_MAX_STR_LEN, "%s", name);
    snprintf(temp->units, PAPI_MIN_STR_LEN, "%s", units);
    snprintf(temp->description, PAPI_MAX_STR_LEN, "%s", description);
    snprintf(temp->path, PATH_MAX, "%s", filename);

    return PAPI_OK;
}

/*
 * find all coretemp information reported by the kernel
 */
static int 
generateEventList(char *base_dir)
{
    char path[PATH_MAX],filename[PATH_MAX];
    char modulename[PAPI_MIN_STR_LEN],
         location[PAPI_MIN_STR_LEN],
         units[PAPI_MIN_STR_LEN],
         description[PAPI_MAX_STR_LEN],
         name[PAPI_MAX_STR_LEN];
    DIR *dir,*d;
    FILE *fff;
    int count = 0;
    struct dirent *hwmonx;
    int i,pathnum;
    int retlen;

#define NUM_PATHS 2
    char paths[NUM_PATHS][PATH_MAX]={
      "device","."
    };

    /* Open "/sys/class/hwmon" */
    dir = opendir(base_dir);
    if ( dir == NULL ) {
       SUBDBG("Can't find %s, are you sure the coretemp module is loaded?\n", 
	       base_dir);
       return 0;
    }

    /* Iterate each /sys/class/hwmonX/device directory */
    while( (hwmonx = readdir(dir) ) ) {
       if ( !strncmp("hwmon", hwmonx->d_name, 5) ) {

	 /* Found a hwmon directory */

	 /* Sometimes the files are in ./, sometimes in device/ */
	 for(pathnum=0;pathnum<NUM_PATHS;pathnum++) {

	    retlen = snprintf(path, PATH_MAX, "%s/%s/%s",
		     base_dir, hwmonx->d_name,paths[pathnum]);
        if (retlen <= 0 || PATH_MAX <= retlen) {
            SUBDBG("Path length is too long.\n");
            return PAPI_EINVAL;
        }
	    SUBDBG("Trying to open %s\n",path);
	    d = opendir(path);
	    if (d==NULL) {
	       continue;
	    }

	    /* Get the name of the module */

	    retlen = snprintf(filename, PAPI_MAX_STR_LEN, "%s/name",path);
        if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen) {
            SUBDBG("Module name too long.\n");
            return PAPI_EINVAL;
        }
	    fff=fopen(filename,"r");
	    if (fff==NULL) {
	       snprintf(modulename, PAPI_MIN_STR_LEN, "Unknown");
	    } else {
	       if (fgets(modulename,PAPI_MIN_STR_LEN,fff)!=NULL) {
	          modulename[strlen(modulename)-1]='\0';
	       }
	       fclose(fff);
	    }

	    SUBDBG("Found module %s\n",modulename);

	  /******************************************************/
	  /* Try handling all events starting with in (voltage) */
	  /******************************************************/


	    /* arbitrary maximum */
	    /* the problem is the numbering can be sparse */
	    /* should probably go back to dirent listing  */
	    
	  for(i=0;i<32;i++) {

	     /* Try looking for a location label */
	     retlen = snprintf(filename, PAPI_MAX_STR_LEN, "%s/in%d_label",
		      path,i);
         if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen) {
            SUBDBG("Failed to construct location label.\n");
            return PAPI_EINVAL;
         }
	     fff=fopen(filename,"r");
	     if (fff==NULL) {
	        strncpy(location,"?",PAPI_MIN_STR_LEN);
	     }
	     else {
	        if (fgets(location,PAPI_MIN_STR_LEN,fff)!=NULL) {
	           location[strlen(location)-1]='\0';
	        }
	        fclose(fff);
	     }

	     /* Look for input temperature */
	     retlen = snprintf(filename, PAPI_MAX_STR_LEN, "%s/in%d_input",
		      path,i);
         if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen) {
            SUBDBG("Failed input temperature string.\n");
            return PAPI_EINVAL;
         }
	     fff=fopen(filename,"r");
	     if (fff==NULL) continue;
	     fclose(fff);

	     retlen = snprintf(name, PAPI_MAX_STR_LEN, "%s:in%i_input", hwmonx->d_name, i);
	     if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen) {
	         SUBDBG("Unable to generate name %s:in%i_input\n", hwmonx->d_name, i);
		 closedir(dir);
		 closedir(d);
	         return ( PAPI_EINVAL );
	     }

	     snprintf(units, PAPI_MIN_STR_LEN, "V");
	     retlen = snprintf(description, PAPI_MAX_STR_LEN, "%s, %s module, label %s",
		      units,modulename,
		      location);
         if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen) {
            SUBDBG("snprintf failed.\n");
            return PAPI_EINVAL;
         }

	     if (insert_in_list(name,units,description,filename)!=PAPI_OK) {
	        goto done_error;
	     }

	     count++;

	  }

	  /************************************************************/
	  /* Try handling all events starting with temp (temperature) */
	  /************************************************************/

	  for(i=0;i<32;i++) {

	     /* Try looking for a location label */
	     retlen = snprintf(filename, PAPI_MAX_STR_LEN, "%s/temp%d_label",
		      path,i);
         if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen) {
            SUBDBG("Location label string failed.\n");
            return PAPI_EINVAL;
         }
	     fff=fopen(filename,"r");
	     if (fff==NULL) {
	        strncpy(location,"?",PAPI_MIN_STR_LEN);
	     }
	     else {
	        if (fgets(location,PAPI_MIN_STR_LEN,fff)!=NULL) {
	           location[strlen(location)-1]='\0';
	        }
	        fclose(fff);
	     }

	     /* Look for input temperature */
	     retlen = snprintf(filename, PAPI_MAX_STR_LEN, "%s/temp%d_input",
		      path,i);
         if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen) {
            SUBDBG("Input temperature string failed.\n");
            return PAPI_EINVAL;
         }
	     fff=fopen(filename,"r");
	     if (fff==NULL) continue;
	     fclose(fff);

	     retlen = snprintf(name, PAPI_MAX_STR_LEN, "%s:temp%i_input", hwmonx->d_name, i);
	     if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen) {
	         SUBDBG("Unable to generate name %s:temp%i_input\n", hwmonx->d_name, i);
            closedir(d);
            closedir(dir);
	         return ( PAPI_EINVAL );
	     }

	     snprintf(units, PAPI_MIN_STR_LEN, "degrees C");
	     retlen = snprintf(description, PAPI_MAX_STR_LEN, "%s, %s module, label %s",
		      units,modulename,
		      location);
         if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen) {
            SUBDBG("snprintf failed.\n");
            return PAPI_EINVAL;
         }

	     if (insert_in_list(name,units,description,filename)!=PAPI_OK) {
	        goto done_error;
	     }

	     count++;
	  }

	  /************************************************************/
	  /* Try handling all events starting with fan (fan)          */
	  /************************************************************/

	  for(i=0;i<32;i++) {

	     /* Try looking for a location label */
	     retlen = snprintf(filename, PAPI_MAX_STR_LEN, "%s/fan%d_label",
		      path,i);
         if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen) {
            SUBDBG("Failed to write fan label string.\n");
            return PAPI_EINVAL;
         }
	     fff=fopen(filename,"r");
	     if (fff==NULL) {
	        strncpy(location,"?",PAPI_MIN_STR_LEN);
	     }
	     else {
	        if (fgets(location,PAPI_MIN_STR_LEN,fff)!=NULL) {
	           location[strlen(location)-1]='\0';
	        }
	        fclose(fff);
	     }

	     /* Look for input fan */
	     retlen = snprintf(filename, PAPI_MAX_STR_LEN, "%s/fan%d_input", path,i);
	     if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen) {
	         SUBDBG("Unable to generate filename %s/fan%d_input\n", path,i);
            closedir(d);
            closedir(dir);
	         return ( PAPI_EINVAL );
	     }

	     fff=fopen(filename,"r");
	     if (fff==NULL) continue;
	     fclose(fff);

	     retlen = snprintf(name, PAPI_MAX_STR_LEN, "%s:fan%i_input", hwmonx->d_name, i);
	     if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen) {
	         SUBDBG("Unable to generate name %s:fan%i_input\n", hwmonx->d_name, i);
            closedir(d);
            closedir(dir);
	         return ( PAPI_EINVAL );
	     }

	     snprintf(units, PAPI_MIN_STR_LEN, "RPM");
	     retlen = snprintf(description, PAPI_MAX_STR_LEN, "%s, %s module, label %s",
		      units,modulename,
		      location);
         if (retlen <= 0 || PAPI_MAX_STR_LEN <= retlen) {
            SUBDBG("snprintf failed.\n");
            return PAPI_EINVAL;
         }

	     if (insert_in_list(name,units,description,filename)!=PAPI_OK) {
	        goto done_error;
	     }

	     count++;

	  }
	  closedir(d);
	 }
       }
    }

    closedir(dir);
    return count;

done_error:
    closedir(d);
    closedir(dir);
    return PAPI_ECMP;
}

static long long
getEventValue( int index ) 
{
    char buf[PAPI_MAX_STR_LEN];
    FILE* fp;
    long result;

    if (_coretemp_native_events[index].stone) {
       return _coretemp_native_events[index].value;
    }

    fp = fopen(_coretemp_native_events[index].path, "r");
    if (fp==NULL) {
       return INVALID_RESULT;
    }

    if (fgets(buf, PAPI_MAX_STR_LEN, fp)==NULL) {
        result=INVALID_RESULT;
    }
    else {
        result=strtoll(buf, NULL, 10);
    }
    fclose(fp);

    return result;
}

/*****************************************************************************
 *******************  BEGIN PAPI's COMPONENT REQUIRED FUNCTIONS  *************
 *****************************************************************************/

/*
 * This is called whenever a thread is initialized
 */
static int
_coretemp_init_thread( hwd_context_t *ctx )
{
  ( void ) ctx;
  return PAPI_OK;
}



/* Initialize hardware counters, setup the function vector table
 * and get hardware information, this routine is called when the 
 * PAPI process is initialized (IE PAPI_library_init)
 */
static int
_coretemp_init_component( int cidx )
{
    int retval = PAPI_OK;
     int i = 0;
     struct temp_event *t,*last;

     if ( is_initialized )
         goto fn_exit;

     is_initialized = 1;

     /* This is the prefered method, all coretemp sensors are symlinked here
      * see $(kernel_src)/Documentation/hwmon/sysfs-interface */
  
     num_events = generateEventList("/sys/class/hwmon");

     if ( num_events < 0 ) {
        char* strCpy;
        strCpy=strncpy(_coretemp_vector.cmp_info.disabled_reason,
        "Cannot open /sys/class/hwmon",PAPI_MAX_STR_LEN);
        _coretemp_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strCpy == NULL) HANDLE_STRING_ERROR;
        retval = PAPI_ECMP;
        goto fn_fail;
     }

     if ( num_events == 0 ) {
        char* strCpy=strncpy(_coretemp_vector.cmp_info.disabled_reason,
        "No coretemp events found",PAPI_MAX_STR_LEN);
        _coretemp_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
        if (strCpy == NULL) HANDLE_STRING_ERROR;
        retval = PAPI_ECMP;
        goto fn_fail;
     }

     t = root;
  
     _coretemp_native_events = (CORETEMP_native_event_entry_t*)
          papi_calloc(num_events, sizeof(CORETEMP_native_event_entry_t));
     if (_coretemp_native_events == NULL) {
          int strErr=snprintf(_coretemp_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "malloc() of _coretemp_native_events failed for %lu bytes.", num_events*sizeof(CORETEMP_native_event_entry_t));
          _coretemp_vector.cmp_info.disabled_reason[PAPI_MAX_STR_LEN-1]=0;
          if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
          retval = PAPI_ENOMEM;
          goto fn_fail;
     }

     do {
        int retlen;
        retlen = snprintf(_coretemp_native_events[i].name, PAPI_MAX_STR_LEN, "%s", t->name);
        if (retlen <= 0 || retlen >= PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;

        retlen = snprintf(_coretemp_native_events[i].path, PATH_MAX, "%s", t->path);
        if (retlen <= 0 || retlen >= PATH_MAX) HANDLE_STRING_ERROR;

        retlen = snprintf(_coretemp_native_events[i].units, PAPI_MIN_STR_LEN, "%s", t->units);
        if (retlen <= 0 || retlen >= PAPI_MIN_STR_LEN) HANDLE_STRING_ERROR;

        retlen = snprintf(_coretemp_native_events[i].description, PAPI_MAX_STR_LEN, "%s",t->description);
        if (retlen <= 0 || retlen >= PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;

	    _coretemp_native_events[i].stone = 0;
	    _coretemp_native_events[i].resources.selector = i + 1;
	    last	= t;
	    t		= t->next;
	    papi_free(last);
	    i++;
     } while (t != NULL);
     root = NULL;

     /* Export the total number of events available */
     _coretemp_vector.cmp_info.num_native_events = num_events;

     /* Export the component id */
     _coretemp_vector.cmp_info.CmpIdx = cidx;

  fn_exit:
     _papi_hwd[cidx]->cmp_info.disabled = retval;
     return retval;
  fn_fail:
     goto fn_exit;
}




/*
 * Control of counters (Reading/Writing/Starting/Stopping/Setup)
 * functions
 */
static int
_coretemp_init_control_state( hwd_control_state_t * ctl)
{
    int i;

    CORETEMP_control_state_t *coretemp_ctl = (CORETEMP_control_state_t *) ctl;

    for ( i=0; i < num_events; i++ ) {
	coretemp_ctl->counts[i] = getEventValue(i);
    }

    /* Set last access time for caching results */
    coretemp_ctl->lastupdate = PAPI_get_real_usec();

    return PAPI_OK;
}

static int
_coretemp_start( hwd_context_t *ctx, hwd_control_state_t *ctl)
{
  ( void ) ctx;
  ( void ) ctl;

  return PAPI_OK;
}

static int
_coretemp_read( hwd_context_t *ctx, hwd_control_state_t *ctl,
	        long long ** events, int flags)
{
    (void) flags;
    (void) ctx;

    CORETEMP_control_state_t* control = (CORETEMP_control_state_t*) ctl;
    long long now = PAPI_get_real_usec();
    int i;

    /* Only read the values from the kernel if enough time has passed */
    /* since the last read.  Otherwise return cached values.          */

    if ( now - control->lastupdate > REFRESH_LAT ) {
	for ( i = 0; i < num_events; i++ ) {
	   control->counts[i] = getEventValue( i );
	}
	control->lastupdate = now;
    }

    /* Pass back a pointer to our results */
    *events = control->counts;

    return PAPI_OK;
}

static int
_coretemp_stop( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    (void) ctx;
    /* read values */
    CORETEMP_control_state_t* control = (CORETEMP_control_state_t*) ctl;
    int i;

    for ( i = 0; i < num_events; i++ ) {
	control->counts[i] = getEventValue( i );
    }

    return PAPI_OK;
}

/* Shutdown a thread */
static int
_coretemp_shutdown_thread( hwd_context_t * ctx )
{
  ( void ) ctx;
  return PAPI_OK;
}


/*
 * Clean up what was setup in  coretemp_init_component().
 */
static int
_coretemp_shutdown_component( ) 
{
    if ( is_initialized ) {
       is_initialized = 0;
       papi_free(_coretemp_native_events);
       _coretemp_native_events = NULL;
    }
    return PAPI_OK;
}


/* This function sets various options in the component
 * The valid codes being passed in are PAPI_SET_DEFDOM,
 * PAPI_SET_DOMAIN, PAPI_SETDEFGRN, PAPI_SET_GRANUL * and PAPI_SET_INHERIT
 */
static int
_coretemp_ctl( hwd_context_t *ctx, int code, _papi_int_option_t *option )
{
    ( void ) ctx;
    ( void ) code;
    ( void ) option;

    return PAPI_OK;
}


static int
_coretemp_update_control_state(	hwd_control_state_t *ptr,
				NativeInfo_t * native, int count,
				hwd_context_t * ctx )
{
    int i, index;
    ( void ) ctx;
    ( void ) ptr;

    for ( i = 0; i < count; i++ ) {
	index = native[i].ni_event;
	native[i].ni_position = _coretemp_native_events[index].resources.selector - 1;
    }
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
_coretemp_set_domain( hwd_control_state_t * cntl, int domain )
{
       (void) cntl;
       if ( PAPI_DOM_ALL != domain )
		return PAPI_EINVAL;

       return PAPI_OK;
}


static int
_coretemp_reset( hwd_context_t *ctx, hwd_control_state_t *ctl )
{
    ( void ) ctx;
    ( void ) ctl;
	
    return PAPI_OK;
}


/*
 * Native Event functions
 */
static int
_coretemp_ntv_enum_events( unsigned int *EventCode, int modifier )
{

     int index;

     switch ( modifier ) {

	case PAPI_ENUM_FIRST:

	   if (num_events==0) {
	      return PAPI_ENOEVNT;
	   }
	   *EventCode = 0;

	   return PAPI_OK;
		

	case PAPI_ENUM_EVENTS:
	
	   index = *EventCode;

	   if ( index < num_events - 1 ) {
	      *EventCode = *EventCode + 1;
	      return PAPI_OK;
	   } else {
	      return PAPI_ENOEVNT;
	   }
	   break;
	
	default:
		return PAPI_EINVAL;
	}
	return PAPI_EINVAL;
}

/*
 *
 */
static int
_coretemp_ntv_code_to_name( unsigned int EventCode, char *name, int len )
{
     int index = EventCode;

     if ( index >= 0 && index < num_events ) {
	strncpy( name, _coretemp_native_events[index].name, len );
	return PAPI_OK;
     }
     return PAPI_ENOEVNT;
}

/*
 *
 */
static int
_coretemp_ntv_code_to_descr( unsigned int EventCode, char *name, int len )
{
     int index = EventCode;

     if ( index >= 0 && index < num_events ) {
	strncpy( name, _coretemp_native_events[index].description, len );
	return PAPI_OK;
     }
     return PAPI_ENOEVNT;
}

static int
_coretemp_ntv_code_to_info(unsigned int EventCode, PAPI_event_info_t *info) 
{

  int index = EventCode;

  if ( ( index < 0) || (index >= num_events )) return PAPI_ENOEVNT;

  strncpy( info->symbol, _coretemp_native_events[index].name, sizeof(info->symbol));
  strncpy( info->long_descr, _coretemp_native_events[index].description, sizeof(info->long_descr));
  strncpy( info->units, _coretemp_native_events[index].units, sizeof(info->units));
  info->units[sizeof(info->units)-1] = '\0';

  return PAPI_OK;
}



/*
 *
 */
papi_vector_t _coretemp_vector = {
	.cmp_info = {
				 /* default component information (unspecified values are initialized to 0) */
				 .name = "coretemp",
				 .short_name = "coretemp",
				 .description = "Linux hwmon temperature and other info",
				 .version = "4.2.1",
				 .num_mpx_cntrs = CORETEMP_MAX_COUNTERS,
				 .num_cntrs = CORETEMP_MAX_COUNTERS,
				 .default_domain = PAPI_DOM_ALL,
				 .available_domains = PAPI_DOM_ALL,
				 .default_granularity = PAPI_GRN_SYS,
				 .available_granularities = PAPI_GRN_SYS,
				 .hardware_intr_sig = PAPI_INT_SIGNAL,

				 /* component specific cmp_info initializations */
				 .fast_real_timer = 0,
				 .fast_virtual_timer = 0,
				 .attach = 0,
				 .attach_must_ptrace = 0,
				 }
	,

	/* sizes of framework-opaque component-private structures */
	.size = {
			 .context = sizeof ( CORETEMP_context_t ),
			 .control_state = sizeof ( CORETEMP_control_state_t ),
			 .reg_value = sizeof ( CORETEMP_register_t ),
			 .reg_alloc = sizeof ( CORETEMP_reg_alloc_t ),
			 }
	,
	/* function pointers in this component */
	.init_thread =          _coretemp_init_thread,
	.init_component =       _coretemp_init_component,
	.init_control_state =   _coretemp_init_control_state,
	.start =                _coretemp_start,
	.stop =                 _coretemp_stop,
	.read =                 _coretemp_read,
	.shutdown_thread =      _coretemp_shutdown_thread,
	.shutdown_component =   _coretemp_shutdown_component,
	.ctl =                  _coretemp_ctl,

	.update_control_state = _coretemp_update_control_state,
	.set_domain =           _coretemp_set_domain,
	.reset =                _coretemp_reset,

	.ntv_enum_events =      _coretemp_ntv_enum_events,
	.ntv_code_to_name =     _coretemp_ntv_code_to_name,
	.ntv_code_to_descr =    _coretemp_ntv_code_to_descr,
	.ntv_code_to_info =     _coretemp_ntv_code_to_info,
};
