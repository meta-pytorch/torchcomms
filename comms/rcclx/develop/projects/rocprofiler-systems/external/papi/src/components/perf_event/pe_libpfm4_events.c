/*
* File:    pe_libpfm4_events.c
* Author:  Vince Weaver vincent.weaver@maine.edu
* Mods:    Gary Mohr
*          gary.mohr@bull.com
*          Modified the perf_event component to use PFM_OS_PERF_EVENT_EXT mode in libpfm4.
*          This adds several new event masks, including cpu=, u=, and k= which give the user
*          the ability to set cpu number to use or control the domain (user, kernel, or both)
*          in which the counter should be incremented.  These are event masks so it is now 
*          possible to have multiple events in the same event set that count activity from 
*          differennt cpu's or count activity in different domains.
*
* Handle the libpfm4 event interface for the perf_event component
*/

#include <string.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"

#include "papi_libpfm4_events.h"
#include "pe_libpfm4_events.h"
#include "perf_event_lib.h"

#include "perfmon/pfmlib.h"
#include "perfmon/pfmlib_perf_event.h"

#define NATIVE_EVENT_CHUNK 1024

// used to step through the attributes when enumerating events
static int attr_idx;

/* alias flags to handle amd_fam17h, amd_fam17h_zen1 both present PMUs*/
static int amd64_fam17h_zen1_present = 0;

/** @class  find_existing_event
 *  @brief  looks up an event, returns it if it exists
 *
 *  @param[in] name
 *             -- name of the event
 *  @param[in] event_table
 *             -- native_event_table structure
 *
 *  @returns returns offset in array
 *
 */

static int find_existing_event(const char *name,
                               struct native_event_table_t *event_table) {
  SUBDBG("Entry: name: %s, event_table: %p, num_native_events: %d\n", name, event_table, event_table->num_native_events);

  int i,event=PAPI_ENOEVNT;

  _papi_hwi_lock( NAMELIB_LOCK );

  for(i=0;i<event_table->num_native_events;i++) {
    // Most names passed in will contain the pmu name, so first we compare to the allocated name (it has pmu name on front)
    if (!strcmp(name,event_table->native_events[i].allocated_name)) {
      SUBDBG("Found allocated_name: %s, libpfm4_idx: %#x, papi_event_code: %#x\n", 
         event_table->native_events[i].allocated_name, event_table->native_events[i].libpfm4_idx, event_table->native_events[i].papi_event_code);
      event=i;
      break;
    }
    // some callers have an event name without the pmu name on the front, so we also compare to the base name (just the event name part)
    if (!strcmp(name,event_table->native_events[i].base_name)) {
      int nameLen = strlen(event_table->native_events[i].base_name);
      // the name we are looking for must be the same length as this event table entry name for them to match
      if (strlen(name) != nameLen + strlen(event_table->native_events[i].mask_string) + 1) {
        continue;
      }
      if(!strcmp(name+nameLen+1, event_table->native_events[i].mask_string)) {
          SUBDBG("Found base_name: %s, mask_string: %s, libpfm4_idx: %#x, papi_event_code: %#x\n",
            event_table->native_events[i].base_name, event_table->native_events[i].mask_string , event_table->native_events[i].libpfm4_idx, event_table->native_events[i].papi_event_code);
          event=i;
          break;
      }
    }
  }
  _papi_hwi_unlock( NAMELIB_LOCK );

  SUBDBG("EXIT: returned: %#x\n", event);
  return event;
}


static int pmu_is_present_and_right_type(pfm_pmu_info_t *pinfo, int type) {
	SUBDBG("ENTER: pinfo: %s %p, pinfo->is_present: %d, "
		"pinfo->type: %#x, type: %#x\n",
		pinfo->name, pinfo, pinfo->is_present, pinfo->type, type);
	if (!pinfo->is_present) {
//		SUBDBG("EXIT: not present\n");
		return 0;
	}

	if ((pinfo->type==PFM_PMU_TYPE_UNCORE) && (type&PMU_TYPE_UNCORE)) {
//		SUBDBG("EXIT: found PFM_PMU_TYPE_UNCORE\n");
		return 1;
	}
	if ((pinfo->type==PFM_PMU_TYPE_CORE) && (type&PMU_TYPE_CORE)) {
//		SUBDBG("EXIT: found PFM_PMU_TYPE_CORE\n");
		return 1;
	}
	if ((pinfo->type==PFM_PMU_TYPE_OS_GENERIC) && (type&PMU_TYPE_OS)) {
//		SUBDBG("EXIT: found PFM_PMU_TYPE_OS_GENERIC\n");
		return 1;
	}

//	SUBDBG("EXIT: not right type\n");
	return 0;
}


/** @class  allocate_native_event
 *  @brief  Allocates a native event
 *
 *  @param[in] name
 *             -- name of the event
 *  @param[in] libpfm4_idx
 *             -- libpfm4 identifier for the event
 *  @param[in] cidx
 *             -- PAPI component index
 *  @param[in] event_table
 *             -- native event table struct
 *
 *  @returns returns a native_event_t or NULL
 *
 */

static struct native_event_t *allocate_native_event(
		const char *name,
		int libpfm4_index, int cidx,
		struct native_event_table_t *event_table) {

	SUBDBG("ENTER: name: %s, libpfm4_index: %#x, event_table: %p, "
		"event_table->pmu_type: %d\n",
		name, libpfm4_index, event_table, event_table->pmu_type);

	int nevt_idx;
	int event_num;
	int encode_failed=0;

	pfm_err_t ret;
	char *event_string=NULL;
	char *pmu_name;
	char *event;
	char *masks;
	char fullname[BUFSIZ];
	struct native_event_t *ntv_evt;

	pfm_perf_encode_arg_t perf_arg;
	pfm_event_info_t einfo;
	pfm_event_attr_info_t ainfo;
	pfm_pmu_info_t pinfo;

	// if no place to put native events, report that allocate failed
	if (event_table->native_events==NULL) {
		SUBDBG("EXIT: no place to put native events\n");
		return NULL;
	}

	// find out if this event is already known
	event_num=find_existing_event(name, event_table);

	/* add the event to our event table */
	_papi_hwi_lock( NAMELIB_LOCK );

	// if we already know this event name,
	// it was created as part of setting up the preset tables
	// we need to use the event table which is already created
	if (event_num >= 0) {
		nevt_idx = event_num;
		ntv_evt = &(event_table->native_events[event_num]);
	} else {
		// set to use a new event table
		// (count of used events not bumped
		// until we are sure setting it up does not get an errror)
		nevt_idx = event_table->num_native_events;
		ntv_evt = &(event_table->native_events[nevt_idx]);
	}

	SUBDBG("event_num: %d, nevt_idx: %d, ntv_evt: %p\n",
		event_num, nevt_idx, ntv_evt);

	/* clear the argument and attribute structures */
	memset(&perf_arg,0,sizeof(pfm_perf_encode_arg_t));
	memset(&(ntv_evt->attr),0,sizeof(struct perf_event_attr));

	// set argument structure fields so the encode
	// function can give us what we need
	perf_arg.attr=&ntv_evt->attr;
	perf_arg.fstr=&event_string;

	// set the size of the perf attr struct before getting pfm encoding
	ntv_evt->attr.size = sizeof(struct perf_event_attr);

	/* use user provided name of the event to get the */
	/* perf_event encoding and a fully qualified event string */
	ret = pfm_get_os_event_encoding(name,
					PFM_PLM0 | PFM_PLM3,
					PFM_OS_PERF_EVENT_EXT,
					&perf_arg);

	// If the encode function failed, skip processing of the event_string
	if ((ret != PFM_SUCCESS) || (event_string == NULL)) {
		SUBDBG("encode failed for event: %s, returned: %d\n",
			name, ret);

		// we need to remember that this event encoding failed
		// but still create the native event table
		// the event table is used by the list so we put what we
		// can get into it
		// but the failure doing the encode causes us to
		// return null to our caller
		encode_failed = 1;

		// Noting the encode_failed error in the attr.config allows
		// any later validate attempts to return an error value

		// ??? .config is 64-bits? --vmw
		ntv_evt->attr.config = 0xFFFFFF;

		// we also want to make it look like a cpu number
		// was not provided as an event mask
		perf_arg.cpu = -1;

		// Why don't we just return NULL here? --vmw
		//return NULL;
	}

	// free string allocated by pfm_get_os_event_encoding
	free(*(perf_arg.fstr));
	// get a copy of the event name and break it up into its parts
	event_string = strdup(name);

	SUBDBG("event_string: %s\n", event_string);

	// get the pmu name, event name and mask list pointers
	// from the event string
	event = strstr (event_string, "::");
	if (event != NULL) {
		*event = 0;     // null terminate pmu name
		event += 2;     // event name follows '::'
		pmu_name = strdup(event_string);
	} else {
		// no pmu name in event string
		pmu_name = malloc(2);
		pmu_name[0] = 0;
		event = event_string;
	}
	masks = strstr (event, ":");
	if (masks != NULL) {
		*masks = 0;     // null terminate event name
		masks += 1;     // masks follow :
	} else {
		masks = "";
	}

	// build event name to find, put a pmu name on it if we have one
	if (strlen(pmu_name) == 0) {
		sprintf(fullname,"%s", event);
	} else {
		sprintf(fullname,"%s::%s", pmu_name, event);
	}

	SUBDBG("pmu_name: %s, event: %s, masks: %s, fullname: %s\n",
		pmu_name, event, masks, fullname);

	// if the libpfm4 index was not provided,
	// try to get one based on the event name passed in.

	/* This may return a value for a disabled PMU */
	if (libpfm4_index == -1) {
		libpfm4_index = pfm_find_event(fullname);
		if (libpfm4_index < 0) {
			free(event_string);
			free(pmu_name);
			_papi_hwi_unlock( NAMELIB_LOCK );
			SUBDBG("EXIT: error from libpfm4 find event\n");
			return NULL;
		}
		SUBDBG("libpfm4_index: %#x\n", libpfm4_index);
	}

	// get this events information from libpfm4,
	// if unavailable return event not found (structure be zeroed)
	memset( &einfo, 0, sizeof( pfm_event_info_t ));
	einfo.size = sizeof(pfm_event_info_t);
	if ((ret = pfm_get_event_info(libpfm4_index,
			PFM_OS_PERF_EVENT_EXT, &einfo)) != PFM_SUCCESS) {
		free(event_string);
		free(pmu_name);
		_papi_hwi_unlock( NAMELIB_LOCK );
		SUBDBG("EXIT: pfm_get_event_info failed with %d\n", ret);
		return NULL;
	}

	// if pmu type is not one supported by this component,
	// return event not found (structure be zeroed)
	memset(&pinfo,0,sizeof(pfm_pmu_info_t));
	pinfo.size = sizeof(pfm_pmu_info_t);
	pfm_get_pmu_info(einfo.pmu, &pinfo);
	if (pmu_is_present_and_right_type(&pinfo, event_table->pmu_type) == 0) {
		free(event_string);
		free(pmu_name);
		_papi_hwi_unlock( NAMELIB_LOCK );
		SUBDBG("EXIT: PMU not supported by this component: einfo.pmu: %d, PFM_PMU_TYPE_CORE: %d\n", einfo.pmu, PFM_PMU_TYPE_CORE);
		return NULL;
	}

	ntv_evt->allocated_name=strdup(name);
	ntv_evt->mask_string=strdup(masks);
	ntv_evt->component=cidx;
	ntv_evt->pmu=pmu_name;
	ntv_evt->base_name=strdup(event);
	ntv_evt->pmu_plus_name=strdup(fullname);
	ntv_evt->libpfm4_idx=libpfm4_index;
	ntv_evt->event_description=strdup(einfo.desc);
	ntv_evt->users=0;      /* is this needed? */
	ntv_evt->cpu=perf_arg.cpu;

	SUBDBG("ntv_evt->mask_string: %p (%s)\n",
		ntv_evt->mask_string, ntv_evt->mask_string);
	char *msk_ptr = strdup(masks);		// get a work copy of the mask string before we free the space it was in
	free(event_string);

	char mask_desc[PAPI_HUGE_STR_LEN] = "";

	// if there is any mask data, collect their descriptions
	if ((msk_ptr != NULL)  &&  (strlen(msk_ptr) > 0)) {
		// go get the descriptions for each of the
		// masks provided with this event
		char *ptr = msk_ptr;
		SUBDBG("ptr: %p (%s)\n", ptr, ptr);
		while (ptr != NULL) {
			char *ptrm = strstr(ptr, ":");
			if (ptrm != NULL) {
				*ptrm = '\0';
				ptrm++;
			}

			// get the length of the mask name
			char *wrk = strchr(ptr, '=');
			unsigned int msk_name_len;
			if (wrk != NULL) {
				msk_name_len = wrk - ptr;
				SUBDBG("Found =, length=%d\n",msk_name_len);
			} else {
				msk_name_len = strlen (ptr);
				SUBDBG("No =, length=%d\n",msk_name_len);
			}

			int i, mask_found=0;
			for (i=0 ; i<einfo.nattrs ; i++) {
				// get this events attribute information
				// from libpfm4, if unavailable return
				// event not found (table must be cleared)
				memset (&ainfo, 0,
					sizeof(pfm_event_attr_info_t));
				ainfo.size = sizeof(pfm_event_attr_info_t);
				ret = pfm_get_event_attr_info(libpfm4_index,
					i, PFM_OS_PERF_EVENT_EXT, &ainfo);
				if (ret != PFM_SUCCESS) {
               free (msk_ptr);
               free(pmu_name);
					SUBDBG("EXIT: error libpfm4 find event: Attribute info not found, libpfm4_index: %#x, ret: %d\n", libpfm4_index, _papi_libpfm4_error(ret));
               _papi_hwi_unlock( NAMELIB_LOCK );
					return NULL;
				}

				// if this is the one we want,
				// append its description
				if ((msk_name_len == strlen(ainfo.name))  &&
					(strncmp(ptr, ainfo.name, msk_name_len) == 0)) {
					mask_found=1;
					SUBDBG("Found mask: libpfm4=%s -- matches %s --  i: %d, %d %zu\n",
						ainfo.name, ptr, i, msk_name_len, strlen(ainfo.name));
					// find out how much space is left in the mask description work buffer we are building
					unsigned int mskleft = sizeof(mask_desc) - strlen(mask_desc);
					// if no space left, just discard this mask description
					if (mskleft <= 1) {
						SUBDBG("EXIT: Attribute description discarded: %s\n", ainfo.desc);
						break;
					}
					// if description buffer is not empty, put in mask description separator
					if (strlen(mask_desc) > 0) {
						strcat (mask_desc, ":");
						mskleft--;
					}
					// if new description will not all fit in buffer, report truncation
					if (mskleft < (strlen(ainfo.desc) + 1)) {
						SUBDBG("EXIT: Attribute description truncated: %s\n", ainfo.desc);
					}
					// move as much of this description as will fit
					strncat (mask_desc, ainfo.desc, mskleft-1);
					mask_desc[mskleft-1] = '\0';
					break;
				}
			}

			/* See if we had a mask that wasn't found */
			if (!mask_found) {
            free(msk_ptr);
            free(pmu_name);
            SUBDBG("EXIT: error libpfm4 find event: Mask not found: %s.\n", ptr);
            _papi_hwi_unlock( NAMELIB_LOCK );
				return NULL;
			}

			// if we have filled the work buffer, we can quit now
			if ( (sizeof(mask_desc) - strlen(mask_desc))  <= 1) {
				break;
			}
			ptr = ptrm;
		}
	}
	ntv_evt->mask_description=strdup(mask_desc);
	SUBDBG("ntv_evt->mask_description: %p (%s)\n", ntv_evt->mask_description, ntv_evt->mask_description);

	// give back space if we got any
	if (msk_ptr != NULL) {
		free (msk_ptr);
	}

	// create a papi table for this native event, put the index into the event sets array of native events into the papi table
	int new_event_code = _papi_hwi_native_to_eventcode(cidx, libpfm4_index, nevt_idx, ntv_evt->allocated_name);
	_papi_hwi_set_papi_event_string((const char *)ntv_evt->allocated_name);
	_papi_hwi_set_papi_event_code(new_event_code, 1);

	ntv_evt->papi_event_code=new_event_code;

	SUBDBG("Using %#x as index for %s\n", ntv_evt->libpfm4_idx, fullname);
	SUBDBG("num_native_events: %d, allocated_native_events: %d\n", event_table->num_native_events, event_table->allocated_native_events);
	SUBDBG("Native Event: papi_event_code: %#x, libpfm4_idx: %#x, pmu: %s, base_name: %s, mask_string: %s, allocated_name: %s\n",
			ntv_evt->papi_event_code, ntv_evt->libpfm4_idx, ntv_evt->pmu, ntv_evt->base_name, ntv_evt->mask_string, ntv_evt->allocated_name);
	SUBDBG("event_table->native_events[%d]: %p, cpu: %d, attr.config: 0x%"PRIx64", attr.config1: 0x%"PRIx64", attr.config2: 0x%"PRIx64", attr.type: 0x%"PRIx32", attr.exclude_user: %d, attr.exclude_kernel: %d, attr.exclude_guest: %d\n",
		nevt_idx, &(event_table->native_events[nevt_idx]), ntv_evt->cpu, ntv_evt->attr.config,
		ntv_evt->attr.config1, ntv_evt->attr.config2, ntv_evt->attr.type,
		ntv_evt->attr.exclude_user, ntv_evt->attr.exclude_kernel, ntv_evt->attr.exclude_guest);

	/* If we've used all of the allocated native events, */
	/* then allocate more room */
	if (event_table->num_native_events >=
				event_table->allocated_native_events-1) {

		SUBDBG("Allocating more room for native events (%d %ld)\n",
			(event_table->allocated_native_events+NATIVE_EVENT_CHUNK),
			(long)sizeof(struct native_event_t) *
			(event_table->allocated_native_events+NATIVE_EVENT_CHUNK));

		event_table->native_events=realloc(event_table->native_events,
				sizeof(struct native_event_t) *
				(event_table->allocated_native_events+NATIVE_EVENT_CHUNK));

		event_table->allocated_native_events+=NATIVE_EVENT_CHUNK;

		// we got new space so we need to reset
		// the pointer to the correct native event in the new space
		ntv_evt = &(event_table->native_events[nevt_idx]);
	}


	// if getting more space for native events failed,
	// report that allocate failed
	if (event_table->native_events==NULL) {
		SUBDBG("EXIT: attempt to get more space for "
			"native events failed\n");
		return NULL;
	}

	// if we created a new event, bump the number used
	if (event_num < 0) {
		event_table->num_native_events++;
	}

	_papi_hwi_unlock( NAMELIB_LOCK );

	if (encode_failed != 0) {
		SUBDBG("EXIT: encoding event failed\n");
		return NULL;
	}

	SUBDBG("EXIT: new_event: %p\n", ntv_evt);
	return ntv_evt;
}


/** @class  get_first_event_next_pmu
 *  @brief  return the first available event that's on an active PMU
 *
 *  @returns returns a libpfm event number
 *  @retval PAPI_ENOEVENT  Could not find an event
 *  Operational note: _pe_libpfm4_init() must be called first to set
 *                    flags for synonymous PMUs. At this writing only 
 *                    amd64_fam17h_zen1_present is defined.
 *  Operational note: We indirectly return the pmu_idx within the
 *                    event data; the calling code uses that to set
 *                    pmu_idx for subsequent calls. All we do is find
 *                    the next valid pmu, if any.
 */

static int
get_first_event_next_pmu(int pmu_idx, int pmu_type)
{
	SUBDBG("ENTER: pmu_idx: %d, pmu_type: %d\n", pmu_idx, pmu_type);
  int pidx, ret;

  pfm_pmu_info_t pinfo;

  // start looking at the next pmu in the list
  pmu_idx++;

	/* We loop forever here and exit if pfm_get_pmu_info() fails.	*/
	/* Before we only went up to PFM_PMU_MAX but this is set at	*/
	/* compile time and might not reflect the number of PMUs if	*/
	/* PAPI is dynamically linked against libpfm4.			*/
  while(1) {

    /* clear the PMU structure (required by libpfm4) */
    memset(&pinfo,0,sizeof(pfm_pmu_info_t));
    pinfo.size = sizeof(pfm_pmu_info_t);
    ret=pfm_get_pmu_info(pmu_idx, &pinfo);

	if (ret==PFM_ERR_INVAL) {
		break;
	}

    if ((ret==PFM_SUCCESS) && amd64_fam17h_zen1_present && strcmp(pinfo.name, "amd64_fam17h") == 0) {
        /* Skip as if invalid; we want the PMU amd64_fam17h_zen1 instead. */
        pmu_idx++;
        continue;
    }
        
    if ((ret==PFM_SUCCESS) && pmu_is_present_and_right_type(&pinfo,pmu_type)) {

      pidx=pinfo.first_event;
      SUBDBG("First event in pmu: %s is %#x\n", pinfo.name, pidx);

      if (pidx<0) {
	/* For some reason no events available */
	/* despite the PMU being active.       */
        /* This can happen, for example with ix86arch */
	/* inside of VMware                           */
      }
      else {
         SUBDBG("EXIT: pidx: %#x\n", pidx);
         return pidx;
      }
    }

    pmu_idx++;

  }

  SUBDBG("EXIT: PAPI_ENOEVNT\n");
  return PAPI_ENOEVNT;
  
}


/***********************************************************/
/* Exported functions                                      */
/***********************************************************/

/** @class  _pe_libpfm4_ntv_name_to_code
 *  @brief  Take an event name and convert it to an event code.
 *
 *  @param[in] *name
 *        -- name of event to convert
 *  @param[out] *event_code
 *        -- pointer to an integer to hold the event code
 *  @param[in] *cidx
 *        -- PAPI component index
 *  @param[in] event_table
 *        -- native event table struct
 *
 *  @retval PAPI_OK event was found and an event assigned
 *  @retval PAPI_ENOEVENT event was not found
 */

int
_pe_libpfm4_ntv_name_to_code( const char *name, unsigned int *event_code,
				int cidx,
				struct native_event_table_t *event_table)
{
  SUBDBG( "ENTER: name: %s, event_code: %p, *event_code: %#x, event_table: %p\n", name, event_code, *event_code, event_table);

  struct native_event_t *our_event;
  int event_num;

  // if we already know this event name, just return its native code
  event_num=find_existing_event(name, event_table);
  if (event_num >= 0) {
	     *event_code=event_table->native_events[event_num].libpfm4_idx;
	     // the following call needs to happen to prevent the internal layer from creating a new papi native event table
	     _papi_hwi_set_papi_event_code(event_table->native_events[event_num].papi_event_code, 1);
	     SUBDBG("EXIT: Found papi_event_code: %#x, libpfm4_idx: %#x\n", event_table->native_events[event_num].papi_event_code, event_table->native_events[event_num].libpfm4_idx);
	     return PAPI_OK;
  }

     // Try to allocate this event to see if it is known by libpfm4, if allocate fails tell the caller it is not valid
     our_event=allocate_native_event(name, -1, cidx, event_table);
     if (our_event==NULL) {
    	 SUBDBG("EXIT: Allocating event: '%s' failed\n", name);
    	 return PAPI_ENOEVNT;
     }

     *event_code = our_event->libpfm4_idx;
     SUBDBG("EXIT: Found code: %#x\n",*event_code);
     return PAPI_OK;
}


/** @class  _pe_libpfm4_ntv_code_to_name
 *  @brief  Take an event code and convert it to a name
 *
 *  @param[in] EventCode
 *        -- PAPI event code
 *  @param[out] *ntv_name
 *        -- pointer to a string to hold the name
 *  @param[in] len
 *        -- length of ntv_name string
 *  @param[in] event_table
 *        -- native event table struct
 *
 *  @retval PAPI_OK       The event was found and converted to a name
 *  @retval PAPI_ENOEVENT The event does not exist
 *  @retval PAPI_EBUF     The event name was too big for ntv_name
 */

int
_pe_libpfm4_ntv_code_to_name(unsigned int EventCode,
			       char *ntv_name, int len,
			       struct native_event_table_t *event_table)
{
	SUBDBG("ENTER: EventCode: %#x, ntv_name: %p, len: %d, event_table: %p\n", EventCode, ntv_name, len, event_table);

	int eidx;
	int papi_event_code;

	// get the attribute index for this papi event
	papi_event_code = _papi_hwi_get_papi_event_code();

	// a papi event code less than 0 is invalid, return error
	if (papi_event_code <= 0) {
		SUBDBG("EXIT: PAPI_ENOEVNT\n");
		return PAPI_ENOEVNT;
	}

	// find our native event table for this papi event code (search list backwards because it improves chances of finding it quickly)
	for (eidx=event_table->num_native_events-1 ; eidx>=0 ; eidx--) {
		if ((papi_event_code == event_table->native_events[eidx].papi_event_code) && (EventCode == ((unsigned)event_table->native_events[eidx].libpfm4_idx))) {
			SUBDBG("Found native_event[%d]: papi_event_code: %#x, libpfm4_idx: %#x\n", eidx, event_table->native_events[eidx].papi_event_code, event_table->native_events[eidx].libpfm4_idx);
			break;
		}
	}

	// if we did not find a match, return an error
	if (eidx < 0) {
		// If we did not find a match in our native event table, then the code passed in has not been
		// allocated yet  It should not be possible to get to this code.  The user has to call the papi
		// code_to_name api with a papi event code for a native event.  But the only way to get one of
		// those is to call either name_to_code or enum_cmp_events first.  When one of these calls is
		// done we allocate the event so it should always be there.

		SUBDBG("EXIT: PAPI_ENOEVNT\n");
		return PAPI_ENOEVNT;
	}

	// if this event is defined by the default pmu, then use only the event name
	// if it is not defined by the default pmu, then use both the pmu name and event name
	char *ename;
	if ((event_table->default_pmu.name) && (strcmp(event_table->default_pmu.name, event_table->native_events[eidx].pmu) == 0)) {
		ename = event_table->native_events[eidx].base_name;
	} else {
		ename = event_table->native_events[eidx].pmu_plus_name;
	}

	// if it will not fit, return error
	if (strlen (ename) >= (unsigned)len) {
		SUBDBG("EXIT: event name %s will not fit in buffer provided\n", ename);
		return PAPI_EBUF;
	}
	strcpy (ntv_name, ename);

	// if this event had masks, also add their names
	char *mname = event_table->native_events[eidx].mask_string;
	if ((mname != NULL)  &&  (strlen(mname) > 0)) {
		if ((strlen(ename) + 8 + strlen(mname)) >= (unsigned)len) {
			SUBDBG("EXIT: Not enough room for event and mask descriptions: need: %u, have: %u", (unsigned)(strlen(ename) + 8 + strlen(mname)), (unsigned)len);
			return PAPI_EBUF;
		}
		strcat (ntv_name, ":");
		strcat (ntv_name, mname);
	}

	SUBDBG("EXIT: event name: %s\n", ntv_name);
	return PAPI_OK;
}


/** @class  _pe_libpfm4_ntv_code_to_descr
 *  @brief  Take an event code and convert it to a description
 *
 *  @param[in] EventCode
 *        -- PAPI event code
 *  @param[out] *ntv_descr
 *        -- pointer to a string to hold the description
 *  @param[in] len
 *        -- length of ntv_descr string
 *  @param[in] event_table
 *        -- native event table struct
 *
 *  @retval PAPI_OK       The event was found and converted to a description
 *  @retval PAPI_ENOEVENT The event does not exist
 *  @retval PAPI_EBUF     The event name was too big for ntv_descr
 *
 *  Return the event description.
 *  If the event has umasks, then include ", masks" and the
 *  umask descriptions follow, separated by commas.
 */


int
_pe_libpfm4_ntv_code_to_descr( unsigned int EventCode,
				 char *ntv_descr, int len,
			         struct native_event_table_t *event_table)
{
	SUBDBG("ENTER: EventCode: %#x, ntv_descr: %p, len: %d: event_table: %p\n", EventCode, ntv_descr, len, event_table);

	int eidx;
	int papi_event_code;
	char *mdesc;
	char *edesc;

	// get the attribute index for this papi event
	papi_event_code = _papi_hwi_get_papi_event_code();

	// a papi event code less than 0 is invalid, return error
	if (papi_event_code <= 0) {
		SUBDBG("EXIT: PAPI_ENOEVNT\n");
		return PAPI_ENOEVNT;
	}

	// find our native event table for this papi event code (search list backwards because it improves chances of finding it quickly)
	for (eidx=event_table->num_native_events-1 ; eidx>=0 ; eidx--) {
		SUBDBG("native_event[%d]: papi_event_code: %#x, libpfm4_idx: %#x\n", eidx, event_table->native_events[eidx].papi_event_code, event_table->native_events[eidx].libpfm4_idx);
		if ((papi_event_code == event_table->native_events[eidx].papi_event_code) && (EventCode == ((unsigned)event_table->native_events[eidx].libpfm4_idx))) {
			break;
		}
	}

	// if we did not find a match, return an error
	if (eidx < 0) {
		// If we did not find a match in our native event table, then the code passed in has not been
		// allocated yet  It should not be possible to get to this code.  The user has to call the papi
		// code_to_name api with a papi event code for a native event.  But the only way to get one of
		// those is to call either name_to_code or enum_cmp_events first.  When one of these calls is
		// done we allocate the event so it should always be there.

		SUBDBG("EXIT: PAPI_ENOEVNT\n");
		return PAPI_ENOEVNT;
	}

	edesc = event_table->native_events[eidx].event_description;

	// if it will not fit, return error
	if (strlen (edesc) >= (unsigned)len) {
		SUBDBG("EXIT: event name %s will not fit in buffer provided\n", edesc);
		return PAPI_EBUF;
	}
	strcpy (ntv_descr, edesc);

	// if this event had masks, also add their descriptions
	mdesc = event_table->native_events[eidx].mask_description;
	if ((mdesc != NULL)  &&  (strlen(mdesc) > 0)) {
		if ((strlen(edesc) + 8 + strlen(mdesc)) >= (unsigned)len) {
			SUBDBG("EXIT: Not enough room for event and mask descriptions: need: %u, have: %u", (unsigned)(strlen(edesc) + 8 + strlen(mdesc)), (unsigned)len);
			return PAPI_EBUF;
		}
		strcat (ntv_descr, ", masks:");
		strcat (ntv_descr, mdesc);
	}

	SUBDBG("EXIT: event description: %s\n", ntv_descr);
	return PAPI_OK;
}


int
_pe_libpfm4_ntv_code_to_info(unsigned int EventCode,
			       PAPI_event_info_t *info,
			       struct native_event_table_t *event_table)
{
	SUBDBG("ENTER: EventCode: %#x, info: %p, event_table: %p\n", EventCode, info, event_table);

	int ret;

	// get the event name first
	if ((ret = _pe_libpfm4_ntv_code_to_name(EventCode, info->symbol, sizeof(info->symbol), event_table)) != PAPI_OK) {
		SUBDBG("EXIT: _pe_libpfm4_ntv_code_to_name returned: %d\n", ret);
		return PAPI_ENOEVNT;
	}

	if ((ret = _pe_libpfm4_ntv_code_to_descr(EventCode, info->long_descr, sizeof(info->long_descr), event_table)) != PAPI_OK) {
		SUBDBG("EXIT: _pe_libpfm4_ntv_code_to_descr returned: %d\n", ret);
		return PAPI_ENOEVNT;
	}

	SUBDBG("EXIT: EventCode: %#x, name: %s, desc: %s\n", EventCode, info->symbol, info->long_descr);
	return PAPI_OK;
}


/** @class  _pe_libpfm4_ntv_enum_events
 *  @brief  Walk through all events in a pre-defined order
 *
 *  @param[in,out] *PapiEventCode
 *        -- PAPI event code to start with
 *  @param[in] modifier
 *        -- describe how to enumerate
 *  @param[in] event_table
 *        -- native event table struct
 *
 *  @retval PAPI_OK       The event was found and converted to a description
 *  @retval PAPI_ENOEVENT The event does not exist
 *  @retval PAPI_ENOIMPL  The enumeration method requested in not implemented
 *
 */

int
_pe_libpfm4_ntv_enum_events( unsigned int *PapiEventCode,
			       int modifier, int cidx,
			       struct native_event_table_t *event_table) {

	SUBDBG("ENTER: PapiEventCode: %p, *PapiEventCode: %#x, modifier: %d, event_table: %p\n", PapiEventCode, *PapiEventCode, modifier, event_table);

	int code,ret, pnum;
	int max_umasks;
	char event_string[BUFSIZ];
	pfm_pmu_info_t pinfo;
	pfm_event_info_t einfo;
	struct native_event_t *our_event;

	/* return first event if so specified */
	if ( modifier == PAPI_ENUM_FIRST ) {
		attr_idx = 0;   // set so if they want attribute information, it will start with the first attribute
		code=get_first_event_next_pmu(-1, event_table->pmu_type);
	   if (code < 0 ) {
	       SUBDBG("EXIT: Invalid component first event code: %d\n", code);
	      return code;
	   }

		// get the event information from libpfm4 (must zero structure)
		memset( &einfo, 0, sizeof( pfm_event_info_t ));
		einfo.size = sizeof(pfm_event_info_t);
		if ((ret = pfm_get_event_info(code, PFM_OS_PERF_EVENT_EXT, &einfo)) != PFM_SUCCESS) {
			SUBDBG("EXIT: pfm_get_event_info returned: %d\n", ret);
			return PAPI_ENOIMPL;
		}

		// get the pmu information from libpfm4 (must zero structure)
		memset( &pinfo, 0, sizeof(pfm_pmu_info_t) );
		pinfo.size = sizeof(pfm_pmu_info_t);
		ret=pfm_get_pmu_info(einfo.pmu, &pinfo);
		if (ret!=PFM_SUCCESS) {
			SUBDBG("EXIT: pfm_get_pmu_info returned: %d\n", ret);
			return ret;
		}

		// build full event name
		sprintf (event_string, "%s::%s", pinfo.name, einfo.name);
		SUBDBG("code: %#x, pmu: %s, event: %s, event_string: %s\n", code, pinfo.name, einfo.name, event_string);

		// go allocate this event, need to create tables used by the get event info call that will probably follow
		if ((our_event = allocate_native_event(event_string, code, cidx, event_table)) == NULL) {
			// allocate may have created the event table but returned NULL to tell the caller the event string was invalid (attempt to encode it failed).
			// if the caller wants to use this event to count something, it will report an error
			// but if the caller is just interested in listing the event, then we need an event table with an event name and libpfm4 index
			int evt_idx;
			if ((evt_idx = find_existing_event(event_string, event_table)) < 0) {
				SUBDBG("EXIT: Allocating event: '%s' failed\n", event_string);
				return PAPI_ENOEVNT;
			}

			// give back the new event code
			*PapiEventCode = event_table->native_events[evt_idx].libpfm4_idx;
			SUBDBG("EXIT: event code: %#x\n", *PapiEventCode);
			return PAPI_OK;
		}

		*PapiEventCode = our_event->libpfm4_idx;

		SUBDBG("EXIT: *PapiEventCode: %#x\n", *PapiEventCode);
		return PAPI_OK;
	}

	/* Handle looking for the next event */
	if ( modifier == PAPI_ENUM_EVENTS ) {
		attr_idx = 0;   // set so if they want attribute information, it will start with the first attribute

		// get the next event code from libpfm4, if there are no more in this pmu find first event in next pmu
		if ((code = pfm_get_event_next(*PapiEventCode)) < 0) {

			// get this events information from libpfm4, we need the pmu number of the last event we processed (table must be cleared)
			memset( &einfo, 0, sizeof( pfm_event_info_t ));
			einfo.size = sizeof(pfm_event_info_t);
			if ((ret = pfm_get_event_info(*PapiEventCode, PFM_OS_PERF_EVENT_EXT, &einfo)) != PFM_SUCCESS) {
				SUBDBG("EXIT: pfm_get_event_info returned: %d\n", ret);
				return PAPI_ENOIMPL;
			}
			SUBDBG("*PapiEventCode: %#x, event: %s\n", *PapiEventCode, einfo.name);

			// get the pmu number of the last event
			pnum = einfo.pmu;

			SUBDBG("pnum: %d\n", pnum);
			code=get_first_event_next_pmu(pnum, event_table->pmu_type);
			if (code < 0) {
				SUBDBG("EXIT: No more PMUs to list, returning: %d\n", code);
				return code;
			}
		}


		// get the event information from libpfm4 (must zero structure)
		memset( &einfo, 0, sizeof( pfm_event_info_t ));
		einfo.size = sizeof(pfm_event_info_t);
		if ((ret = pfm_get_event_info(code, PFM_OS_PERF_EVENT_EXT, &einfo)) != PFM_SUCCESS) {
			SUBDBG("EXIT: pfm_get_event_info returned: %d\n", ret);
			return PAPI_ENOIMPL;
		}

		// get the pmu information from libpfm4 (must zero structure)
		memset( &pinfo, 0, sizeof(pfm_pmu_info_t) );
		pinfo.size = sizeof(pfm_pmu_info_t);
		ret=pfm_get_pmu_info(einfo.pmu, &pinfo);
		if (ret!=PFM_SUCCESS) {
			SUBDBG("EXIT: pfm_get_pmu_info returned: %d\n", ret);
			return ret;
		}

		// build full event name
		sprintf (event_string, "%s::%s", pinfo.name, einfo.name);
		SUBDBG("code: %#x, pmu: %s, event: %s, event_string: %s\n", code, pinfo.name, einfo.name, event_string);

		// go allocate this event, need to create tables used by the get event info call that will follow
		if ((our_event = allocate_native_event(event_string, code, cidx, event_table)) == NULL) {
			// allocate may have created the event table but returned NULL to tell the caller the event string was invalid (attempt to encode it failed).
			// if the caller wants to use this event to count something, it will report an error
			// but if the caller is just interested in listing the event, then we need an event table with an event name and libpfm4 index
			int evt_idx;
			if ((evt_idx = find_existing_event(event_string, event_table)) < 0) {
				SUBDBG("EXIT: Allocating event: '%s' failed\n", event_string);
				return PAPI_ENOEVNT;
			}

			// give back the new event code
			*PapiEventCode = event_table->native_events[evt_idx].libpfm4_idx;
			SUBDBG("EXIT: event code: %#x\n", *PapiEventCode);
			return PAPI_OK;
		}

		// give back the new event code
		*PapiEventCode = our_event->libpfm4_idx;

		SUBDBG("EXIT: *PapiEventCode: %#x\n", *PapiEventCode);
		return PAPI_OK;
	}

	/* We don't handle PAPI_NTV_ENUM_UMASK_COMBOS */
	if ( modifier == PAPI_NTV_ENUM_UMASK_COMBOS ) {
		SUBDBG("EXIT: do not support umask combos yet\n");
		return PAPI_ENOIMPL;
	} 

	/* Enumerate PAPI_NTV_ENUM_UMASKS (umasks on an event) */
	if ( modifier == PAPI_NTV_ENUM_UMASKS ) {
		// get this events information from libpfm4, we need the number of masks this event knows about (table must be cleared)
		memset( &einfo, 0, sizeof( pfm_event_info_t ));
		einfo.size = sizeof(pfm_event_info_t);
		if ((ret = pfm_get_event_info(*PapiEventCode, PFM_OS_PERF_EVENT_EXT, &einfo)) != PFM_SUCCESS) {
			SUBDBG("EXIT: pfm_get_event_info returned: %d\n", ret);
			return PAPI_ENOIMPL;
		}
//      SUBDBG("*PapiEventCode: %#x, einfo.name: %s, einfo.code: %#x, einfo.nattrs: %d\n", *PapiEventCode, einfo.name, einfo.code, einfo.nattrs);

		// set max number of masks
		max_umasks = einfo.nattrs;

		// if we reached last attribute, return error to show we are done with this events masks
		if (attr_idx == max_umasks) {
			SUBDBG("EXIT: already processed all umasks: attr_idx: %d\n", attr_idx);
			return PAPI_ENOEVNT;
		}

		// find the event table for this event, we need the pmu name and event name without any masks
		int ntv_idx = _papi_hwi_get_ntv_idx(_papi_hwi_get_papi_event_code());
		if (ntv_idx < 0) {
			SUBDBG("EXIT: _papi_hwi_get_ntv_idx returned: %d\n", ntv_idx);
			return ntv_idx;
		}
		char *ename = event_table->native_events[ntv_idx].pmu_plus_name;
		if ((ename == NULL)  ||  (strlen(ename) >= sizeof(event_string))) {
			SUBDBG("EXIT: Event name will not fit into buffer\n");
			return PAPI_EBUF;
		}
		strcpy (event_string, ename);
		SUBDBG("event_string: %s\n", event_string);

		// go get the attribute information for this event
		// libpfm4 likes the table cleared
		pfm_event_attr_info_t ainfo;
		memset (&ainfo, 0, sizeof(pfm_event_attr_info_t));
		ainfo.size = sizeof(pfm_event_attr_info_t);
		ret = pfm_get_event_attr_info(*PapiEventCode, attr_idx, PFM_OS_PERF_EVENT_EXT, &ainfo);
		if (ret != PFM_SUCCESS) {
			SUBDBG("EXIT: Attribute info not found, EventCode: %#x, attr_idx: %d, ret: %d\n", *PapiEventCode, attr_idx, _papi_libpfm4_error(ret));
			return _papi_libpfm4_error(ret);
		}
		SUBDBG("*PapiEventCode: %#x, attr_idx: %d, type: %d, name: %s, description: %s\n", *PapiEventCode, attr_idx, ainfo.type, ainfo.name, ainfo.desc);

		if (strlen(event_string) + strlen(ainfo.name) + 35 > sizeof(event_string)) {
			SUBDBG("EXIT: Event name and mask will not fit into buffer\n");
			return PAPI_EBUF;
		}

		strcat (event_string, ":");
		strcat (event_string, ainfo.name);
		switch (ainfo.type) {
			case PFM_ATTR_UMASK:
				break;
			case PFM_ATTR_MOD_BOOL:
			case PFM_ATTR_MOD_INTEGER:
				// a few attributes require a non-zero value to encode correctly (most would accept zero here)
				strcat(event_string,"=0");
				break;
			default:
				SUBDBG("EXIT: Unsupported attribute type: %d", ainfo.type);
				return PAPI_EATTR;
		}

		// go allocate this event, need to create tables used by the get event info call that will follow
		if ((our_event = allocate_native_event(event_string, *PapiEventCode, cidx, event_table)) == NULL) {
			// allocate may have created the event table but returned NULL to tell the caller the event string was invalid.
			// if the caller wants to use this event to count something, it must report the error
			// but if the caller is just interested in listing the event (like this code), then find the table that was created and return its libpfm4 index
			int evt_idx;
			if ((evt_idx = find_existing_event(event_string, event_table)) < 0) {
				SUBDBG("EXIT: Allocating event: '%s' failed\n", event_string);
				return PAPI_ENOEVNT;
			}
			// bump so next time we will use next attribute
			attr_idx++;
			// give back the new event code
			*PapiEventCode = event_table->native_events[evt_idx].libpfm4_idx;
			SUBDBG("EXIT: event code: %#x\n", *PapiEventCode);
			return PAPI_OK;
		}

		// bump so next time we will use next attribute
		attr_idx++;

		// give back the new event code
		*PapiEventCode = our_event->libpfm4_idx;

		SUBDBG("EXIT: event code: %#x\n", *PapiEventCode);
		return PAPI_OK;
	}

	/* Enumerate PAPI_NTV_ENUM_GROUPS (groups on an event) */
	if ( modifier == PAPI_NTV_ENUM_GROUPS ) {
		SUBDBG("EXIT: do not support enumerating groups in this component\n");
		return PAPI_ENOIMPL;
	}

	/* An unknown enumeration method was indicated */

	SUBDBG("EXIT: Invalid modifier argument provided\n");
	return PAPI_ENOIMPL;
}


/** @class  _pe_libpfm4_shutdown
 *  @brief  Shutdown any initialization done by the libpfm4 code
 *
 *  @param[in] event_table
 *        -- native event table struct
 *
 *  @retval PAPI_OK       We always return PAPI_OK
 *
 */

int 
_pe_libpfm4_shutdown(papi_vector_t *my_vector,
		struct native_event_table_t *event_table) {
  SUBDBG("ENTER: event_table: %p\n", event_table);

  int i;

  for (i=0 ; i<PAPI_PMU_MAX ; i++) {
	  if (my_vector->cmp_info.pmu_names[i] != NULL) {
		  free (my_vector->cmp_info.pmu_names[i]);
	  }
  }

  /* clean out and free the native events structure */
  _papi_hwi_lock( NAMELIB_LOCK );

  /* free memory allocated with strdup or malloc */
  for( i=0; i<event_table->num_native_events; i++) {
     free(event_table->native_events[i].base_name);
     free(event_table->native_events[i].pmu_plus_name);
     free(event_table->native_events[i].pmu);
     free(event_table->native_events[i].allocated_name);
     free(event_table->native_events[i].mask_string);
     free(event_table->native_events[i].event_description);
     if (event_table->native_events[i].mask_description != NULL) {
          free(event_table->native_events[i].mask_description);
     }
  }

  free(event_table->native_events);

  _papi_hwi_unlock( NAMELIB_LOCK );

  SUBDBG("EXIT: PAPI_OK\n");
  return PAPI_OK;
}


/** @class  _pe_libpfm4_init
 *  @brief  Initialize the libpfm4 code
 *
 *  @param[in] component
 *        -- pointer to component structure
 *  @param[in] cidx
 *        -- component index
 *  @param[in] event_table
 *        -- native event table structure
 *  @param[in] pmu_type
 *        -- type of PMU to init.  Should only be PMU_TYPE_CORE
 *
 *  @retval PAPI_OK       We initialized correctly
 *  @retval PAPI_ECMP     There was an error initializing the component
 *
 */

int
_pe_libpfm4_init(papi_vector_t *component, int cidx,
		   struct native_event_table_t *event_table,
		   int pmu_type) {

	int detected_pmus=0, found_default=0;
	int i;
	int j=0;
	unsigned int ncnt;

	pfm_err_t retval = PFM_SUCCESS;
	pfm_pmu_info_t pinfo;
	unsigned int strSize;

	/* allocate the native event structure */
	event_table->num_native_events=0;
	event_table->pmu_type=pmu_type;

	event_table->native_events=calloc(NATIVE_EVENT_CHUNK,
					sizeof(struct native_event_t));
	if (event_table->native_events==NULL) {
		return PAPI_ENOMEM;
	}

	event_table->allocated_native_events=NATIVE_EVENT_CHUNK;

	/* Count number of present PMUs */
	detected_pmus=0;
	ncnt=0;

	/* init default pmu */
	/* need to init pinfo or pfmlib might complain */
	memset(&(event_table->default_pmu), 0, sizeof(pfm_pmu_info_t));
	event_table->default_pmu.size = sizeof(pfm_pmu_info_t);
	retval=pfm_get_pmu_info(0, &(event_table->default_pmu));

    SUBDBG("Prescan for aliases.\n");
    /* We have to see if we have aliases in there as separate PMUs, */
    /* we don't want both PMUs with all the events duplicated.      */
    /* For aliases, either is valid alone, but if both are present  */
    /* specify a preference in the code.                            */
    /* Alias: amd64_fam17h_zen1 over amd64_fam17h.                  */
    /* Alias flags are static ints global to this file.             */
    i=0;
	while(1) {
		memset(&pinfo,0,sizeof(pfm_pmu_info_t));
		pinfo.size = sizeof(pfm_pmu_info_t);
		retval=pfm_get_pmu_info(i, &pinfo);

		/* We're done if we hit an invalid PMU entry		    */
		/* We can't check against PFM_PMU_MAX as that might not	*/
		/* match if libpfm4 is dynamically linked		        */

		if (retval==PFM_ERR_INVAL) {
			break;
		}

		if ( (retval==PFM_SUCCESS) && (pinfo.name != NULL) &&
			(pmu_is_present_and_right_type(&pinfo,pmu_type)) &&
            (strcmp(pinfo.name,"amd64_fam17h_zen1") == 0) ) {
            amd64_fam17h_zen1_present = 1;
        }
        i++;
    } 

	SUBDBG("Detected pmus:\n");

	/* Create the list of available PMUs */

	i=0;
	while(1) {
		memset(&pinfo,0,sizeof(pfm_pmu_info_t));
		pinfo.size = sizeof(pfm_pmu_info_t);
		retval=pfm_get_pmu_info(i, &pinfo);

		/* We're done if we hit an invalid PMU entry		*/
		/* We can't check against PFM_PMU_MAX as that might not	*/
		/* match if libpfm4 is dynamically linked		*/

		if (retval==PFM_ERR_INVAL) {
			break;
		}

		if ((retval==PFM_SUCCESS) && (pinfo.name != NULL) &&
			(pmu_is_present_and_right_type(&pinfo,pmu_type))) {

			/* Check for PMUs that should not be included even */
			/* though they are valid */

			/* for backwards compatability reasons               */
			/* adm64_fam17h and zen1 are aliases to the same PMU */
			/* so we should only include one of them.            */

			if ( (strcmp(pinfo.name,"amd64_fam17h") == 0) &&
				amd64_fam17h_zen1_present) {

				i++;
				continue;
			}

			/* if we arrive here the PMU is valid and we want to add it */

			SUBDBG("\t%d %s %s %d\n",i,
				pinfo.name,pinfo.desc,pinfo.type);

			detected_pmus++;
			ncnt+=pinfo.nevents;

			/* FIXME: why do we use 'j' and not base this on detected_pmus? */

			if (j < PAPI_PMU_MAX) {
				component->cmp_info.pmu_names[j++] =
							strdup(pinfo.name);
			}

			/* For backwards compatability reasons PAPI has the */
			/* idea of the "default" PMU that events can be */
			/* assigned to without having to specify the PMU */
			/* explicitly.  We use libpfm4's PMU_TYPE_CORE */
			/* to determine this, but it can cause issues when */
			/* an architecture has multiple CORE PMUs */

			if (pmu_type & PMU_TYPE_CORE) {

				if (!strcmp(pinfo.name,"ix86arch")) {

					/* On x86 chips don't make the generic ix86arch PMU */
					/* the default. */

				} else if (strstr(pinfo.desc,"E-Core")) {

					/* For heterogeneous systems with both E-cores and P-cores */
					/* we use P-core as default.  Currently there isn't a way */
					/* to do this with libpfm4 except or this hack.           */
					/* FIXME: find better way to get this info from libpfm4   */
				}
				else if (pinfo.type!=PFM_PMU_TYPE_CORE) {
					/* skip the "perf" and "perf_raw" PMUs */
				}
				else {
					/* Actually set the PMU as the default one */

					memcpy(&(event_table->default_pmu),
						&pinfo,sizeof(pfm_pmu_info_t));
					found_default++;
				}



				/* For ARM processors override the hw_info model_string */
				/* (gathered from /proc/cpuinfo) with the more complete */
				/* model info provided by libpfm4.                      */
				/* FIXME: is this the proper place to do this? */
				if ( (pinfo.type==PFM_PMU_TYPE_CORE) &&
					( _papi_hwi_system_info.hw_info.vendor >= PAPI_VENDOR_ARM_ARM)) {
					if (strlen(_papi_hwi_system_info.hw_info.model_string) == 0) {
						strSize = sizeof(_papi_hwi_system_info.hw_info.model_string);
						strncpy( _papi_hwi_system_info.hw_info.model_string, pinfo.desc, strSize - 1);
					}
				}
			}

			/* FIXME: this is a remnant from when CORE and UNCORE PMU init  */
			/*        shared the same init code.  It should not be possible */
			/*        to hit this code path anymore.                        */
			if (pmu_type==PMU_TYPE_UNCORE) {
				/* To avoid confusion, no "default" CPU for uncore */
				found_default=1;
			}
		}
		i++;
	}
	SUBDBG("%d native events detected on %d pmus\n",ncnt,detected_pmus);

	if (detected_pmus==0) {
		SUBDBG("Could not find any PMUs\n");
		return PAPI_ENOSUPP;
	}

	/* Error if we couldn't pick a default PMU */
	if (!found_default) {
		return PAPI_ECMP;
	}

	/* Error if we found more than one default PMU */
	/* FIXME: this breaks on heterogeneous CPU systems */
	if (found_default>1) {
		return PAPI_ECOUNT;
	}

	component->cmp_info.num_native_events = ncnt;

	component->cmp_info.num_cntrs = event_table->default_pmu.num_cntrs+
				event_table->default_pmu.num_fixed_cntrs;

	SUBDBG( "num_counters: %d\n", component->cmp_info.num_cntrs );

	/* Setup presets, only if Component 0 and default core PMU */
	if ((cidx==0) && (found_default)) {
		retval = _papi_load_preset_table( (char *)event_table->default_pmu.name,
				event_table->default_pmu.pmu, cidx );
		if ( retval!=PAPI_OK ) {
			return PAPI_ENOEVNT;
		}
	}

	return PAPI_OK;
}

/** @class  _peu_libpfm4_init
 *  @brief  Initialize the libpfm4 code
 *
 *  @param[in] event_table
 *        -- native event table struct
 *
 *  @retval PAPI_OK       We initialized correctly
 *  @retval PAPI_ECMP     There was an error initializing the component
 *
 */

int
_peu_libpfm4_init(papi_vector_t *my_vector, int cidx,
		   struct native_event_table_t *event_table,
		   int pmu_type) {

   int detected_pmus=0;
   int i;
   int j=0;
   pfm_err_t retval = PFM_SUCCESS;
   unsigned int ncnt;
   pfm_pmu_info_t pinfo;

	(void)cidx;

   /* allocate the native event structure */

   event_table->num_native_events=0;
   event_table->pmu_type=pmu_type;

   event_table->native_events=calloc(NATIVE_EVENT_CHUNK,
					   sizeof(struct native_event_t));
   if (event_table->native_events==NULL) {
      return PAPI_ENOMEM;
   }
   event_table->allocated_native_events=NATIVE_EVENT_CHUNK;

   /* Count number of present PMUs */
   detected_pmus=0;
   ncnt=0;

   my_vector->cmp_info.num_cntrs=0;

   SUBDBG("Detected pmus:\n");
	i=0;
	while(1) {
      memset(&pinfo,0,sizeof(pfm_pmu_info_t));
      pinfo.size = sizeof(pfm_pmu_info_t);
      retval=pfm_get_pmu_info(i, &pinfo);

	/* We're done if we hit an invalid PMU entry                    */
	/* We can't check against PFM_PMU_MAX                           */
	/* as that might not match if libpfm4 is dynamically linked     */

	if (retval==PFM_ERR_INVAL) {
		break;
	}

	if ((retval==PFM_SUCCESS) && (pinfo.name != NULL) &&
		(pmu_is_present_and_right_type(&pinfo,pmu_type))) {

	 SUBDBG("\t%d %s %s %d\n",i,pinfo.name,pinfo.desc,pinfo.type);

         detected_pmus++;
	 ncnt+=pinfo.nevents;

	 if ((j < PAPI_PMU_MAX) && (pinfo.name != NULL)) {
	     my_vector->cmp_info.pmu_names[j++] = strdup(pinfo.name);
	 }
         my_vector->cmp_info.num_cntrs += pinfo.num_cntrs+
                                   pinfo.num_fixed_cntrs;
      }
	i++;
   }
   SUBDBG("%d native events detected on %d pmus\n",ncnt,detected_pmus);

   my_vector->cmp_info.num_native_events = ncnt;

   SUBDBG( "num_counters: %d\n", my_vector->cmp_info.num_cntrs );

   return PAPI_OK;
}

