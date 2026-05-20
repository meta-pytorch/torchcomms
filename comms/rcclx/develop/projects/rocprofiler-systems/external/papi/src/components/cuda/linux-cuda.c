/**
 * @file    linux-cuda.c
 *
 * @author  Treece Burgess tburgess@icl.utk.edu (updated in 2024, redesigned to add device qualifier support.)
 * @author  Anustuv Pal   anustuv@icl.utk.edu (updated in 2023, redesigned with multi-threading support.)
 * @author  Tony Castaldo tonycastaldo@icl.utk.edu (updated in 08/2019, to make counters accumulate.)
 * @author  Tony Castaldo tonycastaldo@icl.utk.edu (updated in 2018, to use batch reads and support nvlink metrics.
 * @author  Asim YarKhan yarkhan@icl.utk.edu (updated in 2017 to support CUDA metrics)
 * @author  Asim YarKhan yarkhan@icl.utk.edu (updated in 2015 for multiple CUDA contexts/devices)
 * @author  Heike Jagode (First version, in collaboration with Robert Dietrich, TU Dresden) jagode@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  This file implements a PAPI component that enables PAPI-C to access
 *  hardware monitoring counters for NVIDIA GPU devices through the CuPTI library.
 *
 * The open source software license for PAPI conforms to the BSD
 * License template.
 */

#include <papi.h>
#include <papi_internal.h>
#include <papi_vector.h>

#include <string.h>
#include <stdint.h>

#include "papi_memory.h"
#include "cupti_dispatch.h"
#include "cupti_config.h"
#include "lcuda_debug.h"

#define PAPI_CUDA_MPX_COUNTERS 512
#define PAPI_CUDA_MAX_COUNTERS  30

papi_vector_t _cuda_vector;

/* init and shutdown functions */
static int cuda_init_component(int cidx);
static int cuda_init_thread(hwd_context_t *ctx);
static int cuda_init_control_state(hwd_control_state_t *ctl);
static int cuda_shutdown_thread(hwd_context_t *ctx);
static int cuda_shutdown_component(void);

/* set and update component state */
static int cuda_update_control_state(hwd_control_state_t *ctl,
                                     NativeInfo_t *ntv_info,
                                     int ntv_count, hwd_context_t *ctx);
static int cuda_set_domain(hwd_control_state_t * ctrl, int domain);

/* functions to monitor hardware counters */
static int cuda_start(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int cuda_read(hwd_context_t *ctx, hwd_control_state_t *ctl, long long **val, int flags);
static int cuda_reset(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int cuda_stop(hwd_context_t *ctx, hwd_control_state_t *ctl);
static int cuda_cleanup_eventset(hwd_control_state_t *ctl);
static int cuda_init_private(void);
static int cuda_get_evt_count(int *count);

/* cuda native event conversion utility functions */
static int cuda_ntv_enum_events(unsigned int *event_code, int modifier);
static int cuda_ntv_code_to_name(unsigned int event_code, char *name, int len);
static int cuda_ntv_name_to_code(const char *name, unsigned int *event_code);
static int cuda_ntv_code_to_descr(unsigned int event_code, char *descr, int len);
static int cuda_ntv_code_to_info(unsigned int event_code, PAPI_event_info_t *info);

/* track metadata, such as the EventSet state */
typedef struct {
    int initialized;
    int state;
    int component_id;
} cuda_context_t;

typedef struct {
    int num_events;
    unsigned int domain;
    unsigned int granularity;
    unsigned int overflow;
    unsigned int overflow_signal;
    unsigned int attached;
    int component_id;
    uint64_t *events_id;
    cuptid_info_t info;
    /* struct holding read count, gpu_ctl, etc. */
    cuptip_control_t cuptid_ctx;
} cuda_control_t;

papi_vector_t _cuda_vector = {
    .cmp_info = {
        .name = "cuda",
        .short_name = "cuda",
        .version = "0.1",
        .description = "CUDA profiling via NVIDIA CuPTI interfaces",
        .num_mpx_cntrs = PAPI_CUDA_MPX_COUNTERS,
        .num_cntrs = PAPI_CUDA_MAX_COUNTERS,
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
        .initialized = 0,
    },
    .size = {
        .context = sizeof(cuda_context_t),
        .control_state = sizeof(cuda_control_t),
        .reg_value = 1,
        .reg_alloc = 1,
    },
    .init_component = cuda_init_component,
    .shutdown_component = cuda_shutdown_component,

    .init_thread = cuda_init_thread,
    .shutdown_thread = cuda_shutdown_thread,

    .ntv_enum_events = cuda_ntv_enum_events,
    .ntv_code_to_name = cuda_ntv_code_to_name,
    .ntv_name_to_code = cuda_ntv_name_to_code,
    .ntv_code_to_descr = cuda_ntv_code_to_descr,
    .ntv_code_to_info = cuda_ntv_code_to_info,

    .init_control_state = cuda_init_control_state,
    .set_domain = cuda_set_domain,
    .update_control_state = cuda_update_control_state,
    .cleanup_eventset = cuda_cleanup_eventset,

    .start = cuda_start,
    .stop = cuda_stop,
    .read = cuda_read,
    .reset = cuda_reset,
};

static int cuda_init_component(int cidx)
{
    COMPDBG("Entering with component idx: %d\n", cidx);

    _cuda_vector.cmp_info.CmpIdx = cidx;
    _cuda_vector.cmp_info.num_native_events = -1;
    _cuda_lock = PAPI_NUM_LOCK + NUM_INNER_LOCK + cidx;

    _cuda_vector.cmp_info.disabled = PAPI_EDELAY_INIT;
    sprintf(_cuda_vector.cmp_info.disabled_reason,
            "Not initialized. Access component events to initialize it.");
    return PAPI_EDELAY_INIT;
}

static int cuda_shutdown_component(void)
{
    COMPDBG("Entering.\n");

    if (!_cuda_vector.cmp_info.initialized ||
    _cuda_vector.cmp_info.disabled != PAPI_OK) {
        return PAPI_OK;
    }

    _cuda_vector.cmp_info.initialized = 0;

    return cuptid_shutdown();
}

static int cuda_init_private(void)
{
    int papi_errno = PAPI_OK, len, count = 0;
    const char *disabled_reason;

    _papi_hwi_lock(COMPONENT_LOCK);
    SUBDBG("ENTER\n");

    if (_cuda_vector.cmp_info.initialized) {
        SUBDBG("Skipping cuda_init_private, as the Cuda event table has already been initialized.\n");
        goto fn_exit;
    }

    papi_errno = cuptid_init();
    if (papi_errno != PAPI_OK) {
        /* get and assign the string literal for the disabled reason */
        cuptid_disabled_reason_get(&disabled_reason);
        len = snprintf(_cuda_vector.cmp_info.disabled_reason, PAPI_MAX_STR_LEN, "%s", disabled_reason);
        if (len < 0 || len > PAPI_MAX_STR_LEN) {
            SUBDBG("The disabled reason has been truncated.\n");
        }
        goto fn_fail;
    }

    strcpy(_cuda_vector.cmp_info.disabled_reason, "");

    /* get the number of native events count */
    papi_errno = cuda_get_evt_count(&count);
    _cuda_vector.cmp_info.num_native_events = count;

  fn_exit:
    _cuda_vector.cmp_info.initialized = 1;
    _cuda_vector.cmp_info.disabled = papi_errno;
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    _papi_hwi_unlock(COMPONENT_LOCK);
    return papi_errno;
  fn_fail:
    goto fn_exit;
}

static int check_n_initialize(void)
{
    if (!_cuda_vector.cmp_info.initialized) {
        return cuda_init_private();
    }
    return _cuda_vector.cmp_info.disabled;
}

static int cuda_ntv_enum_events(unsigned int *event_code, int modifier)
{
    SUBDBG("ENTER: event_code: %u, modifier: %d\n", *event_code, modifier);
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
   
    uint64_t code = *(uint64_t *) event_code;
    papi_errno = cuptid_evt_enum(&code, modifier);
    *event_code = (unsigned int) code;
    
fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
fn_fail:
    goto fn_exit;
}

static int cuda_ntv_name_to_code(const char *name, unsigned int *event_code)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }
   
    uint64_t code;
    papi_errno = cuptid_evt_name_to_code(name, &code);
    *event_code = (unsigned int) code;

    fn_exit:
        SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
        return papi_errno;
    fn_fail:
        goto fn_exit;
}

static int cuda_ntv_code_to_name(unsigned int event_code, char *name, int len)
{
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        return papi_errno;
    }

    papi_errno = cuptid_evt_code_to_name((uint64_t) event_code, name, len);

    fn_exit:
        SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
        return papi_errno;
    fn_fail:
        goto fn_exit;
}

static int cuda_ntv_code_to_descr(unsigned int event_code, char *descr, int len)
{
    SUBDBG("ENTER: event_code: %u, descr: %p, len: %d\n", event_code, descr, len);
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = cuptid_evt_code_to_descr((uint64_t) event_code, descr, len);

fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
fn_fail:
    goto fn_exit;
}

static int cuda_ntv_code_to_info(unsigned int event_code, PAPI_event_info_t *info)
{
    SUBDBG("ENTER: event_code: %u, info: %p\n", event_code, info);
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    papi_errno = cuptid_evt_code_to_info((uint64_t) event_code, info);

fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
fn_fail:
    goto fn_exit;
}

static int cuda_init_thread(hwd_context_t *ctx)
{
    cuda_context_t *cuda_ctx = (cuda_context_t *) ctx;
    memset(cuda_ctx, 0, sizeof(*cuda_ctx));
    cuda_ctx->initialized = 1;
    cuda_ctx->component_id = _cuda_vector.cmp_info.CmpIdx;

    return PAPI_OK;
}

static int cuda_shutdown_thread(hwd_context_t *ctx)
{
    cuda_context_t *cuda_ctx = (cuda_context_t *) ctx;
    cuda_ctx->initialized = 0;
    cuda_ctx->state = 0; 

    return PAPI_OK;
}

static int cuda_init_control_state(hwd_control_state_t __attribute__((unused)) *ctl)
{
    COMPDBG("Entering.\n");
    return check_n_initialize();
}

static int cuda_set_domain(hwd_control_state_t __attribute__((unused)) *ctrl, int domain)
{
    COMPDBG("Entering\n");
    if((PAPI_DOM_USER & domain) || (PAPI_DOM_KERNEL & domain) || (PAPI_DOM_OTHER & domain) || (PAPI_DOM_ALL & domain))
        return (PAPI_OK);
    else
        return (PAPI_EINVAL);
}

static int update_native_events(cuda_control_t *, NativeInfo_t *, int);

static int cuda_update_control_state(hwd_control_state_t *ctl, NativeInfo_t *ntv_info,
                              int ntv_count,
                              hwd_context_t *ctx __attribute__((unused)))
{  
    SUBDBG("ENTER: ctl: %p, ntv_info: %p, ntv_count: %d, ctx: %p\n", ctl, ntv_info, ntv_count, ctx);
    int papi_errno = check_n_initialize();
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    /* needed to make sure multipass events are caught with proper error code (PAPI_EMULPASS)*/
    if (ntv_count == 0) {
        return PAPI_OK;
    }

    cuda_control_t *cuda_ctl = (cuda_control_t *) ctl;

    /* allocating memoory for total number of devices */
    if (cuda_ctl->info == NULL) {
        papi_errno = cuptid_thread_info_create(&(cuda_ctl->info));
        if (papi_errno != PAPI_OK) {
            goto fn_exit;
        }   
    }
   
    papi_errno = update_native_events(cuda_ctl, ntv_info, ntv_count);
    if (papi_errno != PAPI_OK) {
        goto fn_exit;
    }

    /* needed to make sure multipass events are caught with proper error code (PAPI_EMULPASS)*/
    papi_errno = cuptid_ctx_create(cuda_ctl->info, &(cuda_ctl->cuptid_ctx), cuda_ctl->events_id, cuda_ctl->num_events);

fn_exit:
    SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
    return papi_errno;
}

struct event_map_item {
    int event_id;
    int frontend_idx;
};

static int compare(const void *a, const void *b)
{
    struct event_map_item *A = (struct event_map_item *) a;
    struct event_map_item *B = (struct event_map_item *) b;
    return  A->event_id - B->event_id;
}

int update_native_events(cuda_control_t *ctl, NativeInfo_t *ntv_info,
                         int ntv_count)
{
    int papi_errno = PAPI_OK;
    struct event_map_item sorted_events[PAPI_CUDA_MAX_COUNTERS];

    if (ntv_count != ctl->num_events) {
        ctl->events_id = papi_realloc(ctl->events_id,
                                      ntv_count * sizeof(*ctl->events_id));
        if (ctl->events_id == NULL) {
            papi_errno = PAPI_ENOMEM;
            goto fn_fail;
        }

        ctl->num_events = ntv_count;
    }

    int i;
    for (i = 0; i < ntv_count; ++i) {
        sorted_events[i].event_id = ntv_info[i].ni_event;
        sorted_events[i].frontend_idx = i;
    }

    for (i = 0; i < ntv_count; ++i) {
        ctl->events_id[i] = sorted_events[i].event_id;
        ntv_info[sorted_events[i].frontend_idx].ni_position = i;
    }

  fn_exit:
    return papi_errno;
  fn_fail:
    ctl->num_events = 0;
    goto fn_exit;
}

/** @class cuda_start
  * @brief Start counting Cuda hardware events.
  *
  * @param *ctx
  *   Vestigial pointer are to structures defined in the components.
  *   They are opaque to the framework and defined as void.
  *   They are remapped to real data in the component rountines that use them.
  *   In this case Cuda.
  * @param *ctl
  *   Contains the encoding's necessary for the hardware to set the counters
  *   to the appropriate conditions.
*/
static int cuda_start(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    COMPDBG("Entering.\n");
    int papi_errno, i;
    cuda_context_t *cuda_ctx = (cuda_context_t *) ctx;
    cuda_control_t *cuda_ctl = (cuda_control_t *) ctl;
   
    /* will need to flesh this out more and decide if I want to keep this, may not need it 
    if (cuda_ctx->state & CUDA_EVENTS_OPENED) {
        SUBDBG("Error! Cannot PAPI_start more than one eventset at a time for every component.");
        papi_errno = PAPI_ECNFLCT;
        goto fn_fail;
    }
    */

    papi_errno = cuptid_ctx_create(cuda_ctl->info, &(cuda_ctl->cuptid_ctx), cuda_ctl->events_id, cuda_ctl->num_events);
    if (papi_errno != PAPI_OK)
        goto fn_fail;

    /* start profiling */
    papi_errno = cuptid_ctx_start( (void *) cuda_ctl->cuptid_ctx);
    if (papi_errno != PAPI_OK)
        goto fn_fail;   

    /* update the EventSet state to running */
    cuda_ctx->state = CUDA_EVENTS_RUNNING;

   fn_exit:
       SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
       return papi_errno;
   fn_fail:
       /* same as above may need to flesh this out more. */
       cuda_ctx->state = CUDA_EVENTS_STOPPED;
       goto fn_exit;
}

/** @class cuda_read
  * @brief Read the Cuda hardware counters.
  *
  * @param *ctl
  *   Contains the encoding's necessary for the hardware to set the counters
  *   to the appropriate conditions.
  * @param **val
  *   Holds the counter values for each added Cuda native event.
*/
static int cuda_read(hwd_context_t __attribute__((unused)) *ctx, hwd_control_state_t *ctl, long long **val, int __attribute__((unused)) flags)
{
    COMPDBG("Entering.\n");
    int papi_errno = PAPI_OK;
    cuda_control_t *cuda_ctl = (cuda_control_t *) ctl;
    SUBDBG("ENTER: ctx: %p, ctl: %p, val: %p, flags: %d\n", ctx, ctl, val, flags);

    if (cuda_ctl->cuptid_ctx == NULL) {
        SUBDBG("Error! Cannot PAPI_read counters for an eventset that has not been PAPI_start'ed.");
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

    papi_errno = cuptid_ctx_read( cuda_ctl->cuptid_ctx, val );
  
  fn_exit:
      SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
      return papi_errno;
  fn_fail:
      goto fn_exit;
}

/** @class cuda_reset
  * @brief Reset the Cuda hardware event counts.
  *
  * @param *ctl
  *   Contains the encoding's necessary for the hardware to set the counters 
  *   to the appropriate conditions.
*/
static int cuda_reset(hwd_context_t __attribute__((unused)) *ctx, hwd_control_state_t *ctl)
{
    int papi_errno;
    cuda_control_t *cuda_ctl = (cuda_control_t *) ctl;

    if (cuda_ctl->cuptid_ctx == NULL) {
        SUBDBG("Cannot reset counters for an eventset that has not been started.");
        return PAPI_EMISC;
    }

    papi_errno = cuptid_ctx_reset(cuda_ctl->cuptid_ctx);
     
    return papi_errno;
}

/** @class cuda_stop
  * @brief Stop counting Cuda hardware events.
  *
  * @param *ctx
  *   Vestigial pointer are to structures defined in the components.
  *   They are opaque to the framework and defined as void.
  *   They are remapped to real data in the component rountines that use them.
  *   In this case Cuda.
  * @param *ctl
  *   Contains the encoding's necessary for the hardware to set the counters
  *   to the appropriate conditions. E.g. Stopped or running.
*/
int cuda_stop(hwd_context_t *ctx, hwd_control_state_t *ctl)
{
    COMPDBG("Entering.\n");
    int papi_errno = PAPI_OK;
    cuda_context_t *cuda_ctx = (cuda_context_t *) ctx;
    cuda_control_t *cuda_ctl = (cuda_control_t *) ctl;

    if (cuda_ctx->state == CUDA_EVENTS_STOPPED) {
        SUBDBG("Error! Cannot PAPI_stop counters for an eventset that has not been PAPI_start'ed.");
        papi_errno = PAPI_EMISC;
        goto fn_fail;
    }

    /* stop counting */
    papi_errno = cuptid_ctx_stop(cuda_ctl->cuptid_ctx);
    if (papi_errno != PAPI_OK) {
        goto fn_fail;
    }

    /* free memory that was used */
    papi_errno = cuptid_ctx_destroy( &(cuda_ctl->cuptid_ctx) );
    if (papi_errno != PAPI_OK) {
    } 

    /* update EventSet state to stopped  */
    cuda_ctx->state = CUDA_EVENTS_STOPPED;
    cuda_ctl->cuptid_ctx = NULL;

   fn_exit:
     SUBDBG("EXIT: %s\n", PAPI_strerror(papi_errno));
     return papi_errno;
   fn_fail:
     goto fn_exit;
}

/** @class cuda_cleanup_eventset
  * @brief Remove all Cuda hardware events from a PAPI event set.
  *
  * @param *ctl
  *   Contains the encoding's necessary for the hardware to set the counters
  *   to the appropriate conditions.
*/
static int cuda_cleanup_eventset(hwd_control_state_t *ctl)
{
    COMPDBG("Entering.\n");
    int papi_errno;
    cuda_control_t *cuda_ctl = (cuda_control_t *) ctl;

    if (cuda_ctl->info) {
        papi_errno = cuptid_thread_info_destroy(&(cuda_ctl->info));
        if (papi_errno != PAPI_OK)
            return papi_errno;
    }

    /* free int array of event id's and reset number of events */
    papi_free(cuda_ctl->events_id);
    cuda_ctl->events_id = NULL;
    cuda_ctl->num_events = 0;

    return PAPI_OK;
}

/** @class cuda_get_evt_count
  * @brief Helper function to count the number of Cuda base event names. 
  *        This count is shown in the util papi_component_avail.
  *
  * @param *count
  *   Count of Cuda base hardware event names.
*/
static int cuda_get_evt_count(int *count)
{
    uint64_t event_code = 0;

    if (cuptid_evt_enum(&event_code, PAPI_ENUM_FIRST) == PAPI_OK) {
        ++(*count);
         
    }
    while (cuptid_evt_enum(&event_code, PAPI_ENUM_EVENTS) == PAPI_OK) {
        ++(*count);
    }

    return PAPI_OK;
}
