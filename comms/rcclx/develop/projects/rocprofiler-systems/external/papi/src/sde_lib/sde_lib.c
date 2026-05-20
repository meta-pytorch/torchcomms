/**
 * @file    sde_lib.c
 * @author  Anthony Danalis
 *          adanalis@icl.utk.edu
 *
 * @brief
 *  This is the main implementation of the functionality needed to
 *  support SDEs in third party libraries.
 */

#include "sde_lib_internal.h"
#include "sde_lib_lock.h"

#define DLSYM_CHECK(name)                \
    do {                                 \
        if ( NULL != (err=dlerror()) ) { \
            name##_ptr = NULL;           \
            SDEDBG("obtain_papi_symbols(): Unable to load symbol %s: %s\n", #name, err);\
            return;                      \
        }                                \
    } while (0)

static long long sdei_compute_q1(void *param);
static long long sdei_compute_med(void *param);
static long long sdei_compute_q3(void *param);
static long long sdei_compute_min(void *param);
static long long sdei_compute_max(void *param);
static inline long long sdei_compute_quantile(void *param, int percent);
static inline long long sdei_compute_edge(void *param, int which_edge);

int papi_sde_compare_long_long(const void *p1, const void *p2);
int papi_sde_compare_int(const void *p1, const void *p2);
int papi_sde_compare_double(const void *p1, const void *p2);
int papi_sde_compare_float(const void *p1, const void *p2);

/** This global variable points to the head of the control state list **/
papisde_control_t *_papisde_global_control = NULL;
int papi_sde_version = PAPI_SDE_VERSION;

#if defined(USE_LIBAO_ATOMICS)
AO_TS_t _sde_hwd_lock_data;
#else //defined(USE_LIBAO_ATOMICS)
pthread_mutex_t _sde_hwd_lock_data;
#endif //defined(USE_LIBAO_ATOMICS)

/*******************************************************************************/
/* Function pointers magic for functions that we expect to access from libpapi */
/*******************************************************************************/

__attribute__((__common__)) void (*papi_sde_check_overflow_status_ptr)(uint32_t cntr_id, long long int value);
__attribute__((__common__)) int  (*papi_sde_set_timer_for_overflow_ptr)(void);

static inline void
sdei_check_overflow_status(uint32_t cntr_uniq_id, long long int latest){
    if( NULL != papi_sde_check_overflow_status_ptr )
        (*papi_sde_check_overflow_status_ptr)(cntr_uniq_id, latest);
}

inline int
sdei_set_timer_for_overflow(void){
    if( NULL != papi_sde_set_timer_for_overflow_ptr )
        return (*papi_sde_set_timer_for_overflow_ptr)();
    return -1;
}

/*
  The folling function will look for symbols from libpapi.so. If the application
  that linked against libsde has used the static PAPI library (libpapi.a)
  then dlsym will fail to find them, but the __attribute__((__common__)) should
  do the trick.
*/
static inline void obtain_papi_symbols(void){
    char *err;
    int dlsym_err = 0;

    // In case of static linking the function pointers will be automatically set
    // by the linker and the dlopen()/dlsym() would fail at runtime, so we want to
    // check if the linker has done its magic first.
    if( (NULL != papi_sde_check_overflow_status_ptr) &&
        (NULL != papi_sde_set_timer_for_overflow_ptr)
      ){
        return;
    }

    (void)dlerror(); // Clear the internal string so we can diagnose errors later on.

    void *handle = dlopen(NULL, RTLD_NOW|RTLD_GLOBAL);
    if( NULL != (err = dlerror()) ){
        SDEDBG("obtain_papi_symbols(): %s\n",err);
        dlsym_err = 1;
        return;
    }

    // We need this function to inform the SDE component in libpapi about the value of created counters.
    papi_sde_check_overflow_status_ptr = dlsym(handle, "papi_sde_check_overflow_status");
    DLSYM_CHECK(papi_sde_check_overflow_status);

    papi_sde_set_timer_for_overflow_ptr = dlsym(handle, "papi_sde_set_timer_for_overflow");
    DLSYM_CHECK(papi_sde_set_timer_for_overflow);

    if( !dlsym_err ){
        SDEDBG("obtain_papi_symbols(): All symbols from libpapi.so have been successfully acquired.\n");
    }

    return;
}


/*************************************************************************/
/* API Functions for libraries.                                          */
/*************************************************************************/

/** This function initializes SDE internal data-structures for an individual
  software library and returns an opaque handle to these structures.
  @param[in] name_of_library -- (const char *) library name.
  @param[out] sde_handle -- (papi_handle_t) opaque pointer to sde structure for initialized library.
  */
papi_handle_t
papi_sde_init(const char *name_of_library)
{
    papisde_library_desc_t *tmp_lib;

    papisde_control_t *gctl = sdei_get_global_struct();
    if(gctl->disabled)
        return NULL;

    // We have to emulate PAPI's SUBDBG to get the same behavior
    _sde_be_verbose = (NULL != getenv("PAPI_VERBOSE"));
    char *tmp= getenv("PAPI_DEBUG");
    if( (NULL != tmp) && (0 != strlen(tmp)) && (strstr(tmp, "SUBSTRATE") || strstr(tmp, "ALL")) ){
        _sde_debug = 1;
    }

    SDEDBG("Registering library: '%s'\n", name_of_library);

    obtain_papi_symbols();

    // Lock before we read and/or modify the global structures.
    sde_lock();

    // Put the actual work in a different function so we call it from other
    // places.  We have to do this because we cannot call
    // papi_sde_init() from places in the code which already call
    // lock()/unlock(), or we will end up with deadlocks.
    tmp_lib = (papisde_library_desc_t *)do_sde_init(name_of_library, gctl);

    sde_unlock();

    SDEDBG("Library '%s' has been registered.\n",name_of_library);

    return tmp_lib;
}

/** This function disables SDE activity for a specific library, or for all libraries
  that use SDEs until papi_sde_enable() is called.
  @param[in] handle -- (papi_handle_t) opaque pointer to sde structure for a specific library.
                       If NULL then SDEs will be disabled at a global level.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
  */
int
papi_sde_disable( papi_handle_t handle ){

    sde_lock();
    papisde_control_t *gctl = sdei_get_global_struct();

    // If the caller did not specify a library, then disable all SDEs.
    if( NULL == handle ){
        gctl->disabled = 1;
    }else{
        // else disable the specified library.
        papisde_library_desc_t *lib_handle = (papisde_library_desc_t *) handle;
        lib_handle->disabled = 1;
    }
    sde_unlock();
    return SDE_OK;
}

/** This function enables SDE activity for a specific library, or for all libraries
  that use SDEs.
  @param[in] handle -- (papi_handle_t) opaque pointer to sde structure for a specific library.
                       If NULL then SDEs will be enabled at a global level. Note that if
                       SDEs for a specific library have been explicitly disabled, then they
                       must be explicitly enabled passing that libary's handle. Calling
                       papi_sde_enabled(NULL) will only enable SDEs at the global level.
                       It will not recursivelly enable SDEs for individual libraries.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
  */
int
papi_sde_enable( papi_handle_t handle ){

    sde_lock();
    papisde_control_t *gctl = sdei_get_global_struct();

    // If the caller did not specify a library, then disable all SDEs.
    if( NULL == handle ){
        gctl->disabled = 0;
    }else{
        // else disable the specified library.
        papisde_library_desc_t *lib_handle = (papisde_library_desc_t *) handle;
        lib_handle->disabled = 0;
    }
    sde_unlock();
    return SDE_OK;
}

/** This function frees all SDE internal data-structures for an individual
  software library including all memory allocated by the counters of that library.
  @param[in] handle -- (papi_handle_t) opaque pointer to sde structure for initialized library.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
  */
int
papi_sde_shutdown( papi_handle_t handle ){
    papisde_library_desc_t *lib_handle, *tmp_lib, *next_lib, *prev_lib;
    int i;

    lib_handle = (papisde_library_desc_t *) handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==lib_handle) || lib_handle->disabled || (NULL==gctl) || gctl->disabled)
        return SDE_OK;

    SDEDBG("papi_sde_shutdown(): for library '%s'.\n", lib_handle->libraryName);

    sde_lock();

    sde_counter_t *all_lib_counters;
    int item_cnt = ht_to_array(lib_handle->lib_counters, &all_lib_counters);

    for(i=0; i<item_cnt; i++){
        char *cntr_name = all_lib_counters[i].name;
        sdei_delete_counter(lib_handle, cntr_name);
    }

    // We don't need the serialized array any more. Besides, the pointers inside
    // its elements have _not_ been copied, so they are junk by now, since we
    // deleted the counters.
    free(all_lib_counters);

    // Keep the `gctl` struct consistent
    // 1. If the lib head is this one, just set to next (could be NULL)
    // 2. Otherwise, find the prev_lib and set prev_lib->next = next_lib
    next_lib = lib_handle->next;
    if (gctl->lib_list_head == lib_handle) {
        gctl->lib_list_head = next_lib;
    } else {
        prev_lib = NULL;
        tmp_lib = gctl->lib_list_head;
        while (tmp_lib != lib_handle && tmp_lib != NULL)
        {
            prev_lib = tmp_lib;
            tmp_lib = tmp_lib->next;
        }
        if (prev_lib != NULL) {
            prev_lib->next = next_lib;
        }
    }

    free(lib_handle->libraryName);
    free(lib_handle);

    sde_unlock();
    return SDE_OK;
}


/** This function registers an event name and counter within the SDE
  data structure attached to the handle. A default description for an event is
  synthesized from the library name and the event name when they are registered.
  @param[in] handle -- pointer (of opaque type papi_handle_t) to sde structure for an individual library.
  @param[in] event_name -- (const char *) name of the event.
  @param[in] cntr_mode -- (int) the mode of the counter (one of: PAPI_SDE_RO, PAPI_SDE_RW and one of: PAPI_SDE_DELTA, PAPI_SDE_INSTANT).
  @param[in] cntr_type -- (int) the type of the counter (PAPI_SDE_long_long, PAPI_SDE_int, PAPI_SDE_double, PAPI_SDE_float).
  @param[in] counter -- pointer to a variable that stores the value for the event.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
  */
int
papi_sde_register_counter( papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter )
{
    papisde_library_desc_t *lib_handle;
    int ret_val = SDE_OK;
    cntr_class_specific_t cntr_union;

    if( NULL != event_name )
        SDEDBG("Prepaing to register counter: '%s'.\n", event_name);

    lib_handle = (papisde_library_desc_t *) handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==lib_handle) || lib_handle->disabled || (NULL==gctl) || gctl->disabled)
        return SDE_OK;

    cntr_union.cntr_basic.data = counter;

    sde_lock();
    ret_val = sdei_setup_counter_internals( lib_handle, event_name, cntr_mode, cntr_type, CNTR_CLASS_REGISTERED, cntr_union );
    sde_unlock();

    return ret_val;
}

/** This function registers an event name and (caller provided) callback function
  within the SDE data structure attached to the handle.
  A default description for an event is
  synthesized from the library name and the event name when they are registered.
  @param[in] handle -- pointer (of opaque type papi_handle_t) to sde structure for an individual library.
  @param[in] event_name -- (const char *) name of the event.
  @param[in] cntr_mode -- (int) the mode of the counter (one of: PAPI_SDE_RO, PAPI_SDE_RW and one of: PAPI_SDE_DELTA, PAPI_SDE_INSTANT).
  @param[in] cntr_type -- (int) the type of the counter (PAPI_SDE_long_long, PAPI_SDE_int, PAPI_SDE_double, PAPI_SDE_float).
  @param[in] fp_counter -- pointer to a callback function that SDE will call when PAPI_read/stop/accum is called.
  @param[in] param -- (void *) opaque parameter that will be passed to the callback function every time it's called.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
  */
int
papi_sde_register_counter_cb( papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, papi_sde_fptr_t callback, void *param )
{
    papisde_library_desc_t *lib_handle;
    int ret_val = SDE_OK;
    cntr_class_specific_t cntr_union;

    if( NULL != event_name )
        SDEDBG("Prepaing to register fp_counter: '%s'.\n", event_name);

    lib_handle = (papisde_library_desc_t *) handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==lib_handle) || lib_handle->disabled || (NULL==gctl) || gctl->disabled)
        return SDE_OK;

    cntr_union.cntr_cb.callback = callback;
    cntr_union.cntr_cb.param = param;

    sde_lock();
    ret_val = sdei_setup_counter_internals( lib_handle, event_name, cntr_mode, cntr_type, CNTR_CLASS_CB, cntr_union );
    sde_unlock();

    return ret_val;
}

/** This function unregisters (removes) an event name and counter from the SDE data structures.
  @param[in] handle -- pointer (of opaque type papi_handle_t) to sde structure for an individual library.
  @param[in] event_name -- (const char *) name of the event that is being unregistered.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
  */
int
papi_sde_unregister_counter( papi_handle_t handle, const char *event_name)
{
    papisde_library_desc_t *lib_handle;
    int error;
    char *full_event_name;
    int ret_val;

    SDEDBG("Preparing to unregister counter: '%s'.\n",event_name);

    lib_handle = (papisde_library_desc_t *) handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==lib_handle) || lib_handle->disabled || (NULL==gctl) || gctl->disabled)
        return SDE_OK;

    if( NULL == lib_handle->libraryName ){
        SDE_ERROR("papi_sde_unregister_counter(): 'handle' is clobbered. Unable to unregister counter.");
        return SDE_EINVAL;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = (char *)malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    SDEDBG("Unregistering counter: '%s' from SDE library: %s.\n", full_event_name, lib_handle->libraryName);

    // After this point we will be modifying data structures, so we need to acquire a lock.
    // This function has multiple exist points. If you add more, make sure you unlock before each one of them.
    sde_lock();

    error = sdei_delete_counter( lib_handle, full_event_name );
    // Check if we found a registered counter, or if it never existed.
    if( error ){
        SDE_ERROR("Counter '%s' has not been registered by library '%s'.", full_event_name, lib_handle->libraryName);
        free(full_event_name);
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    // We will not use the name beyond this point
    free(full_event_name);
    ret_val = SDE_OK;
fn_exit:
    sde_unlock();
    return ret_val;
}


/** This function optionally replaces an event's default description with a
  description provided by the library developer within the SDE data structure
  attached to the handle.
  @param[in] handle -- pointer (of opaque type papi_handle_t) to sde structure for an individual library.
  @param[in] event_name -- (const char *) name of the event.
  @param[in] event_description -- (const char *) description of the event.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
  */
int
papi_sde_describe_counter( void *handle, const char *event_name, const char *event_description )
{
    sde_counter_t *tmp_item;
    papisde_library_desc_t *lib_handle;
    char *full_event_name;
    int ret_val;

    lib_handle = (papisde_library_desc_t *) handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==lib_handle) || lib_handle->disabled || (NULL==gctl) || gctl->disabled)
        return SDE_OK;

    if( NULL == lib_handle->libraryName ){
        SDE_ERROR("papi_sde_describe_counter(): 'handle' is clobbered. Unable to add description for counter.");
        return SDE_EINVAL;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = (char *)malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    // After this point we will be modifying data structures, so we need to acquire a lock.
    // This function has multiple exist points. If you add more, make sure you unlock before each one of them.
    sde_lock();

    tmp_item = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);
    if( NULL != tmp_item ){
        tmp_item->description = strdup(event_description);
        free(full_event_name);
        ret_val = SDE_OK;
        goto fn_exit;
    }
    SDEDBG("papi_sde_describe_counter() Event: '%s' is not registered in SDE library: '%s'\n", full_event_name, lib_handle->libraryName);
    // We will not use the name beyond this point
    free(full_event_name);
    ret_val = SDE_EINVAL;
fn_exit:
    sde_unlock();
    return ret_val;
}



/** This function adds an event counter to a group. A group is created automatically
    the first time a counter is added to it.
  @param[in] handle -- pointer (of opaque type papi_handle_t) to sde structure for an individual library.
  @param[in] group_name -- (const char *) name of the group.
  @param[in] group_flags -- (uint32_t) one of PAPI_SDE_SUM, PAPI_SDE_MAX, PAPI_SDE_MIN to define how the members of the group will be used to compute the group's value.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
  */
int
papi_sde_add_counter_to_group(papi_handle_t handle, const char *event_name, const char *group_name, uint32_t group_flags)
{
    papisde_library_desc_t *lib_handle;
    sde_counter_t *tmp_item, *tmp_group;
    uint32_t cntr_group_uniq_id;
    char *full_event_name, *full_group_name;
    int ret_val;

    lib_handle = (papisde_library_desc_t *) handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==lib_handle) || lib_handle->disabled || (NULL==gctl) || gctl->disabled)
        return SDE_OK;

    SDEDBG("Adding counter: %s into group %s\n",event_name, group_name);

    if( NULL == lib_handle->libraryName ){
        SDE_ERROR("papi_sde_add_counter_to_group(): 'handle' is clobbered. Unable to add counter to group.");
        return SDE_EINVAL;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = (char *)malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    // After this point we will be modifying data structures, so we need to acquire a lock.
    // This function has multiple exist points. If you add more, make sure you unlock before each one of them.
    sde_lock();

    // Check to make sure that the event is already registered. This is not the place to create a placeholder.
    tmp_item = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);
    if( NULL == tmp_item ){
        SDE_ERROR("papi_sde_add_counter_to_group(): Unable to find counter: '%s'.",full_event_name);
        free(full_event_name);
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    // We will not use the name beyond this point
    free(full_event_name);

    str_len = strlen(lib_handle->libraryName)+strlen(group_name)+2+1; // +2 for "::" and +1 for '\0'
    full_group_name = (char *)malloc(str_len*sizeof(char));
    snprintf(full_group_name, str_len, "%s::%s", lib_handle->libraryName, group_name);

    // Check to see if the group exists already. Otherwise we need to create it.
    tmp_group = ht_lookup_by_name(lib_handle->lib_counters, full_group_name);
    if( NULL == tmp_group ){

        papisde_control_t *gctl = sdei_get_global_struct();

        // We use the current number of registered events as the uniq id of the counter group, and we
        // increment it because counter groups are treated as real counters by the outside world.
        // They are first class citizens.
        cntr_group_uniq_id = gctl->num_reg_events++;
        gctl->num_live_events++;

        SDEDBG("%s line %d: Unique ID for new counter group = %d\n", __FILE__, __LINE__, cntr_group_uniq_id);

        tmp_group = (sde_counter_t *)calloc(1, sizeof(sde_counter_t));
        tmp_group->cntr_class = CNTR_CLASS_GROUP;
        tmp_group->glb_uniq_id = cntr_group_uniq_id;
        // copy the name because we will free the malloced space further down in this function.
        tmp_group->name = strdup(full_group_name);
        // make a copy here, because we will free() the 'name' and the 'description' separately.
        tmp_group->description = strdup( full_group_name );
        tmp_group->which_lib = lib_handle;
        tmp_group->u.cntr_group.group_flags = group_flags;

        (void)ht_insert(lib_handle->lib_counters, ht_hash_name(full_group_name), tmp_group);
        (void)ht_insert(gctl->all_reg_counters, ht_hash_id(cntr_group_uniq_id), tmp_group);

    }else{
        if( NULL == tmp_group->u.cntr_group.group_head ){
            if( CNTR_CLASS_PLACEHOLDER == tmp_group->cntr_class ){
                tmp_group->cntr_class = CNTR_CLASS_GROUP;
            }else{
                SDE_ERROR("papi_sde_add_counter_to_group(): Found an empty counter group: '%s'. This might indicate that a cleanup routine is not doing its job.", group_name);
            }

        }

        // make sure the caller is not trying to change the flags of the group after it has been created.
        if( tmp_group->u.cntr_group.group_flags != group_flags ){
            SDE_ERROR("papi_sde_add_counter_to_group(): Attempting to add counter '%s' to counter group '%s' with incompatible group flags.", event_name, group_name);
            free(full_group_name);
            ret_val = SDE_EINVAL;
            goto fn_exit;
        }
    }

    // Add the new counter to the group's head.
    papisde_list_entry_t *new_head = (papisde_list_entry_t *)calloc(1, sizeof(papisde_list_entry_t));
    new_head->item = tmp_item;
    new_head->next = tmp_group->u.cntr_group.group_head;
    tmp_group->u.cntr_group.group_head = new_head;
    if( SDE_OK != sdei_inc_ref_count(tmp_item) ){
        SDE_ERROR("papi_sde_add_counter_to_group(): Error while adding counter '%s' to counter group: '%s'.", tmp_item->name, group_name);
    }

    free(full_group_name);
    ret_val = SDE_OK;
fn_exit:
    sde_unlock();
    return ret_val;
}

/**

  This function creates a counter whose memory is allocated and managed by libsde,
  in contrast with papi_sde_register_counter(), which works with counters that are managed
  by the user library that is calling this function.
  This counter can only by modified via the functions papi_sde_inc_counter()
  and papi_sde_reset_counter(). This has two benefits over a counter which
  lives inside the user library and is modified directly by that library:
  A) Our counter and the modifying API is guaranteed to be thread safe.
  B) Since libsde knows about each change in the value of the counter,
     overflowing is accurate.
  However, this approach has higher overhead than executing "my_cntr += value" inside
  a user library.

  @param[in] handle -- pointer (of opaque type papi_handle_t) to sde structure for an individual library.
  @param[in] event_name -- (const char *) name of the event.
  @param[in] cntr_mode -- (int) the mode of the counter (one of: PAPI_SDE_RO, PAPI_SDE_RW and one of: PAPI_SDE_DELTA, PAPI_SDE_INSTANT).
  @param[out] cntr_handle -- address of a pointer in which libsde will store a handle to the newly created counter.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
*/
int
papi_sde_create_counter( papi_handle_t handle, const char *event_name, int cntr_mode, void **cntr_handle )
{
    int ret_val;
    long long int *counter_data;
    char *full_event_name;
    papisde_library_desc_t *lib_handle;
    sde_counter_t *cntr;
    cntr_class_specific_t cntr_union;

    lib_handle = (papisde_library_desc_t *) handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==lib_handle) || lib_handle->disabled || (NULL==gctl) || gctl->disabled)
        return SDE_OK;

    if( NULL != event_name )
        SDEDBG("Preparing to create counter: '%s'.\n", event_name);

    sde_lock();

    if( NULL == lib_handle->libraryName ){
        SDE_ERROR("papi_sde_create_counter(): 'handle' is clobbered. Unable to create counter.");
        return SDE_EINVAL;
    }

    SDEDBG("Adding created counter: '%s' with mode: '%d' in SDE library: %s.\n", event_name, cntr_mode, lib_handle->libraryName);

    // Created counters use memory allocated by libsde, not the user library.
    counter_data = (long long int *)calloc(1, sizeof(long long int));
    cntr_union.cntr_basic.data = counter_data;

    ret_val = sdei_setup_counter_internals( lib_handle, event_name, cntr_mode, PAPI_SDE_long_long, CNTR_CLASS_CREATED, cntr_union );
    if( SDE_OK != ret_val ){
        goto fn_exit;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = (char *)malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    cntr = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);
    if(NULL == cntr) {
        SDEDBG("Logging counter '%s' not properly inserted in SDE library '%s'\n", full_event_name, lib_handle->libraryName);
        free(full_event_name);
        ret_val = SDE_ECMP;
        goto fn_exit;
    }

    if( NULL != cntr_handle ){
        *(sde_counter_t **)cntr_handle = cntr;
    }

    free(full_event_name);
    ret_val = SDE_OK;
fn_exit:
    sde_unlock();
    return ret_val;
}


// The following function works only for counters created using papi_sde_create_counter().
int
papi_sde_inc_counter( papi_handle_t cntr_handle, long long int increment)
{
    long long int *ptr;
    sde_counter_t *tmp_cntr;
    int ret_val;

    tmp_cntr = (sde_counter_t *)cntr_handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==tmp_cntr) || (NULL==tmp_cntr->which_lib) || tmp_cntr->which_lib->disabled || (NULL==gctl) || gctl->disabled)
        return SDE_OK;

    sde_lock();

    if( !IS_CNTR_CREATED(tmp_cntr) || (NULL == tmp_cntr->u.cntr_basic.data) ){
        SDE_ERROR("papi_sde_inc_counter(): 'cntr_handle' is clobbered. Unable to modify value of counter.");
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    if( PAPI_SDE_long_long != tmp_cntr->cntr_type ){
        SDE_ERROR("papi_sde_inc_counter(): Counter is not of type \"long long int\" and cannot be modified using this function.");
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    SDEDBG("Preparing to increment counter: '%s::%s' by %lld.\n", tmp_cntr->which_lib->libraryName, tmp_cntr->name, increment);

    ptr = tmp_cntr->u.cntr_basic.data;
    *ptr += increment;

    sdei_check_overflow_status(tmp_cntr->glb_uniq_id, *ptr);

    ret_val = SDE_OK;
fn_exit:
    sde_unlock();
    return ret_val;
}

/*
  @param[in] handle -- pointer (of opaque type papi_handle_t) to sde structure for an individual library.
  @param[in] cset_name -- (const char *) name of the counting set.
  @param[out] cset_handle -- address of a pointer in which libsde will store a handle to the newly created counting set.
  @param[out] -- (int) the return value is SDE_OK on success, or an error code on failure.
*/
int
papi_sde_create_counting_set( papi_handle_t handle, const char *cset_name, void **cset_handle )
{
    int ret_val;
    sde_counter_t *tmp_cset_handle;
    char *full_cset_name;
    papisde_library_desc_t *lib_handle;
    cntr_class_specific_t cntr_union;

    SDEDBG("papi_sde_create_counting_set()\n");

    lib_handle = (papisde_library_desc_t *) handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==lib_handle) || lib_handle->disabled || (NULL==gctl) || gctl->disabled)
        return SDE_OK;

    if( NULL != cset_name )
        SDEDBG("Preparing to create counting set: '%s'.\n", cset_name);

    if( NULL == lib_handle->libraryName ){
        SDE_ERROR("papi_sde_create_counting_set(): 'handle' is clobbered. Unable to create counting set.");
        return SDE_EINVAL;
    }

    SDEDBG("Adding counting set: '%s' in SDE library: %s.\n", cset_name, lib_handle->libraryName);

    // Allocate the structure for the hash table.
    cntr_union.cntr_cset.data = (cset_hash_table_t *)calloc(1,sizeof(cset_hash_table_t));
    if( NULL == cntr_union.cntr_cset.data )
        return SDE_ENOMEM;

    ret_val = sdei_setup_counter_internals( lib_handle, cset_name, PAPI_SDE_DELTA|PAPI_SDE_RO, PAPI_SDE_long_long, CNTR_CLASS_CSET, cntr_union );
    if( SDE_OK != ret_val )
        return ret_val;

    size_t str_len = strlen(lib_handle->libraryName)+strlen(cset_name)+2+1; // +2 for "::" and +1 for '\0'
    full_cset_name = (char *)malloc(str_len*sizeof(char));
    snprintf(full_cset_name, str_len, "%s::%s", lib_handle->libraryName, cset_name);

    tmp_cset_handle = ht_lookup_by_name(lib_handle->lib_counters, full_cset_name);
    if(NULL == tmp_cset_handle) {
        SDEDBG("Recorder '%s' not properly inserted in SDE library '%s'\n", full_cset_name, lib_handle->libraryName);
        free(full_cset_name);
        return SDE_ECMP;
    }

    if( NULL != cset_handle ){
        *(sde_counter_t **)cset_handle = tmp_cset_handle;
    }

    free(full_cset_name);

    return SDE_OK;
}

int
papi_sde_counting_set_remove( void *cset_handle, size_t hashable_size, const void *element, uint32_t type_id)
{
    sde_counter_t *tmp_cset;
    int ret_val = SDE_OK;

    tmp_cset = (sde_counter_t *)cset_handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==tmp_cset) || (NULL==tmp_cset->which_lib) || tmp_cset->which_lib->disabled || (NULL==gctl) || gctl->disabled)
        return SDE_OK;

    sde_lock();

    if( !IS_CNTR_CSET(tmp_cset) || (NULL == tmp_cset->u.cntr_cset.data) ){
        SDE_ERROR("papi_sde_counting_set_remove(): Counting set is clobbered. Unable to remove element.");
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    SDEDBG("Preparing to remove element from counting set: '%s::%s'.\n", tmp_cset->which_lib->libraryName, tmp_cset->name);
    ret_val = cset_remove_elem(tmp_cset->u.cntr_cset.data, hashable_size, element, type_id);

fn_exit:
    sde_unlock();
    return ret_val;
}



int
papi_sde_counting_set_insert( void *cset_handle, size_t element_size, size_t hashable_size, const void *element, uint32_t type_id)
{
    sde_counter_t *tmp_cset;
    int ret_val = SDE_OK;

    tmp_cset = (sde_counter_t *)cset_handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==tmp_cset) || (NULL==tmp_cset->which_lib) || tmp_cset->which_lib->disabled || (NULL==gctl) || gctl->disabled)
        return SDE_OK;

    sde_lock();

    if( !IS_CNTR_CSET(tmp_cset) || (NULL == tmp_cset->u.cntr_cset.data) ){
        SDE_ERROR("papi_sde_counting_set_insert(): Counting set is clobbered. Unable to insert element.");
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    SDEDBG("Preparing to insert element in counting set: '%s::%s'.\n", tmp_cset->which_lib->libraryName, tmp_cset->name);
    ret_val = cset_insert_elem(tmp_cset->u.cntr_cset.data, element_size, hashable_size, element, type_id);

fn_exit:
    sde_unlock();
    return ret_val;
}


int
papi_sde_create_recorder( papi_handle_t handle, const char *event_name, size_t typesize, int (*cmpr_func_ptr)(const void *p1, const void *p2), void **record_handle )
{
    int ret_val, i;
    sde_counter_t *tmp_rec_handle;
    cntr_class_specific_t aux_cntr_union;
    char *aux_event_name;
    size_t str_len;
    char *full_event_name;
    cntr_class_specific_t cntr_union;
#define _SDE_MODIFIER_COUNT 6
    const char *modifiers[_SDE_MODIFIER_COUNT] = {":CNT",":MIN",":Q1",":MED",":Q3",":MAX"};
    // Add a NULL pointer for symmetry with the 'modifiers' vector, since the modifier ':CNT' does not have a function pointer.
    long long (*func_ptr_vec[_SDE_MODIFIER_COUNT])(void *) = {NULL, sdei_compute_min, sdei_compute_q1, sdei_compute_med, sdei_compute_q3, sdei_compute_max};

    papisde_library_desc_t *lib_handle = (papisde_library_desc_t *)handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==lib_handle) || lib_handle->disabled || (NULL==gctl) || gctl->disabled)
        return SDE_OK;

    sde_lock();

    if( NULL == lib_handle->libraryName ){
        SDE_ERROR("papi_sde_create_recorder(): 'handle' is clobbered. Unable to create recorder.");
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    SDEDBG("Preparing to create recorder: '%s' with typesize: '%d' in SDE library: %s.\n", event_name, (int)typesize, lib_handle->libraryName);

    // Allocate the "Exponential Storage" structure for the recorder data and meta-data.
    cntr_union.cntr_recorder.data = (recorder_data_t *)calloc(1,sizeof(recorder_data_t));
    // Allocate the first chunk of recorder data.
    cntr_union.cntr_recorder.data->ptr_array[0] = malloc(EXP_CONTAINER_MIN_SIZE*typesize);
    cntr_union.cntr_recorder.data->total_entries = EXP_CONTAINER_MIN_SIZE;
    cntr_union.cntr_recorder.data->typesize = typesize;
    cntr_union.cntr_recorder.data->used_entries = 0;

    ret_val = sdei_setup_counter_internals( lib_handle, event_name, PAPI_SDE_DELTA|PAPI_SDE_RO, PAPI_SDE_long_long, CNTR_CLASS_RECORDER, cntr_union );
    if( SDE_OK != ret_val )
        return ret_val;

    str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = (char *)malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    tmp_rec_handle = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);
    if(NULL == tmp_rec_handle) {
        SDEDBG("Recorder '%s' not properly inserted in SDE library '%s'\n", full_event_name, lib_handle->libraryName);
        free(full_event_name);
        ret_val = SDE_ECMP;
        goto fn_exit;
    }

    // We will not use the name beyond this point
    free(full_event_name);

    if( NULL != record_handle ){
      *(sde_counter_t **)record_handle = tmp_rec_handle;
    }

    // At this point we are done creating the recorder and we will create the additional events which will appear as modifiers of the recorder.
    str_len = 0;
    for(i=0; i<_SDE_MODIFIER_COUNT; i++){
        size_t tmp_len = strlen(modifiers[i]);
        if( tmp_len > str_len )
            str_len = tmp_len;
    }
    str_len += strlen(event_name)+1;
    aux_event_name = (char *)calloc(str_len, sizeof(char));

    snprintf(aux_event_name, str_len, "%s%s", event_name, modifiers[0]);
    SDEDBG("papi_sde_create_recorder(): Preparing to register aux counter: '%s' in SDE library: %s.\n", aux_event_name, lib_handle->libraryName);

    // The field that holds the number of used entries in the recorder structure will become the counter of the new auxiliary event.
    aux_cntr_union.cntr_basic.data = &(tmp_rec_handle->u.cntr_recorder.data->used_entries);
    ret_val = sdei_setup_counter_internals( lib_handle, (const char *)aux_event_name, PAPI_SDE_INSTANT|PAPI_SDE_RO, PAPI_SDE_long_long, CNTR_CLASS_REGISTERED, aux_cntr_union );
    if( SDE_OK != ret_val ){
        SDEDBG("papi_sde_create_recorder(): Registration of aux counter: '%s' in SDE library: %s FAILED.\n", aux_event_name, lib_handle->libraryName);
        free(aux_event_name);
        goto fn_exit;
    }

    // If the caller passed NULL as the function pointer, then they do _not_ want the quantiles. Otherwise, create them.
    if( NULL != cmpr_func_ptr ){
        for(i=1; i<_SDE_MODIFIER_COUNT; i++){
            sde_sorting_params_t *sorting_params;

            sorting_params = (sde_sorting_params_t *)malloc(sizeof(sde_sorting_params_t)); // This will be free()-ed by papi_sde_unregister_counter()
            sorting_params->recording = tmp_rec_handle;
            sorting_params->cmpr_func_ptr = cmpr_func_ptr;

            snprintf(aux_event_name, str_len, "%s%s", event_name, modifiers[i]);

            SDEDBG("papi_sde_create_recorder(): Preparing to register aux fp counter: '%s' in SDE library: %s.\n", aux_event_name, lib_handle->libraryName);
            // clear the previous entries;
            memset(&aux_cntr_union, 0, sizeof(aux_cntr_union));
            aux_cntr_union.cntr_cb.callback = func_ptr_vec[i];
            aux_cntr_union.cntr_cb.param = sorting_params;
            ret_val = sdei_setup_counter_internals(lib_handle, (const char *)aux_event_name, PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_long_long, CNTR_CLASS_CB, aux_cntr_union );
            if( SDE_OK != ret_val ){
                SDEDBG("papi_sde_create_recorder(): Registration of aux counter: '%s' in SDE library: %s FAILED.\n", aux_event_name, lib_handle->libraryName);
                free(aux_event_name);
                goto fn_exit;
            }
        }
    }

    free(aux_event_name);
    ret_val = SDE_OK;
fn_exit:
    sde_unlock();
    return ret_val;
}

int
papi_sde_record( void *record_handle, size_t typesize, const void *value)
{
    sde_counter_t *tmp_rcrd;
    int ret_val;

    tmp_rcrd = (sde_counter_t *)record_handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==tmp_rcrd) || (NULL==tmp_rcrd->which_lib) || tmp_rcrd->which_lib->disabled || (NULL==gctl) || gctl->disabled)
        return SDE_OK;

    SDEDBG("Preparing to record value of size %lu at address: %p\n",typesize, value);

    sde_lock();

    if( !IS_CNTR_RECORDER(tmp_rcrd) || (NULL == tmp_rcrd->u.cntr_recorder.data) ){
        SDE_ERROR("papi_sde_record(): 'record_handle' is clobbered. Unable to record value.");
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    ret_val = exp_container_insert_element(tmp_rcrd->u.cntr_recorder.data, typesize, value);

fn_exit:
    sde_unlock();
    return ret_val;
}


// This function neither frees the allocated, nor does it zero it. It only resets the counter of used entries so that
// the allocated space can be resused (and overwritten) by future calls to record().
int
papi_sde_reset_recorder( void *record_handle )
{
    sde_counter_t *tmp_rcrdr;
    int ret_val;

    tmp_rcrdr = (sde_counter_t *)record_handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==tmp_rcrdr) || (NULL==tmp_rcrdr->which_lib) || tmp_rcrdr->which_lib->disabled || (NULL==gctl) || gctl->disabled)
        return SDE_OK;

    sde_lock();

    if( !IS_CNTR_RECORDER(tmp_rcrdr) || NULL == tmp_rcrdr->u.cntr_recorder.data ){
        SDE_ERROR("papi_sde_reset_recorder(): 'record_handle' is clobbered. Unable to reset recorder.");
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    // NOTE: do _not_ free the chunks and do _not_ reset "cntr_recorder.data->total_entries"
    tmp_rcrdr->u.cntr_recorder.data->used_entries = 0;
    free( tmp_rcrdr->u.cntr_recorder.data->sorted_buffer );
    tmp_rcrdr->u.cntr_recorder.data->sorted_buffer = NULL;
    tmp_rcrdr->u.cntr_recorder.data->sorted_entries = 0;

    ret_val = SDE_OK;
fn_exit:
    sde_unlock();
    return ret_val;
}


// The following function works only for counters created using papi_sde_create_counter().
int
papi_sde_reset_counter( void *cntr_handle )
{
    long long int *ptr;
    sde_counter_t *tmp_cntr;
    int ret_val;

    tmp_cntr = (sde_counter_t *)cntr_handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==tmp_cntr) || (NULL==tmp_cntr->which_lib) || tmp_cntr->which_lib->disabled || (NULL==gctl) || gctl->disabled)
        return SDE_OK;

    sde_lock();

    if( !IS_CNTR_CREATED(tmp_cntr) ){
        SDE_ERROR("papi_sde_reset_counter(): Counter is not created by PAPI, so it cannot be reset.");
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    ptr = (long long int *)(tmp_cntr->u.cntr_basic.data);

    if( NULL == ptr ){
        SDE_ERROR("papi_sde_reset_counter(): Counter structure is clobbered. Unable to reset value of counter.");
        ret_val = SDE_EINVAL;
        goto fn_exit;
    }

    *ptr = 0; // Reset the counter.

    ret_val = SDE_OK;
fn_exit:
    sde_unlock();
    return ret_val;
}


/** This function finds the handle associated with a created counter, or a recorder,
  given the library handle and the event name.
  @param[in] handle -- (void *) pointer to sde structure for an individual
  library
  @param[in] event_name -- name of the event
  */
void
*papi_sde_get_counter_handle( void *handle, const char *event_name)
{
    sde_counter_t *counter_handle;
    papisde_library_desc_t *lib_handle;
    char *full_event_name;

    lib_handle = (papisde_library_desc_t *) handle;
    papisde_control_t *gctl = _papisde_global_control;
    if( (NULL==lib_handle) || lib_handle->disabled || (NULL==gctl) || gctl->disabled)
        return NULL;

    if( NULL == lib_handle->libraryName ){
        SDE_ERROR("papi_sde_get_counter_handle(): 'handle' is clobbered.");
        return NULL;
    }

    size_t str_len = strlen(lib_handle->libraryName)+strlen(event_name)+2+1; // +2 for "::" and +1 for '\0'
    full_event_name = (char *)malloc(str_len*sizeof(char));
    snprintf(full_event_name, str_len, "%s::%s", lib_handle->libraryName, event_name);

    // After this point we will be accessing shared data structures, so we need to acquire a lock.
    sde_lock();
    counter_handle = ht_lookup_by_name(lib_handle->lib_counters, full_event_name);
    sde_unlock();

    free(full_event_name);

    return counter_handle;
}


/*************************************************************************/
/* Utility Functions.                                                    */
/*************************************************************************/

int
papi_sde_compare_long_long(const void *p1, const void *p2){
    long long n1, n2;
    n1 = *(long long *)p1;
    n2 = *(long long *)p2;

    if( n1 < n2 ) return -1;
    if( n1 > n2 ) return 1;
    return 0;
}

int
papi_sde_compare_int(const void *p1, const void *p2){
    int n1, n2;
    n1 = *(int *)p1;
    n2 = *(int *)p2;

    if( n1 < n2 ) return -1;
    if( n1 > n2 ) return 1;
    return 0;
}

int
papi_sde_compare_double(const void *p1, const void *p2){
    double n1, n2;
    n1 = *(double *)p1;
    n2 = *(double *)p2;

    if( n1 < n2 ) return -1;
    if( n1 > n2 ) return 1;
    return 0;
}

int
papi_sde_compare_float(const void *p1, const void *p2){
    float n1, n2;
    n1 = *(float *)p1;
    n2 = *(float *)p2;

    if( n1 < n2 ) return -1;
    if( n1 > n2 ) return 1;
    return 0;
}

#define _SDE_CMP_MIN 0
#define _SDE_CMP_MAX 1

// This function returns a "long long" which contains a pointer to the
// data element that corresponds to the edge (min/max), so that it works
// for all types of data, not only integers.
static inline long long sdei_compute_edge(void *param, int which_edge){
    void *edge = NULL, *edge_copy;
    long long elem_cnt;
    long long current_size, cumul_size = 0;
    void *src;
    int i, chunk;
    size_t typesize;
    sde_counter_t *rcrd;
    int (*cmpr_func_ptr)(const void *p1, const void *p2);


    rcrd = ((sde_sorting_params_t *)param)->recording;
    elem_cnt = rcrd->u.cntr_recorder.data->used_entries;
    typesize = rcrd->u.cntr_recorder.data->typesize;

    cmpr_func_ptr = ((sde_sorting_params_t *)param)->cmpr_func_ptr;

    // The return value is supposed to be a pointer to the correct element, therefore zero
    // is a NULL pointer, which should tell the caller that there was a problem.
    if( (0 == elem_cnt) || (NULL == cmpr_func_ptr) )
        return 0;

    // If there is a sorted (contiguous) buffer, but it's stale, we need to free it.
    // The value of elem_cnt (rcrd->u.cntr_recorder.data->used_entries) can
    // only increase, or be reset to zero, but when it is reset to zero
    // (by papi_sde_reset_recorder()) the buffer will be freed (by the same function).
    if( (NULL != rcrd->u.cntr_recorder.data->sorted_buffer) &&
        (rcrd->u.cntr_recorder.data->sorted_entries < elem_cnt) ){

        free( rcrd->u.cntr_recorder.data->sorted_buffer );
        rcrd->u.cntr_recorder.data->sorted_buffer = NULL;
        rcrd->u.cntr_recorder.data->sorted_entries = 0;
    }

    // Check if a sorted contiguous buffer is already there. If there is, return
    // the first or last element (for MIN, or MAX respectively).
    if( NULL != rcrd->u.cntr_recorder.data->sorted_buffer ){
        if( _SDE_CMP_MIN == which_edge )
            edge = rcrd->u.cntr_recorder.data->sorted_buffer;
        if( _SDE_CMP_MAX == which_edge )
            edge = (char *)(rcrd->u.cntr_recorder.data->sorted_buffer) + (elem_cnt-1)*typesize;
    }else{
        // Make "edge" point to the beginning of the first chunk.
        edge = rcrd->u.cntr_recorder.data->ptr_array[0];
        if ( NULL == edge )
            return 0;

        cumul_size = 0;
        for(chunk=0; chunk<EXP_CONTAINER_ENTRIES; chunk++){
           current_size = ((long long)1<<chunk) * EXP_CONTAINER_MIN_SIZE;
           src = rcrd->u.cntr_recorder.data->ptr_array[chunk];

           for(i=0; (i < (elem_cnt-cumul_size)) && (i < current_size); i++){
               void *next_elem = (char *)src + i*typesize;
               int rslt = cmpr_func_ptr(next_elem, edge);

               // If the new element is smaller than the current min and we are looking for the min, then keep it.
               if( (rslt < 0) && (_SDE_CMP_MIN == which_edge) )
                   edge = next_elem;
               // If the new element is larger than the current max and we are looking for the max, then keep it.
               if( (rslt > 0) && (_SDE_CMP_MAX == which_edge) )
                   edge = next_elem;
           }

           cumul_size += current_size;

           if( cumul_size >= elem_cnt )
               break;
        }
    }

    // We might free the sorted_buffer (when it becomes stale), so we can't return "edge".
    // Therefore, we allocate fresh space for the resulting element and copy it there.
    // Since we do not know when the user will use this pointer, we will not be able
    // to free it, so it is the responibility of the user (who calls PAPI_read()) to
    // free this memory.
    edge_copy = malloc( 1 * typesize);
    memcpy(edge_copy, edge, 1 * typesize);

    // A pointer is guaranteed to fit inside a long long, so cast it and return a long long.
    return (long long)edge_copy;
}



// This function returns a "long long" which contains a pointer to the
// data element that corresponds to the edge (min/max), so that it works
// for all types of data, not only integers.

// NOTE: This function allocates memory for one element and returns a pointer
// to this memory. Since we do not know when the user will use this pointer, we
// can not free it, so it is the responibility of the user
// (the code that calls PAPI_read()) to free this memory.
static inline long long sdei_compute_quantile(void *param, int percent){
    long long quantile, elem_cnt;
    void *result_data;
    size_t typesize;
    sde_counter_t *rcrd;
    int (*cmpr_func_ptr)(const void *p1, const void *p2);

    rcrd = ((sde_sorting_params_t *)param)->recording;
    elem_cnt = rcrd->u.cntr_recorder.data->used_entries;
    typesize = rcrd->u.cntr_recorder.data->typesize;

    cmpr_func_ptr = ((sde_sorting_params_t *)param)->cmpr_func_ptr;

    // The return value is supposed to be a pointer to the correct element, therefore zero
    // is a NULL pointer, which should tell the caller that there was a problem.
    if( (0 == elem_cnt) || (NULL == cmpr_func_ptr) )
        return 0;

    // If there is a sorted (contiguous) buffer, but it's stale, we need to free it.
    // The value of elem_cnt (rcrd->u.cntr_recorder.data->used_entries) can
    // only increase, or be reset to zero, but when it is reset to zero
    // (by papi_sde_reset_recorder()) the buffer will be freed (by the same function).
    if( (NULL != rcrd->u.cntr_recorder.data->sorted_buffer) &&
        (rcrd->u.cntr_recorder.data->sorted_entries < elem_cnt) ){

        free( rcrd->u.cntr_recorder.data->sorted_buffer );
        rcrd->u.cntr_recorder.data->sorted_buffer = NULL;
        rcrd->u.cntr_recorder.data->sorted_entries = 0;
    }

    // Check if a sorted buffer is already there. If there isn't, allocate one.
    if( NULL == rcrd->u.cntr_recorder.data->sorted_buffer ){
        rcrd->u.cntr_recorder.data->sorted_buffer = malloc(elem_cnt * typesize);
        exp_container_to_contiguous(rcrd->u.cntr_recorder.data, rcrd->u.cntr_recorder.data->sorted_buffer);
        // We set this field so we can test later to see if the allocated buffer is stale.
        rcrd->u.cntr_recorder.data->sorted_entries = elem_cnt;
    }
    void *sorted_buffer = rcrd->u.cntr_recorder.data->sorted_buffer;

    qsort(sorted_buffer, elem_cnt, typesize, cmpr_func_ptr);
    void *tmp_ptr = (char *)sorted_buffer + typesize*((elem_cnt*percent)/100);

    // We might free the sorted_buffer (when it becomes stale), so we can't return "tmp_ptr".
    // Therefore, we allocate fresh space for the resulting element and copy it there.
    // Since we do not know when the user will use this pointer, we will not be able
    // to free it, so it is the responibility of the user (who calls PAPI_read()) to
    // free this memory.
    result_data = malloc(typesize);
    memcpy(result_data, tmp_ptr, typesize);

    // convert the pointer into a long long so we can return it.
    quantile = (long long)result_data;

    return quantile;
}


static long long sdei_compute_q1(void *param){
    return sdei_compute_quantile(param, 25);
}
static long long sdei_compute_med(void *param){
    return sdei_compute_quantile(param, 50);
}
static long long sdei_compute_q3(void *param){
    return sdei_compute_quantile(param, 75);
}
static long long sdei_compute_min(void *param){
    return sdei_compute_edge(param, _SDE_CMP_MIN);
}
static long long sdei_compute_max(void *param){
    return sdei_compute_edge(param, _SDE_CMP_MAX);
}


