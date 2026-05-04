/**
 * @file    sde_lib.h
 * @author  Anthony Danalis
 *          adanalis@icl.utk.edu
 *
 * @ingroup papi_components
 *
 * @brief
 *  SDE prototypes and macros.
 */

#if !defined(PAPI_SDE_LIB_H)
#define PAPI_SDE_LIB_H

#include <stdio.h>
#include <stdint.h>
#include <stddef.h>
#include <stdarg.h>

#define PAPI_SDE_VERSION_NUMBER(_maj,_min) ( ((_maj)<<16) | (_min) )
#define PAPI_SDE_VERSION PAPI_SDE_VERSION_NUMBER(1,0)

#define PAPI_SDE_RO       0x00
#define PAPI_SDE_RW       0x01
#define PAPI_SDE_DELTA    0x00
#define PAPI_SDE_INSTANT  0x10

#define PAPI_SDE_long_long 0x0
#define PAPI_SDE_int       0x1
#define PAPI_SDE_double    0x2
#define PAPI_SDE_float     0x3

#define PAPI_SDE_SUM       0x0
#define PAPI_SDE_MAX       0x1
#define PAPI_SDE_MIN       0x2

// The following values have been defined such that they match the
// corresponding PAPI values from papi.h
#define SDE_OK          0     /**< No error */
#define SDE_EINVAL     -1     /**< Invalid argument */
#define SDE_ENOMEM     -2     /**< Insufficient memory */
#define SDE_ECMP       -4     /**< Not supported by component */
#define SDE_ENOEVNT    -7     /**< Event does not exist */
#define SDE_EMISC      -14    /**< Unknown error code */

#define register_fp_counter register_counter_cb
#define papi_sde_register_fp_counter papi_sde_register_counter_cb

#define destroy_counter unregister_counter
#define destroy_counting_set unregister_counter
#define papi_sde_destroy_counter papi_sde_unregister_counter
#define papi_sde_destroy_counting_set papi_sde_unregister_counter

#pragma GCC visibility push(default)

extern int _sde_be_verbose;
extern int _sde_debug;
#define SDEDBG(format, args...) { if(_sde_debug){fprintf(stderr,format, ## args);} }

static inline void SDE_ERROR( const char *format, ... ){
    va_list args;
    if ( _sde_be_verbose ) {
        va_start( args, format );
        fprintf( stderr, "PAPI SDE Error: " );
        vfprintf( stderr, format, args );
        fprintf( stderr, "\n" );
        va_end( args );
    }
}

#ifdef __cplusplus
extern "C" {
#endif

#define GET_FLOAT_SDE(x) *((float *)&x)
#define GET_DOUBLE_SDE(x) *((double *)&x)
/*
 * GET_SDE_RECORDER_ADDRESS() USAGE EXAMPLE:
 * If SDE recorder logs values of type 'double':
 *     double *ptr = GET_SDE_RECORDER_ADDRESS(papi_event_value[6], double);
 *     for (j=0; j<CNT; j++)
 *        printf("    %d: %.4e\n",j, ptr[j]);
 */
#define GET_SDE_RECORDER_ADDRESS(x,rcrd_type) ((rcrd_type *)x)

// The following type needs to be declared here, instead of
// sde_lib_internal.h because we need to expose it to the user.

typedef struct cset_list_object_s cset_list_object_t;
struct cset_list_object_s {
    uint32_t count;
    uint32_t type_id;
    size_t type_size;
    void *ptr;
    cset_list_object_t *next;
};

typedef long long int (*papi_sde_fptr_t)( void * );
typedef int (*papi_sde_cmpr_fptr_t)( void * );
typedef void * papi_handle_t;

typedef struct papi_sde_fptr_struct_s {
    papi_handle_t (*init)(const char *lib_name );
    int (*shutdown)(papi_handle_t handle);
    int (*register_counter)( papi_handle_t handle, const char *event_name, int mode, int type, void *counter );
    int (*register_counter_cb)( papi_handle_t handle, const char *event_name, int mode, int type, papi_sde_fptr_t callback, void *param );
    int (*unregister_counter)( papi_handle_t handle, const char *event_name );
    int (*describe_counter)( papi_handle_t handle, const char *event_name, const char *event_description );
    int (*add_counter_to_group)( papi_handle_t handle, const char *event_name, const char *group_name, uint32_t group_flags );
    int (*create_counter)( papi_handle_t handle, const char *event_name, int cntr_type, void **cntr_handle );
    int (*inc_counter)( papi_handle_t cntr_handle, long long int increment );
    int (*create_recorder)( papi_handle_t handle, const char *event_name, size_t typesize, int (*cmpr_func_ptr)(const void *p1, const void *p2), void **record_handle );
    int (*create_counting_set)( papi_handle_t handle, const char *cset_name, void **cset_handle );
    int (*counting_set_insert)( void *cset_handle, size_t element_size, size_t hashable_size, const void *element, uint32_t type_id );
    int (*counting_set_remove)( void *cset_handle, size_t hashable_size, const void *element, uint32_t type_id );
    int (*record)( void *record_handle, size_t typesize, const void *value );
    int (*reset_recorder)(void *record_handle );
    int (*reset_counter)( void *cntr_handle );
    void *(*get_counter_handle)(papi_handle_t handle, const char *event_name);
}papi_sde_fptr_struct_t;


papi_handle_t papi_sde_init(const char *name_of_library );
int papi_sde_shutdown(papi_handle_t handle);
int papi_sde_register_counter(papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, void *counter );
int papi_sde_register_counter_cb(papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, papi_sde_fptr_t callback, void *param );
int papi_sde_unregister_counter( void *handle, const char *event_name );
int papi_sde_describe_counter(papi_handle_t handle, const char *event_name, const char *event_description );
int papi_sde_add_counter_to_group(papi_handle_t handle, const char *event_name, const char *group_name, uint32_t group_flags );
int papi_sde_create_counter( papi_handle_t handle, const char *event_name, int cntr_mode, void **cntr_handle );
int papi_sde_inc_counter( void *cntr_handle, long long int increment );
int papi_sde_create_recorder( papi_handle_t handle, const char *event_name, size_t typesize, int (*cmpr_func_ptr)(const void *p1, const void *p2), void **record_handle );
int papi_sde_create_counting_set( papi_handle_t handle, const char *cset_name, void **cset_handle );
int papi_sde_counting_set_insert( void *cset_handle, size_t element_size, size_t hashable_size, const void *element, uint32_t type_id );
int papi_sde_counting_set_remove( void *cset_handle, size_t hashable_size, const void *element, uint32_t type_id );
int papi_sde_record( void *record_handle, size_t typesize, const void *value );
int papi_sde_reset_recorder(void *record_handle );
int papi_sde_reset_counter( void *cntr_handle );
void *papi_sde_get_counter_handle( papi_handle_t handle, const char *event_name);

int papi_sde_compare_long_long(const void *p1, const void *p2);
int papi_sde_compare_int(const void *p1, const void *p2);
int papi_sde_compare_double(const void *p1, const void *p2);
int papi_sde_compare_float(const void *p1, const void *p2);

papi_handle_t papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct);

#define POPULATE_SDE_FPTR_STRUCT( _A_ ) do{\
    _A_.init = papi_sde_init;\
    _A_.shutdown = papi_sde_shutdown;\
    _A_.register_counter = papi_sde_register_counter;\
    _A_.register_counter_cb = papi_sde_register_counter_cb;\
    _A_.unregister_counter = papi_sde_unregister_counter;\
    _A_.describe_counter = papi_sde_describe_counter;\
    _A_.add_counter_to_group = papi_sde_add_counter_to_group;\
    _A_.create_counter = papi_sde_create_counter;\
    _A_.inc_counter = papi_sde_inc_counter;\
    _A_.create_recorder = papi_sde_create_recorder;\
    _A_.create_counting_set = papi_sde_create_counting_set;\
    _A_.record = papi_sde_record;\
    _A_.reset_recorder = papi_sde_reset_recorder;\
    _A_.reset_counter = papi_sde_reset_counter;\
    _A_.get_counter_handle = papi_sde_get_counter_handle;\
}while(0)

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif // !defined(PAPI_SDE_LIB_H)
