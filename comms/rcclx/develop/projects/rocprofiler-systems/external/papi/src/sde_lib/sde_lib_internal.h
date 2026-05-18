/**
 * @file    sde_lib_internal.h
 * @author  Anthony Danalis
 *          adanalis@icl.utk.edu
 *
 * @ingroup papi_components
 */

#if !defined(PAPI_SDE_LIB_INTERNAL_H)
#define PAPI_SDE_LIB_INTERNAL_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <dlfcn.h>
#include <assert.h>
#include "sde_lib.h"

#define EXP_CONTAINER_ENTRIES 52
#define EXP_CONTAINER_MIN_SIZE 2048

#define PAPISDE_HT_SIZE 512

#define is_readonly(_X_)  (PAPI_SDE_RO      == ((_X_)&0x0F))
#define is_readwrite(_X_) (PAPI_SDE_RW      == ((_X_)&0x0F))
#define is_delta(_X_)     (PAPI_SDE_DELTA   == ((_X_)&0xF0))
#define is_instant(_X_)   (PAPI_SDE_INSTANT == ((_X_)&0xF0))

typedef struct sde_counter_s sde_counter_t;
typedef struct sde_sorting_params_s sde_sorting_params_t;
typedef struct papisde_list_entry_s papisde_list_entry_t;
typedef struct papisde_library_desc_s papisde_library_desc_t;
typedef struct papisde_control_s papisde_control_t;
typedef struct recorder_data_s recorder_data_t;

/** This global variable is defined in sde_lib.c and points to the head of the control state list **/
extern papisde_control_t *_papisde_global_control;

// _SDE_HASH_BUCKET_COUNT_ should not be a power of two, and even better it should be a prime.
#if defined(SDE_HASH_SMALL) // 7.4KB storage
  #define _SDE_HASH_BUCKET_COUNT_ 61
  #define _SDE_HASH_BUCKET_WIDTH_ 5
#else                        // 124KB storage (5222 elements)
  #define _SDE_HASH_BUCKET_COUNT_ 373
  #define _SDE_HASH_BUCKET_WIDTH_ 14
#endif

// defining SDE_HASH_IS_FUZZY to 1 will make the comparisons operation of the hash table
// (which is used in the "counting sets") faster, but inaccurate. As a result, some input
// elements might collide onto the same hash table entry, even if they are different.
// If speed is more important than accurate counting for your library, then setting
// SDE_HASH_IS_FUZZY to 1 is recommented.
#define SDE_HASH_IS_FUZZY 0

typedef struct cset_hash_decorated_object_s {
    uint32_t count;
    uint32_t type_id;
    size_t type_size;
    void *ptr;
} cset_hash_decorated_object_t;

/*
typedef struct sde_list_object_s sde_list_object_t;
struct sde_list_object_s {
    sde_hash_decorated_object_t object;
    sde_list_object_t *next;
};
*/

typedef struct cset_hash_bucket_s {
    uint32_t occupied;
    uint64_t keys[_SDE_HASH_BUCKET_WIDTH_];
    cset_hash_decorated_object_t objects[_SDE_HASH_BUCKET_WIDTH_];
} cset_hash_bucket_t;

typedef struct cset_hash_table_s {
    cset_hash_bucket_t buckets[_SDE_HASH_BUCKET_COUNT_];
    cset_list_object_t *overflow_list;
} cset_hash_table_t;


/* Hash table entry */
struct papisde_list_entry_s {
    sde_counter_t *item;
    papisde_list_entry_t *next;
};

struct recorder_data_s{
   void *ptr_array[EXP_CONTAINER_ENTRIES];
   long long total_entries;
   long long used_entries;
   size_t typesize;
   void *sorted_buffer;
   long long sorted_entries;
};

typedef struct cntr_class_basic_s {
   void *data;
} cntr_class_basic_t;

typedef struct cntr_class_callback_s {
   papi_sde_fptr_t callback;
   void *param;
} cntr_class_callback_t;

typedef struct cntr_class_recorder_s {
   recorder_data_t *data;
} cntr_class_recorder_t;

typedef struct cntr_class_cset_s {
   cset_hash_table_t *data;
} cntr_class_cset_t;

typedef struct cntr_class_group_s {
   papisde_list_entry_t *group_head;
   uint32_t group_flags;
} cntr_class_group_t;

typedef union cntr_class_specific_u{
   cntr_class_basic_t cntr_basic;
   cntr_class_callback_t cntr_cb;
   cntr_class_recorder_t cntr_recorder;
   cntr_class_cset_t cntr_cset;
   cntr_class_group_t cntr_group;
} cntr_class_specific_t;

struct sde_counter_s {
   uint32_t glb_uniq_id;
   char *name;
   char *description;
   uint32_t cntr_class;
   cntr_class_specific_t u;
   long long int previous_data;
   int overflow;
   int cntr_type;
   int cntr_mode;
   int ref_count;
   papisde_library_desc_t *which_lib;
};

struct sde_sorting_params_s{
   sde_counter_t *recording;
   int (*cmpr_func_ptr)(const void *p1, const void *p2);
};

enum CNTR_CLASS{
    CNTR_CLASS_REGISTERED = 0x1,
    CNTR_CLASS_CREATED = 0x2,
    CNTR_CLASS_BASIC = 0x3, // both previous types combined.
    CNTR_CLASS_CB = 0x4,
    CNTR_CLASS_RECORDER = 0x8,
    CNTR_CLASS_CSET = 0x10,
    CNTR_CLASS_PLACEHOLDER = 0x1000,
    CNTR_CLASS_GROUP = 0x2000
};

#define IS_CNTR_REGISTERED(_CNT) ( CNTR_CLASS_REGISTERED == (_CNT)->cntr_class )
#define IS_CNTR_CREATED(_CNT) ( CNTR_CLASS_CREATED == (_CNT)->cntr_class )
#define IS_CNTR_BASIC(_CNT) ( CNTR_CLASS_BASIC & (_CNT)->cntr_class )
#define IS_CNTR_CALLBACK(_CNT) ( CNTR_CLASS_CB == (_CNT)->cntr_class )
#define IS_CNTR_RECORDER(_CNT) ( CNTR_CLASS_RECORDER == (_CNT)->cntr_class )
#define IS_CNTR_CSET(_CNT) ( CNTR_CLASS_CSET == (_CNT)->cntr_class )
#define IS_CNTR_PLACEHOLDER(_CNT) ( CNTR_CLASS_PLACEHOLDER == (_CNT)->cntr_class )
#define IS_CNTR_GROUP(_CNT) ( CNTR_CLASS_GROUP == (_CNT)->cntr_class )

/* This type describes one library. This is the type of the handle returned by papi_sde_init(). */
struct papisde_library_desc_s {
    char* libraryName;
    papisde_list_entry_t lib_counters[PAPISDE_HT_SIZE];
    uint32_t disabled;
    papisde_library_desc_t *next;
};

/* One global variable of this type holds pointers to all other SDE meta-data */
struct papisde_control_s {
    uint32_t num_reg_events; /* This number only increases, so it can be used as a uniq id */
    uint32_t num_live_events; /* This number decreases at unregister() */
    uint32_t disabled;
    papisde_library_desc_t *lib_list_head;
    uint32_t activeLibCount;
    papisde_list_entry_t all_reg_counters[PAPISDE_HT_SIZE];
};

int sdei_setup_counter_internals( papi_handle_t handle, const char *event_name, int cntr_mode, int cntr_type, enum CNTR_CLASS cntr_class, cntr_class_specific_t cntr_union );
int sdei_delete_counter(papisde_library_desc_t* lib_handle, const char *name);
int sdei_inc_ref_count(sde_counter_t *counter);
int sdei_read_counter_group( sde_counter_t *counter, long long int *rslt_ptr );
void sdei_counting_set_to_list( void *cset_handle, cset_list_object_t **list_head );
int sdei_read_and_update_data_value( sde_counter_t *counter, long long int previous_value, long long int *rslt_ptr );
int sdei_hardware_write( sde_counter_t *counter, long long int new_value );
int sdei_set_timer_for_overflow(void);

papisde_control_t *sdei_get_global_struct(void);
sde_counter_t *ht_lookup_by_id(papisde_list_entry_t *hash_table, uint32_t uniq_id);
sde_counter_t *ht_lookup_by_name(papisde_list_entry_t *hash_table, const char *name);
sde_counter_t *ht_delete(papisde_list_entry_t *hash_table, int ht_key, uint32_t uniq_id);
void ht_insert(papisde_list_entry_t *hash_table, int ht_key, sde_counter_t *sde_counter);
int ht_to_array(papisde_list_entry_t *hash_table, sde_counter_t **rslt_array);
uint32_t ht_hash_name(const char *str);
uint32_t ht_hash_id(uint32_t uniq_id);
papi_handle_t do_sde_init(const char *name_of_library, papisde_control_t *gctl);
sde_counter_t *allocate_and_insert(papisde_control_t *gctl, papisde_library_desc_t* lib_handle, const char *name, uint32_t uniq_id, int cntr_mode, int cntr_type, enum CNTR_CLASS cntr_class, cntr_class_specific_t cntr_union);
void exp_container_to_contiguous(recorder_data_t *exp_container, void *cont_buffer);
int exp_container_insert_element(recorder_data_t *exp_container, size_t typesize, const void *value);
void exp_container_init(sde_counter_t *handle, size_t typesize);
void papi_sde_counting_set_to_list(void *cset_handle, cset_list_object_t **list_head);
int cset_insert_elem(cset_hash_table_t *hash_ptr, size_t element_size, size_t hashable_size, const void *element, uint32_t type_id);
int cset_remove_elem(cset_hash_table_t *hash_ptr, size_t hashable_size, const void *element, uint32_t type_id);
cset_list_object_t *cset_to_list(cset_hash_table_t *hash_ptr);
int cset_delete(cset_hash_table_t *hash_ptr);

#pragma GCC visibility push(default)

int sde_ti_reset_counter( uint32_t );
int sde_ti_read_counter( uint32_t, long long int * );
int sde_ti_write_counter( uint32_t, long long );
int sde_ti_name_to_code( const char *, uint32_t * );
int sde_ti_is_simple_counter( uint32_t );
int sde_ti_is_counter_set_to_overflow( uint32_t );
int sde_ti_set_counter_overflow( uint32_t, int );
char * sde_ti_get_event_name( int );
char * sde_ti_get_event_description( int );
int sde_ti_get_num_reg_events( void );
int sde_ti_shutdown( void );

#pragma GCC visibility pop

/*************************************************************************/
/* Hashing code below copied verbatim from the "fast-hash" project:      */
/* https://github.com/ztanml/fast-hash                                   */
/*************************************************************************/

// Compression function for Merkle-Damgard construction.
#define mix(h) ({                                \
                 (h) ^= (h) >> 23;               \
                 (h) *= 0x2127599bf4325c37ULL;   \
                 (h) ^= (h) >> 47; })


static inline uint64_t fasthash64(const void *buf, size_t len, uint64_t seed)
{
    const uint64_t    m = 0x880355f21e6d1965ULL;
    const uint64_t *pos = (const uint64_t *)buf;
    const uint64_t *end = pos + (len / 8);
    const uint32_t *pos2;
    uint64_t h = seed ^ (len * m);
    uint64_t v;

    while (pos != end) {
        v  = *pos++;
        h ^= mix(v);
        h *= m;
    }

    pos2 = (const uint32_t*)pos;
    v = 0;

    switch (len & 7) {
    case 7: v ^= (uint64_t)pos2[6] << 48;
            /* fall through */
    case 6: v ^= (uint64_t)pos2[5] << 40;
            /* fall through */
    case 5: v ^= (uint64_t)pos2[4] << 32;
            /* fall through */
    case 4: v ^= (uint64_t)pos2[3] << 24;
            /* fall through */
    case 3: v ^= (uint64_t)pos2[2] << 16;
            /* fall through */
    case 2: v ^= (uint64_t)pos2[1] << 8;
            /* fall through */
    case 1: v ^= (uint64_t)pos2[0];
        h ^= mix(v);
        h *= m;
    }

    return mix(h);
}

#endif // !defined(PAPI_SDE_LIB_INTERNAL_H)
