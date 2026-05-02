#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "sde_lib.h"

// API functions (FORTRAN 77 friendly).
papi_handle_t papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct);
void gamum_init_(void);
void gamum_unreg_(void);
void gamum_do_work_(void);

// The following counter is a global variable that can be directly
// modified by programs linking with this library.
long long int gamum_cnt_i1;

// The following counters are hidden to programs linking with
// this library, so they can not be directly modified.
static double cnt_d1, cnt_d2, cnt_d3, cnt_d4, cnt_d5;
static long long int cnt_i2, cnt_i3;
static double cnt_rm1, cnt_rm2;
static papi_handle_t handle;
static void *cntr_handle;

// For internal use only.
static papi_handle_t gamum_papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct);

static const char *event_names[2] = {
    "event_with_characters____ __up______to_______60_bytes",
    "event_with_very_long_name_which_is_meant_to_exceed_128_bytes_or_in_other_words_the_size_of_two_cache_lines_so_we_see_if_it_makes_a_difference_in_performance"
};


void gamum_init_(void){
    cnt_d1 = cnt_d2 = cnt_d3 = cnt_d4 = cnt_d5 = 1;
    gamum_cnt_i1 = cnt_i2 = 0;
    papi_sde_fptr_struct_t fptr_struct;

    POPULATE_SDE_FPTR_STRUCT( fptr_struct );
    (void)gamum_papi_sde_hook_list_events(&fptr_struct);

    return;
}

papi_handle_t papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct){
    return gamum_papi_sde_hook_list_events(fptr_struct);
}

papi_handle_t gamum_papi_sde_hook_list_events( papi_sde_fptr_struct_t *fptr_struct){
    handle = fptr_struct->init("Gamum");
    fptr_struct->register_counter(handle, "rm1", PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_double, &cnt_rm1);
    fptr_struct->register_counter(handle, "ev1", PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_double, &cnt_d1);
    fptr_struct->add_counter_to_group(handle, "ev1", "group0", PAPI_SDE_SUM);
    fptr_struct->register_counter(handle, "ev2", PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_double, &cnt_d2);
    fptr_struct->register_counter(handle, "ev3", PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_double, &cnt_d3);
    fptr_struct->register_counter(handle, "ev4", PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_double, &cnt_d4);
    fptr_struct->add_counter_to_group(handle, "ev4", "group0", PAPI_SDE_SUM);
    fptr_struct->register_counter(handle, "ev5", PAPI_SDE_RO|PAPI_SDE_INSTANT, PAPI_SDE_double, &cnt_d5);
    fptr_struct->register_counter(handle, "rm2", PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_double, &cnt_rm2);
    fptr_struct->register_counter(handle, "i1",  PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_long_long, &gamum_cnt_i1);
    fptr_struct->register_counter(handle, "i2",  PAPI_SDE_RO|PAPI_SDE_DELTA,   PAPI_SDE_long_long, &cnt_i2);

    fptr_struct->create_counter(handle, "papi_counter", PAPI_SDE_RO|PAPI_SDE_INSTANT, &cntr_handle );

    fptr_struct->register_counter(handle, event_names[0], PAPI_SDE_RO|PAPI_SDE_DELTA, PAPI_SDE_long_long, &cnt_i3);

    return handle;
}

void gamum_unreg_(void){
    papi_sde_unregister_counter(handle, "rm1");
    papi_sde_unregister_counter(handle, "rm2");
}

void gamum_do_work_(void){
    cnt_d1 += 0.1;
    cnt_d2 += 0.111;
    cnt_d3 += 0.2;
    cnt_d4 += 0.222;
    cnt_d5 += 0.3;
    gamum_cnt_i1 += 6;
    cnt_i2 += 101;
    cnt_i3 += 33;
    papi_sde_inc_counter(cntr_handle, 5);
    papi_sde_inc_counter(papi_sde_get_counter_handle(handle, "papi_counter"), 1);
}
