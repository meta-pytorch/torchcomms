#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "papi.h"
#include "papi_test.h"

#define EV_THRESHOLD 10

void cclib_init(void);
void cclib_do_work(void);
void cclib_do_more_work(void);
void setup_PAPI(int *event_set, int threshold);

int remaining_handler_invocations = 22;
int be_verbose = 0;

int main(int argc, char **argv){
    int i, ret, event_set = PAPI_NULL;
    long long counter_values[1] = {0};

    if( (argc > 1) && !strcmp(argv[1], "-verbose") )
        be_verbose = 1;

    cclib_init();

    setup_PAPI(&event_set, EV_THRESHOLD);

    // --- Start PAPI
    if((ret=PAPI_start(event_set)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_start", ret );
    }

    for(i=0; i<4; i++){

        cclib_do_work();

        // --- Read the event counters _and_ reset them
        if((ret=PAPI_accum(event_set, counter_values)) != PAPI_OK){
            test_fail( __FILE__, __LINE__, "PAPI_accum", ret );
        }
        if( be_verbose ) printf("Epsilon count in cclib_do_work(): %lld\n",counter_values[0]);
        counter_values[0] = 0;

        cclib_do_more_work();

        // --- Read the event counters _and_ reset them
        if((ret=PAPI_accum(event_set, counter_values)) != PAPI_OK){
            test_fail( __FILE__, __LINE__, "PAPI_accum", ret );
        }
        if( be_verbose ) printf("Epsilon count in cclib_do_more_work(): %lld\n",counter_values[0]);
        counter_values[0] = 0;

    }

    // --- Stop PAPI
    if((ret=PAPI_stop(event_set, counter_values)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_stop", ret );
    }

    if( remaining_handler_invocations <= 1 ) // Let's allow for up to one missed signal, or race condition.
        test_pass(__FILE__);
    else
        test_fail( __FILE__, __LINE__, "SDE overflow handler was not invoked as expected!", 0 );

    // The following "return" is dead code, because both test_pass() and test_fail() call exit(),
    // however, we need it to prevent compiler warnings.
    return 0;
}


void overflow_handler(int event_set, void *address, long long overflow_vector, void *context){
    char event_name[PAPI_MAX_STR_LEN];
    int ret, *event_codes, event_index, number=1;

    (void)address;
    (void)context;

    if( (ret = PAPI_get_overflow_event_index(event_set, overflow_vector, &event_index, &number)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_get_overflow_event_index", ret );
    }

    number = event_index+1;
    event_codes = (int *)calloc(number, sizeof(int));

    if( (ret = PAPI_list_events( event_set, event_codes, &number)) != PAPI_OK ){
        test_fail( __FILE__, __LINE__, "PAPI_list_events", ret );
    }

    if( (ret=PAPI_event_code_to_name(event_codes[event_index], event_name)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_event_code_to_name", ret );
    }
    free(event_codes);

    if( be_verbose ){
        printf("Event \"%s\" at index: %d exceeded its threshold again.\n",event_name, event_index);
        fflush(stdout);
    }

    if( !strcmp(event_name, "sde:::Lib_With_CC::epsilon_count") || !event_index )
        remaining_handler_invocations--;


    return;
}

void setup_PAPI(int *event_set, int threshold){
    int ret, event_code;

    if((ret=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT){
        test_fail( __FILE__, __LINE__, "PAPI_library_init", ret );
    }

    if((ret=PAPI_create_eventset(event_set)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_create_eventset", ret );
    }

    if((ret=PAPI_event_name_to_code("sde:::Lib_With_CC::epsilon_count", &event_code)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_event_name_to_code", ret );
    }

    if((ret=PAPI_add_event(*event_set, event_code)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_add_event", ret );
    }

    if((ret = PAPI_overflow(*event_set, event_code, threshold, PAPI_OVERFLOW_HARDWARE, overflow_handler)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_overflow", ret );
    }

    return;
}

