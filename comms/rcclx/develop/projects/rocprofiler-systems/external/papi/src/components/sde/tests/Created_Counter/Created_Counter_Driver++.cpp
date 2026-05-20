#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "papi.h"
#include "papi_test.h"

void cclib_init(void);
void cclib_do_work(void);
void cclib_do_more_work(void);
void setup_PAPI(int *event_set);

long long int epsilon_v[10] = {14LL, 11LL, 8LL, 13LL, 8LL, 10LL, 12LL, 11LL, 6LL, 8LL};
int be_verbose = 0;

int main(int argc, char **argv){
    int i, ret, event_set = PAPI_NULL;
    int discrepancies = 0;
    long long counter_values[1] = {0};

    if( (argc > 1) && !strcmp(argv[1], "-verbose") )
        be_verbose = 1;

    cclib_init();

    setup_PAPI(&event_set);

    // --- Start PAPI
    if((ret=PAPI_start(event_set)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_start", ret );
    }

    for(i=0; i<10; i++){

        cclib_do_work();

        // --- Read the event counters _and_ reset them
        if((ret=PAPI_accum(event_set, counter_values)) != PAPI_OK){
            test_fail( __FILE__, __LINE__, "PAPI_accum", ret );
        }
        if( be_verbose ) printf("Epsilon count in cclib_do_work(): %lld\n",counter_values[0]);
        if( counter_values[0] != epsilon_v[i] ){
            discrepancies++;
        }
        counter_values[0] = 0;

    }

    // --- Stop PAPI
    if((ret=PAPI_stop(event_set, counter_values)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_stop", ret );
    }

    if( !discrepancies )
        test_pass(__FILE__);
    else
        test_fail( __FILE__, __LINE__, "SDE counter values are wrong!", 0 );

    // The following "return" is dead code, because both test_pass() and test_fail() call exit(),
    // however, we need it to prevent compiler warnings.
    return 0;
}


void setup_PAPI(int *event_set){
    int ret;

    if((ret=PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT){
        test_fail( __FILE__, __LINE__, "PAPI_library_init", ret );
    }

    if((ret=PAPI_create_eventset(event_set)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_create_eventset", ret );
    }

    if((ret=PAPI_add_named_event(*event_set, "sde:::CPP_Lib_With_CC::epsilon_count")) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_add_named_event", ret );
    }

    return;
}

