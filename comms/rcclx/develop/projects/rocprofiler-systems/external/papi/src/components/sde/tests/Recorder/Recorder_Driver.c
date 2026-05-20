#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "papi.h"
#include "papi_test.h"

void recorder_init_(void);
void recorder_do_work_(void);
void setup_PAPI(int *event_set);

long long int expectation[10] = {20674LL, 50122LL, 112964LL, 32904LL, 101565LL, 56993LL, 58388LL, 122543LL, 62312LL, 52433LL};

int main(int argc, char **argv){
    int i, j, ret, event_set = PAPI_NULL;
    int discrepancies = 0;
    int be_verbose = 0;
    long long counter_values[2];

    if( (argc > 1) && !strcmp(argv[1], "-verbose") )
        be_verbose = 1;

    recorder_init_();

    setup_PAPI(&event_set);

    // --- Start PAPI
    if((ret=PAPI_start(event_set)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_start", ret );
    }

    for(i=0; i<10; i++){

        recorder_do_work_();

        // --- read the event counters
        if((ret=PAPI_read(event_set, counter_values)) != PAPI_OK){
            test_fail( __FILE__, __LINE__, "PAPI_read", ret );
        }

        long long *ptr = (long long *)counter_values[1];

        if( be_verbose ){
            printf("The number of recordings is: %lld (ptr is: %p)\n",counter_values[0],(void *)counter_values[1]);
            for(j=0; j<counter_values[0]; j++){
                printf("%lld ",*(ptr+j));
            }
            printf("\n");
        }

        free(ptr);
    }

    // --- Stop PAPI
    if((ret=PAPI_stop(event_set, counter_values)) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_stop", ret );
    }

    if( counter_values[0] != 10 ){
        discrepancies++;
    }
    long long *ptr = (long long *)counter_values[1];
    for(j=0; j<10; j++){
        if( *(ptr+j) != expectation[j] ){
            discrepancies++;
        }
    }
    free(ptr);

    if( !discrepancies )
        test_pass(__FILE__);
    else
        test_fail( __FILE__, __LINE__, "SDE values in recorder are wrong!", 0 );

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

    if((ret=PAPI_add_named_event(*event_set, "sde:::Lib_With_Recorder::simple_recording:CNT")) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_add_named_event", ret );
    }

    if((ret=PAPI_add_named_event(*event_set, "sde:::Lib_With_Recorder::simple_recording")) != PAPI_OK){
        test_fail( __FILE__, __LINE__, "PAPI_add_named_event", ret );
    }

    return;
}

