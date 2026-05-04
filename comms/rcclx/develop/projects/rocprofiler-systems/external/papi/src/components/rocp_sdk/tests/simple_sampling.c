#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <papi.h>
#include <papi_test.h>

#define NUM_EVENTS (12)

extern void launch_kernel(int device_id);
int eventset = PAPI_NULL;
volatile int gv=0;

const char *events[NUM_EVENTS] = {
    "rocp_sdk:::SQ_CYCLES:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=3",
    "rocp_sdk:::SQ_INSTS:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=0",
    "rocp_sdk:::SQ_INSTS:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=1",
    "rocp_sdk:::SQ_INSTS:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=2",
    "rocp_sdk:::SQ_INSTS:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=3",
    "rocp_sdk:::SQ_BUSY_CYCLES:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=0",
    "rocp_sdk:::SQ_BUSY_CYCLES:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=1",
    "rocp_sdk:::SQ_BUSY_CYCLES:device=0:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=2",
    "rocp_sdk:::SQ_BUSY_CYCLES:device=0:DIMENSION_INSTANCE=0",
    "rocp_sdk:::SQ_BUSY_CYCLES:device=1:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=0",
    "rocp_sdk:::SQ_BUSY_CYCLES:device=1:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=1",
    "rocp_sdk:::SQ_BUSY_CYCLES:device=1:DIMENSION_INSTANCE=0:DIMENSION_SHADER_ENGINE=2"
};


void *thread_main(void *arg){
    long long counters[NUM_EVENTS] = { 0 };
    while(0==gv){;}
    usleep(150*1000);
    for(int i=0; i<30; i++){
        printf("Sample: %2d\n", gv);
        fflush(stdout);
        PAPI_read(eventset, counters);
        for (int i = 0; i < NUM_EVENTS; ++i) {
            printf("%s: %.2lfM\n", events[i], (double)counters[i]/1e6);
            fflush(stdout);
        }
        printf("\n");
	fflush(stdout);
        usleep(30*1000);
        ++gv;
    }
    return NULL;
}


int main(int argc, char *argv[])
{
    int papi_errno;

    papi_errno = PAPI_library_init(PAPI_VER_CURRENT);
    if (papi_errno != PAPI_VER_CURRENT) {
        test_fail(__FILE__, __LINE__, "PAPI_library_init", papi_errno);
    }

    papi_errno = PAPI_create_eventset(&eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_create_eventset", papi_errno);
    }

    for (int i = 0; i < NUM_EVENTS; ++i) {
        papi_errno = PAPI_add_named_event(eventset, events[i]);
        if (papi_errno != PAPI_OK) {
            test_fail(__FILE__, __LINE__, "PAPI_add_named_event", papi_errno);
        }
    }

    long long counters[NUM_EVENTS] = { 0 };
    papi_errno = PAPI_start(eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_start", papi_errno);
    }

    pthread_t tid;
    pthread_create(&tid, NULL, thread_main, NULL);

    printf("---------------------  launch_kernel(0)\n");
    gv = 1;
    launch_kernel(0);

    usleep(20000);

    papi_errno = PAPI_read(eventset, counters);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_read", papi_errno);
    }
    printf("---------------------  PAPI_read()\n");

    for (int i = 0; i < NUM_EVENTS; ++i) {
        printf("%s: %.2lfM\n", events[i], (double)counters[i]/1e6);
    }

    papi_errno = PAPI_stop(eventset, counters);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_stop", papi_errno);
    }

    printf("---------------------  PAPI_stop()\n");

    for (int i = 0; i < NUM_EVENTS; ++i) {
        printf("%s: %.2lfM\n", events[i], (double)counters[i]/1e6);
    }
    
    papi_errno = PAPI_cleanup_eventset(eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_cleanup_eventset", papi_errno);
    }

    papi_errno = PAPI_destroy_eventset(&eventset);
    if (papi_errno != PAPI_OK) {
        test_fail(__FILE__, __LINE__, "PAPI_destroy_eventset", papi_errno);
    }

    PAPI_shutdown();
    test_pass(__FILE__);
    return 0;
}
