#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include "papi.h"
#include "branch.h"

volatile int iter_count, global_var1, global_var2;
volatile int result;
volatile unsigned int b, z1, z2, z3, z4;

void branch_driver(char *papi_event_name, int junk, hw_desc_t *hw_desc, char* outdir){
    int papi_eventset = PAPI_NULL;
    int i, iter, sz, ret_val, max_iter = 16*1024;
    long long int cnt;
    double avg, round;
    FILE* ofp_papi;
    const char *sufx = ".branch";
    int l = strlen(outdir)+strlen(papi_event_name)+strlen(sufx);

    (void)hw_desc;

    char *papiFileName = (char *)calloc( 1+l, sizeof(char) );
    if (l != (sprintf(papiFileName, "%s%s%s", outdir, papi_event_name, sufx))) {
        goto error0;
    }
    if (NULL == (ofp_papi = fopen(papiFileName,"w"))) {
        fprintf(stderr, "Unable to open file %s.\n", papiFileName);
        goto error0;
    }

    // Initialize undecidible values for the BRNG macro.
    z1 = junk*7;
    z2 = (junk+4)/(junk+1);
    z3 = junk;
    z4 = (z3+z2)/z1;

    ret_val = PAPI_create_eventset( &papi_eventset );
    if (ret_val != PAPI_OK){
        goto error1;
    }

    ret_val = PAPI_add_named_event( papi_eventset, papi_event_name );
    if (ret_val != PAPI_OK){
        goto error1;
    }

    BRANCH_BENCH(1);
    BRANCH_BENCH(2);
    BRANCH_BENCH(3);
    BRANCH_BENCH(4);
    BRANCH_BENCH(4a);
    BRANCH_BENCH(4b);
    BRANCH_BENCH(5);
    BRANCH_BENCH(5a);
    BRANCH_BENCH(5b);
    BRANCH_BENCH(6);
    BRANCH_BENCH(7);

    if( result == 143526 ){
        printf("Random side effect\n");
    }

    ret_val = PAPI_cleanup_eventset( papi_eventset );
    if (ret_val != PAPI_OK ){
        goto error1;
    }
    ret_val = PAPI_destroy_eventset( &papi_eventset );
    if (ret_val != PAPI_OK ){
        goto error1;
    }

error1:
    fclose(ofp_papi);
error0:
    free(papiFileName);
    return;
}

long long int branch_char_b1(int size, int event_set){
    int retval;
    long long int value;

    if ( (retval=PAPI_start(event_set)) != PAPI_OK){
        return -1;
    }

    /*
        1.  Conditional EXECUTED = 2
        1.  Conditional RETIRED  = 2
        2.  Conditional TAKEN    = 1.5
        4.  Direct JUMP          = 0
        3.  Branch MISPREDICT    = 0
        5.  All Branches         = 2
    */

    iter_count = 1;
    global_var2 = 1;
    do{
        if ( iter_count < (size/2) ){
            global_var2 += 2;
        }
        BRNG();
        iter_count++;
    }while(iter_count<size);

    if ( (retval=PAPI_stop(event_set, &value)) != PAPI_OK){
        return -1;
    }

    return value;

}

long long int branch_char_b2(int size, int event_set){
    int retval;
    long long int value;

    if ( (retval=PAPI_start(event_set)) != PAPI_OK){
        return -1;
    }

    /*
        1.  Conditional EXECUTED = 2
        1.  Conditional RETIRED  = 2
        2.  Conditional TAKEN    = 1
        4.  Direct JUMP          = 0
        3.  Branch MISPREDICT    = 0
        5.  All Branches         = 2
    */
    iter_count = 1;
    global_var2 = 1;
    do{
        global_var2+=2;
        if ( iter_count < global_var2 ){
            global_var1+=2;
        }
        BRNG();
        iter_count++;
    }while(iter_count<size);


    if ( (retval=PAPI_stop(event_set, &value)) != PAPI_OK){
        return -1;
    }
    return value;

}

long long int branch_char_b3(int size, int event_set){
    int retval;
    long long int value;

    if ( (retval=PAPI_start(event_set)) != PAPI_OK){
        return -1;
    }

    /*
        1.  Conditional EXECUTED = 2
        1.  Conditional RETIRED  = 2
        2.  Conditional TAKEN    = 2
        4.  Direct JUMP          = 0
        3.  Branch MISPREDICT    = 0
        5.  All Branches         = 2
    */
    iter_count = 1;
    global_var2 = 1;
    do{
        global_var2+=2;
        if ( iter_count > global_var2 ){
            global_var1+=2;
        }
        BRNG();
        iter_count++;
    }while(iter_count<size);


    if ( (retval=PAPI_stop(event_set, &value)) != PAPI_OK){
        return -1;
    }
    return value;

}

long long int branch_char_b4(int size, int event_set){
    int retval;
    long long int value;

    if ( (retval=PAPI_start(event_set)) != PAPI_OK){
        return -1;
    }

    /*
        1.  Conditional EXECUTED = 2
        1.  Conditional RETIRED  = 2
        2.  Conditional TAKEN    = 1.5
        4.  Direct JUMP          = 0
        3.  Branch MISPREDICT    = 0.5
        5.  All Branches         = 2
    */

    iter_count = 1;
    do{
        iter_count++;
        BUSY_WORK();
        BRNG();
        if ( (result % 2) == 0 ){
            global_var1+=2;
        }
        BUSY_WORK();
    }while(iter_count<size);

    if ( (retval=PAPI_stop(event_set, &value)) != PAPI_OK){
        return -1;
    }

    return value;

}

long long int branch_char_b4a(int size, int event_set){
    int retval;
    long long int value;

    if ( (retval=PAPI_start(event_set)) != PAPI_OK){
        return -1;
    }

    iter_count = 1;
    do{
        iter_count++;
        BUSY_WORK();
        BRNG();
        if ( (result % 2) == 0 ){
            BUSY_WORK();
            if( (global_var1 % 2) == 0 ){
                global_var2++;
            }
            global_var1+=2;
        }
        BUSY_WORK();
    }while(iter_count<size);

    if ( (retval=PAPI_stop(event_set, &value)) != PAPI_OK){
        return -1;
    }

    return value;

}

long long int branch_char_b4b(int size, int event_set){
    int retval;
    long long int value;

    if ( (retval=PAPI_start(event_set)) != PAPI_OK){
        return -1;
    }

    iter_count = 1;
    do{
        iter_count++;
        BUSY_WORK();
        BRNG();
        if ( (result % 2) == 0 ){
            BUSY_WORK();
            if( (global_var1 % 2) != 0 ){
                global_var2++;
            }
            global_var1+=2;
        }
        BUSY_WORK();
    }while(iter_count<size);

    if ( (retval=PAPI_stop(event_set, &value)) != PAPI_OK){
        return -1;
    }

    return value;

}

long long int branch_char_b5(int size, int event_set){
    int retval;
    long long int value;

    if ( (retval=PAPI_start(event_set)) != PAPI_OK){
        return -1;
    }

    /*
        1.  Conditional EXECUTED = 2.5
        1.  Conditional RETIRED  = 2
        2.  Conditional TAKEN    = 1.5
        4.  Direct JUMP          = 0
        3.  Branch MISPREDICT    = 0.5
        5.  All Branches         = ??
    */

    iter_count = 1;
    global_var2 = 0;
    do{
        iter_count++;
        BUSY_WORK();
        BRNG();
        if ( (result % 2) == 0 ){
            global_var1+=2;
        }
    }while(iter_count<size);

    if ( (retval=PAPI_stop(event_set, &value)) != PAPI_OK){
        return -1;
    }

    return value;

}

long long int branch_char_b5a(int size, int event_set){
    int retval;
    long long int value;

    if ( (retval=PAPI_start(event_set)) != PAPI_OK){
        return -1;
    }


    iter_count = 1;
    global_var2 = 0;
    do{
        iter_count++;
        BUSY_WORK();
        BRNG();
        if ( (result % 2) == 0 ){
            if( (global_var1 % 2) == 0 ){
                global_var2++;
            }
            global_var1+=2;
            BUSY_WORK();
        }
    }while(iter_count<size);

    if ( (retval=PAPI_stop(event_set, &value)) != PAPI_OK){
        return -1;
    }

    return value;

}

long long int branch_char_b5b(int size, int event_set){
    int retval;
    long long int value;

    if ( (retval=PAPI_start(event_set)) != PAPI_OK){
        return -1;
    }


    iter_count = 1;
    global_var2 = 0;
    do{
        iter_count++;
        BUSY_WORK();
        BRNG();
        if ( (result % 2) == 0 ){
            if( (global_var1 % 2) != 0 ){
                global_var2++;
            }
            global_var1+=2;
            BUSY_WORK();
        }
    }while(iter_count<size);

    if ( (retval=PAPI_stop(event_set, &value)) != PAPI_OK){
        return -1;
    }

    return value;

}

long long int branch_char_b6(int size, int event_set){
    int retval;
    long long int value;

    if ( (retval=PAPI_start(event_set)) != PAPI_OK){
        return -1;
    }

    /*
        1.  Conditional JUMP    = 2
        2.  Conditional TAKEN   = 1
        3.  Branch MISPREDICT   = 0
        4.  Direct JUMP         = 1
        5.  All Branches        = 3
    */

    iter_count = 1;
    global_var2 = 1;
    do{
        BRNG();
        global_var2+=2;
        if ( iter_count < global_var2 ){
            global_var1+=2;
            goto zzz;
        }
        BRNG();
   zzz: iter_count++;
        BRNG();
    }while(iter_count<size);


    if ( (retval=PAPI_stop(event_set, &value)) != PAPI_OK){
        return -1;
    }
    return value;

}

long long int branch_char_b7(int size, int event_set){
    int retval;
    long long int value;

    if ( (retval=PAPI_start(event_set)) != PAPI_OK){
        return -1;
    }

    /*
        1.  Conditional JUMP    = 1
        2.  Conditional TAKEN   = 1
        3.  Branch MISPREDICT   = 0
        4.  Direct JUMP         = 0
        5.  All Branches        = 1
    */

    iter_count = 1;
    global_var2 = 1;
    do{
        global_var2=global_var2+2;
        iter_count++;
    }while(iter_count<size);

    if ( (retval=PAPI_stop(event_set, &value)) != PAPI_OK){
        return -1;
    }
    return value;

}
