#!/bin/bash

DRV_F=icache_seq.c
KRN_F=icache_seq_kernel.c
HEAD_F=icache_seq.h

TRUE_IF=1
FALSE_IF=0


################################################################################
create_common_prefix(){
  cat <<EOF
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <math.h>

#include "papi.h"
#include "icache_seq.h"

EOF
}


################################################################################
create_kernel(){

    basic_block_copies=$1;
    block_type=$2;


    for((i=0; i<$((${basic_block_copies}-1)); i++)); do
        deref[$i]=$(($i+1))
    done

    available=$((${basic_block_copies}-1));
    indx=0;

    for((i=1; i<${basic_block_copies}; i++)); do
        rnd=$((RANDOM % ${available}))
        next=${deref[${rnd}]};
        # If the next jump is too close, try one more time.
        if (( ${next} <= $((${indx}+2)) && ${next} > ${indx} )); then
            rnd=$((RANDOM % ${available}))
            next=${deref[${rnd}]};
        fi
        permutation[${indx}]=$next;
        indx=${next};
        deref[${rnd}]=${deref[$((${available}-1))]} # replace the element we used with the last one
        ((available--)); # reduce the number of available elements (to ditch the last one).
    done

    permutation[${indx}]=-1;
    last_link_in_chain=${indx};


    if (( $block_type == $TRUE_IF )); then
        echo "long long seq_kernel_TRUE_IF_${basic_block_copies}(int epilogue){"
    else
        echo "long long seq_kernel_FALSE_IF_${basic_block_copies}(int epilogue){"
    fi
    cat <<EOF
    int jj, is_zero;
    unsigned int z1 = 12345, z2 = 12345, z3 = 22745, z4 = 82395;
    unsigned result = 0;
    unsigned int b;
    long long cntr_value;
    cntr_value = 0;
    is_zero = global_zero;

EOF

    for((i=0; i<${basic_block_copies}; i++)); do
        echo ""
        if (( $block_type == $TRUE_IF )); then
            echo "    if( is_zero < 3 ){"
            echo "        RNG();"
            echo "    }"
            echo "    is_zero *= result;"
        else
            echo "    if( is_zero > 3 ){"
            echo "        RNG();"
            echo "    }"
            echo "    result = z1 ^ z2 ^ z3 ^ z4;"
            echo "    is_zero *= result;"
        fi
    done

    cat <<EOF

    if( DO_COPY == epilogue ){
        // Access enough elements to flush the shared caches.
        for(jj=0; jj<BUF_ELEM_CNT; jj++){
            is_zero += (int)buff[jj];
        }
    }

    cntr_value += is_zero;

EOF
    echo ""
    echo "    return cntr_value;"
    echo "}"

}


################################################################################
create_caller(){

    basic_block_copies=$1;
    dl_reps=$2;


    cat <<EOF
int seq_jumps_${basic_block_copies}x${dl_reps}(int iter_count, int eventset, int epilogue, int branch_type, int run_type, FILE* ofp_papi){
    long long int cntr_value = 0;
    int i, j, is_zero = 13;
    int ret;
    char fname[512];
EOF
    echo "    void *obj[${dl_reps}];"
    echo "    void *sym_true_if[${dl_reps}];"
    echo "    void *sym_false_if[${dl_reps}];"
    echo ""
    echo "    for(j=0; j<${dl_reps}; j++){"
    echo ""
    echo "        snprintf(fname, 512, \"icache_seq_kernel_%d.so\", j);"
    echo "        obj[j] = dlopen(fname, RTLD_NOW|RTLD_LOCAL);"
    echo "        if( NULL == obj[j] ){"
    echo "            fprintf(stderr,\"dlopen() %d failed on %s because: %s\nIs the directory containing '%s' in your LD_LIBRARY_PATH?\n\", j, fname, dlerror(), fname);"
    echo "            return -1;"
    echo "        }"
    echo "        sym_true_if[j] = dlsym(obj[j], \"seq_kernel_TRUE_IF_${basic_block_copies}\");"
    echo "        if( NULL == sym_true_if[j] ){"
    echo "            fprintf(stderr,\"dlsym() %d failed because: %s\n\", j, dlerror());"
    echo "            return -1;"
    echo "        }"
    echo "        sym_false_if[j] = dlsym(obj[j], \"seq_kernel_FALSE_IF_${basic_block_copies}\");"
    echo "        if( NULL == sym_true_if[j] ){"
    echo "            fprintf(stderr,\"dlsym() %d failed because: %s\n\", j, dlerror());"
    echo "            return -1;"
    echo "        }"
    echo "    }"
    echo ""
    echo "    if((ret=PAPI_start(eventset)) != PAPI_OK){"
    echo "        return -1;"
    echo "    }"
    echo ""
    echo "    if( TRUE_IF == branch_type ){"
    echo "        for(i=0; i<iter_count; i++){"
    echo "            // cntr_value += seq_kernel_TRUE_IF_${basic_block_copies}(eventset, epilogue);"
    echo "            for(j=0; j<${dl_reps}; j++){"
    echo "                cntr_value += ((long long (*)(int))sym_true_if[j])(epilogue);"
    echo "            }"
    echo "        }"
    echo "    }else{"
    echo "        for(i=0; i<iter_count; i++){"
    echo "            // cntr_value += seq_kernel_FALSE_IF_${basic_block_copies}(eventset, epilogue);"
    echo "            for(j=0; j<${dl_reps}; j++){"
    echo "                cntr_value += ((long long (*)(int))sym_false_if[j])(epilogue);"
    echo "            }"
    echo "        }"
    echo "    }"

    cat <<EOF

    if((ret=PAPI_stop(eventset, &cntr_value)) != PAPI_OK){
        return -1;
    }

EOF

    echo "    for(j=0; j<${dl_reps}; j++){"
    echo "        snprintf(fname, 512, \"seq_kernel_%d.so\", j);"
    echo "        if( dlclose(obj[j]) ){"
    echo "            fprintf(stderr,\"dlclose() %d failed on %s because: %s\n\", j, fname, dlerror());"
    echo "            return -1;"
    echo "        }"
    echo "    }"

    echo "    if( COLD_RUN != run_type ){"
    echo "        fprintf(ofp_papi, \"%lf\\n\", ((double)cntr_value)/(${basic_block_copies}*${dl_reps}*(double)iter_count) );"
    echo "    }"

    cat <<EOF

    is_zero = global_zero * (int)cntr_value;

    return is_zero;
}
EOF
}


################################################################################
create_functions(){
    basic_block_copies=$1;

    dl_reps=1;

    if (( $basic_block_copies >= 10000 )); then
        dl_reps=$(( ${basic_block_copies}/5000 ))
        tmp=$(( ${basic_block_copies}/${dl_reps} ))
        basic_block_copies=$tmp
    else
        create_kernel $basic_block_copies $j $TRUE_IF >> ${KRN_F}
        create_kernel $basic_block_copies $j $FALSE_IF >> ${KRN_F}
        echo "" >> ${KRN_F}
        echo "long long seq_kernel_TRUE_IF_${basic_block_copies}(int epilogue);" >> ${HEAD_F}
        echo "long long seq_kernel_FALSE_IF_${basic_block_copies}(int epilogue);" >> ${HEAD_F}
    fi
    echo "int seq_jumps_${basic_block_copies}x${dl_reps}(int iter_count, int eventset, int epilogue, int branch_type, int run_type, FILE* ofp_papi);" >> ${HEAD_F}

    create_caller ${basic_block_copies} $dl_reps >> ${DRV_F}
    echo "" >> ${DRV_F}

}


################################################################################
create_main(){

    cat <<EOF
void seq_driver(FILE* ofp_papi, char* papi_event_name, int init, int show_progress){
    int ret, exp_cnt=0, side_effect=0;
    size_t i;
    int eventset = PAPI_NULL;

    // Fill up the buffer with some nonsense numbers that will round to zero.
    for(i=0; i<BUF_ELEM_CNT; i++){
        buff[i] = floor( ((float)i+0.71)/((float)i+8.0*(float)init) );
        if( (int)buff[i] != 0 )
            fprintf(stderr,"WARNING: this element should have been zero: buff[%lu] = %d (%f). The branch benchmarks might not work properly.\n",i, (int)buff[i], buff[i]);
    }

    // Set the variable to zero in a way that the compiler cannot figure it out.
    global_zero = (int)floor( (buff[3]+1) / (buff[9]+getpid()) );

    //set up PAPI
    if((ret=PAPI_create_eventset(&eventset)) != PAPI_OK){
        for(i=0; i<strlen("Total:100%  Current test:"); i++) putchar('\b');
        fflush(stdout);
        return;
    }
    if((ret=PAPI_add_named_event(eventset, papi_event_name)) != PAPI_OK){
        for(i=0; i<strlen("Total:100%  Current test:"); i++) putchar('\b');
        fflush(stdout);
        return;
    }

    side_effect = init;
EOF

    for copy_type in "NO_COPY"; do
        for ((prm=1; prm<=$#; prm++)); do
            basic_block_copies=${!prm}
            dl_reps=1;
            if (( $basic_block_copies >= 10000 )); then
                dl_reps=$(( ${basic_block_copies}/5000 ))
                tmp=$(( ${basic_block_copies}/${dl_reps} ))
                basic_block_copies=$tmp
            fi
            echo "    if( show_progress ){"
            echo "        printf(\"%3d%%\b\b\b\b\",(100*exp_cnt)/(4*$#));"
            echo "        exp_cnt++;"
            echo "        fflush(stdout);"
            echo "    }"
            echo "    side_effect += seq_jumps_${basic_block_copies}x${dl_reps}(1, eventset, NO_COPY, TRUE_IF, COLD_RUN, NULL);"
            echo "    if(side_effect < init){"
            echo "        return;"
            echo "    }"
            echo "    side_effect += seq_jumps_${basic_block_copies}x${dl_reps}(150, eventset, ${copy_type}, TRUE_IF, NORMAL_RUN, ofp_papi);"
            echo "    if(side_effect < init){"
            echo "        return;"
            echo "    }"
            echo ""
        done
    done

    for copy_type in "NO_COPY"; do
        for ((prm=1; prm<=$#; prm++)); do
            basic_block_copies=${!prm}
            dl_reps=1;
            if (( $basic_block_copies >= 10000 )); then
                dl_reps=$(( ${basic_block_copies}/5000 ))
                tmp=$(( ${basic_block_copies}/${dl_reps} ))
                basic_block_copies=$tmp
            fi
            echo "    if( show_progress ){"
            echo "        printf(\"%3d%%\b\b\b\b\",(100*exp_cnt)/(4*$#));"
            echo "        exp_cnt++;"
            echo "        fflush(stdout);"
            echo "    }"
            echo "    side_effect += seq_jumps_${basic_block_copies}x${dl_reps}(1, eventset, NO_COPY, FALSE_IF, COLD_RUN, NULL);"
            echo "    if(side_effect < init){"
            echo "        return;"
            echo "    }"
            echo "    side_effect += seq_jumps_${basic_block_copies}x${dl_reps}(150, eventset, ${copy_type}, FALSE_IF, NORMAL_RUN, ofp_papi);"
            echo "    if(side_effect < init){"
            echo "        return;"
            echo "    }"
            echo ""
        done
    done
    cat <<EOF

    if( show_progress ){
        size_t i;
        printf("100%%");
        for(i=0; i<strlen("Total:100%  Current test:100%"); i++) putchar('\b');
        fflush(stdout);
    }

    if( 174562 == side_effect ){
        printf("Random side-effect\n");
    }

    ret = PAPI_cleanup_eventset( eventset );
    if (ret != PAPI_OK ){
        return;
    }
    ret = PAPI_destroy_eventset( &eventset );
    if (ret != PAPI_OK ){
        return;
    }


    return;
}
EOF
}

echo "#include \"icache.h\"" > ${HEAD_F}
echo "#include <dlfcn.h>" >> ${HEAD_F}

echo "" >> ${HEAD_F}
echo "float buff[BUF_ELEM_CNT];" >> ${HEAD_F}
echo "volatile int global_zero;" >> ${HEAD_F}
echo "" >> ${HEAD_F}

create_common_prefix > ${DRV_F}
create_common_prefix > ${KRN_F}
for sz in 10 20 30 50 100 150 200 300 400 600 800 1200 1600 2400 3200 5000 10000 15000 20000 25000 35000 40000 50000 60000;  do
    create_functions ${sz}
done

create_main 10 20 30 50 100 150 200 300 400 600 800 1200 1600 2400 3200 5000 10000 15000 20000 25000 35000 40000 50000 60000 >> ${DRV_F}

