#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <unistd.h>
#include <stdint.h>
#include <string.h>
#include <papi.h>
#include "instr.h"

int sum_i32=0;
float sum_f32=0.0;
double sum_f64=0.0;

void test_int_add(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    int i32_00, i32_01, i32_02, i32_03, i32_04, i32_05, i32_06, i32_07, i32_08, i32_09;

    /* Initialize the variables with values that the compiler cannot guess. */
    i32_00 =  p/2;
    i32_01 = -p/3;
    i32_02 =  p/4;
    i32_03 = -p/5;
    i32_04 =  p/6;
    i32_05 = -p/7;
    i32_06 =  p/8;
    i32_07 = -p/9;
    i32_08 =  p/10;
    i32_09 = -p/11;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){

            i32_01 += i32_00;
            i32_02 += i32_01;
            i32_03 += i32_02;
            i32_04 += i32_03;
            i32_05 += i32_04;
            i32_06 += i32_05;
            i32_07 += i32_06;
            i32_08 += i32_07;
            i32_09 += i32_08;
            i32_00 += i32_09;

            i32_01 += i32_00;
            i32_02 += i32_01;
            i32_03 += i32_02;
            i32_04 += i32_03;
            i32_05 += i32_04;
            i32_06 += i32_05;
            i32_07 += i32_06;
            i32_08 += i32_07;
            i32_09 += i32_08;
            i32_00 += i32_09;

            i32_01 += i32_00;
            i32_02 += i32_01;
            i32_03 += i32_02;
            i32_04 += i32_03;
            i32_05 += i32_04;
            i32_06 += i32_05;
            i32_07 += i32_06;
            i32_08 += i32_07;
            i32_09 += i32_08;
            i32_00 += i32_09;

            i32_01 += i32_00;
            i32_02 += i32_01;
            i32_03 += i32_02;
            i32_04 += i32_03;
            i32_05 += i32_04;
            i32_06 += i32_05;
            i32_07 += i32_06;
            i32_08 += i32_07;
            i32_09 += i32_08;
            i32_00 += i32_09;

            i32_01 += i32_00;
            i32_02 += i32_01;
            i32_03 += i32_02;
            i32_04 += i32_03;
            i32_05 += i32_04;
            i32_06 += i32_05;
            i32_07 += i32_06;
            i32_08 += i32_07;
            i32_09 += i32_08;
            i32_00 += i32_09;

        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 50LL*N*M, (double)ev_values[0]/(50.0*N*M));

    sum_i32 += i32_00 + i32_01 + i32_02 + i32_03 + i32_04 + i32_05 + i32_06 + i32_07 + i32_08 + i32_09;

clean_up:

    return;
}


void test_int_add_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    int i32_00, i32_01, i32_02, i32_03, i32_04, i32_05, i32_06, i32_07;
    int i32_08, i32_09, i32_10, i32_11;
    int i32_100, i32_101, i32_102;

    /* Initialize the variables with values that the compiler cannot guess. */
    i32_00 =  2*p;
    i32_01 = -p/3;
    i32_02 =  p/4;
    i32_03 = -p/5;
    i32_04 =  p/6;
    i32_05 = -p/7;
    i32_06 =  p/8;
    i32_07 = -p/9;
    i32_08 =  1+p/2;
    i32_09 =  1-p/2;
    i32_10 =  1+p/3;
    i32_11 =  1-p/3;

    i32_100 =  17;
    i32_101 = -18;
    i32_102 =  12;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p == 12345678 ){
        p /= 2;
        i32_100 *= 13;
        i32_101 *= 12;
        i32_102 *= 11;
    }else{
        // Almost certainly this is what will execute and all variables will
        // end up with the value zero, but the compiler doesn't know that.
        i32_100 /= i32_00+16;
        i32_101 /= i32_00+17;
        i32_102 /= i32_00+11;
    }

#define I32_ADDS(_X) {i32_00 += _X; i32_01 += _X; i32_02 += _X; i32_03 += _X; i32_04 += _X; i32_05 += _X; i32_06 += _X; i32_07 += _X; i32_08 += _X; i32_09 += _X; i32_10 += _X; i32_11 += _X;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            I32_ADDS(i32_100);
            I32_ADDS(i32_101);
            I32_ADDS(i32_102);
            if( i32_100 > 100 ){
                I32_ADDS(i32_07);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 12LL*3LL*N*M, (double)ev_values[0]/(12.0*3.0*N*M));

    sum_i32 += i32_00 + i32_01 + i32_02 + i32_03 + i32_04 + i32_05 + i32_06 + i32_07;
    sum_i32 += i32_08 + i32_09 + i32_10 + i32_11;

clean_up:

    return;
}


void test_int_mul_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    int i32_00, i32_01, i32_02, i32_03, i32_04, i32_05, i32_06, i32_07;
    int i32_08, i32_09, i32_10, i32_11;
    int i32_100, i32_101, i32_102;

    /* Initialize the variables with values that the compiler cannot guess. */
    i32_00 =  2*p;
    i32_01 = -p/3;
    i32_02 =  p/4;
    i32_03 = -p/5;
    i32_04 =  p/6;
    i32_05 = -p/7;
    i32_06 =  p/8;
    i32_07 =  1/p;
    i32_08 =  1+p/2;
    i32_09 =  1-p/2;
    i32_10 =  1+p/3;
    i32_11 =  1-p/3;

    i32_100 =  17;
    i32_101 = -18;
    i32_102 =  12;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p == 12345678 ){
        p /= 2;
        i32_100 *= 13;
        i32_101 *= 12;
        i32_102 *= 11;
    }else{
        // Almost certainly this is what will execute and all variables will
        // end up with the value one, but the compiler doesn't know that.
        i32_100 = 1 + i32_100 / (i32_00+16);
        i32_101 = 1 + i32_101 / (i32_00+17);
        i32_102 = 1 + i32_102 / (i32_00+11);
    }

#define I32_MULS(_X) {i32_00 *= _X; i32_01 *= _X; i32_02 *= _X; i32_03 *= _X; i32_04 *= _X; i32_05 *= _X; i32_06 *= _X; i32_07 *= _X; i32_08 *= _X; i32_09 *= _X; i32_10 *= _X; i32_11 *= _X;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            I32_MULS(i32_100);
            I32_MULS(i32_101);
            I32_MULS(i32_102);
            if( i32_100 > 100 ){
                I32_MULS(i32_07);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 50LL*N*M, (double)ev_values[0]/(50.0*N*M));

    sum_i32 += i32_00 + i32_01 + i32_02 + i32_03 + i32_04 + i32_05 + i32_06 + i32_07;
    sum_i32 += i32_08 + i32_09 + i32_10 + i32_11;

clean_up:

    return;
}


void test_int_div_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    int i32_00, i32_01, i32_02, i32_03, i32_04, i32_05, i32_06, i32_07;
    int i32_08, i32_09, i32_10, i32_11;
    int i32_100, i32_101, i32_102;

    /* Initialize the variables with values that the compiler cannot guess. */
    i32_00 =  2*p;
    i32_01 = -p/3;
    i32_02 =  p/4;
    i32_03 = -p/5;
    i32_04 =  p/6;
    i32_05 = -p/7;
    i32_06 =  p/8;
    i32_07 =  1+1/p;
    i32_08 =  1+p/2;
    i32_09 =  1-p/2;
    i32_10 =  1+p/3;
    i32_11 =  1-p/3;

    i32_100 =  17;
    i32_101 = -18;
    i32_102 =  12;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p == 12345678 ){
        p /= 2;
        i32_100 *= 13;
        i32_101 *= 12;
        i32_102 *= 11;
    }else{
        // Almost certainly this is what will execute and all variables will
        // end up with the value one, but the compiler doesn't know that.
        i32_100 = 1 + i32_100 / (i32_00+16);
        i32_101 = 1 + i32_101 / (i32_00+17);
        i32_102 = 1 + i32_102 / (i32_00+11);
    }

#define I32_DIVS(_X) {i32_00 /= _X; i32_01 /= _X; i32_02 /= _X; i32_03 /= _X; i32_04 /= _X; i32_05 /= _X; i32_06 /= _X; i32_07 /= _X; i32_08 /= _X; i32_09 /= _X; i32_10 /= _X; i32_11 /= _X;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            I32_DIVS(i32_100);
            I32_DIVS(i32_101);
            I32_DIVS(i32_102);
            if( i32_100 > 100 ){
                I32_DIVS(i32_07);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 50LL*N*M, (double)ev_values[0]/(50.0*N*M));

    sum_i32 += i32_00 + i32_01 + i32_02 + i32_03 + i32_04 + i32_05 + i32_06 + i32_07;
    sum_i32 += i32_08 + i32_09 + i32_10 + i32_11;

clean_up:

    return;
}


////////////////////////////////////////////////////////////////////////////////
// f32 ADD

void test_f32_add(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    float f32_00, f32_01, f32_02, f32_03;

    /* Initialize the variables with values that the compiler cannot guess. */
    f32_00 =  (float)p/1.02;
    f32_01 = -(float)p/1.03;
    f32_02 =  (float)p/1.04;
    f32_03 = -(float)p/1.05;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

#define F32ADD_BLOCK() {f32_01 += f32_00; f32_02 += f32_01; f32_03 += f32_02; f32_00 += f32_03;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F32ADD_BLOCK();
            F32ADD_BLOCK();
            F32ADD_BLOCK();
            F32ADD_BLOCK();
            F32ADD_BLOCK();
            F32ADD_BLOCK();
            F32ADD_BLOCK();
            F32ADD_BLOCK();
            F32ADD_BLOCK();
            F32ADD_BLOCK();
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    long long int fp_op_count = 40LL*N*M; // There are only 50 FP operations.
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], fp_op_count, (double)ev_values[0]/(double)fp_op_count);

    sum_f32 += f32_00 + f32_01 + f32_02 + f32_03;

clean_up:

    return;
}


void test_f32_add_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    float f32_00, f32_01, f32_02, f32_03, f32_04, f32_05, f32_06, f32_07;
    float f32_08, f32_09, f32_10, f32_11;
    float f32_100, f32_101, f32_102;

    /* Initialize the variables with values that the compiler cannot guess. */
    f32_00 =  p/1.2;
    f32_01 = -p/1.3;
    f32_02 =  p/1.4;
    f32_03 = -p/1.5;
    f32_04 =  p/1.6;
    f32_05 = -p/1.7;
    f32_06 =  p/1.8;
    f32_07 = -p/1.9;
    f32_08 =  p/2.0;
    f32_09 = -p/2.1;
    f32_10 =  p/2.2;
    f32_11 = -p/2.3;

    f32_100 =  0.00100;
    f32_101 = -0.00101;
    f32_102 =  0.00102;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p == 123456 ){
        p /= 2;
        f32_100 *= 1.045;
        f32_101 *= 1.054;
        f32_102 *= 1.067;
    }

#define F32_ADDS(_X) {f32_00 += _X; f32_01 += _X; f32_02 += _X; f32_03 += _X; f32_04 += _X; f32_05 += _X; f32_06 += _X; f32_07 += _X; f32_08 += _X; f32_09 += _X; f32_10 += _X; f32_11 += _X;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F32_ADDS(f32_100);
            F32_ADDS(f32_101);
            F32_ADDS(f32_102);
            if( p < 2 ){
                F32_ADDS(f32_00);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 12LL*3LL*N*M, (double)ev_values[0]/(12.0*3.0*N*M));

    sum_f32 += f32_00 + f32_01 + f32_02 + f32_03 + f32_04 + f32_05 + f32_06 + f32_07;
    sum_f32 += f32_08 + f32_09 + f32_10 + f32_11;

clean_up:

    return;
}


////////////////////////////////////////////////////////////////////////////////
// f32 SUB

void test_f32_sub(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    float f32_00, f32_01, f32_02, f32_03;

    /* Initialize the variables with values that the compiler cannot guess. */
    f32_00 =  (float)p/1.02;
    f32_01 = -(float)p/1.03;
    f32_02 =  (float)p/1.04;
    f32_03 = -(float)p/1.05;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

#define F32SUB_BLOCK() {f32_01 -= f32_00; f32_02 -= f32_01; f32_03 -= f32_02; f32_00 -= f32_03;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F32SUB_BLOCK();
            F32SUB_BLOCK();
            F32SUB_BLOCK();
            F32SUB_BLOCK();
            F32SUB_BLOCK();
            F32SUB_BLOCK();
            F32SUB_BLOCK();
            F32SUB_BLOCK();
            F32SUB_BLOCK();
            F32SUB_BLOCK();
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    long long int fp_op_count = 40LL*N*M; // There are only 50 FP operations.
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], fp_op_count, (double)ev_values[0]/(double)fp_op_count);

    sum_f32 += f32_00 + f32_01 + f32_02 + f32_03;

clean_up:

    return;
}


void test_f32_sub_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    float f32_00, f32_01, f32_02, f32_03, f32_04, f32_05, f32_06, f32_07;
    float f32_08, f32_09, f32_10, f32_11;
    float f32_100, f32_101, f32_102;

    /* Initialize the variables with values that the compiler cannot guess. */
    f32_00 =  p/1.2;
    f32_01 = -p/1.3;
    f32_02 =  p/1.4;
    f32_03 = -p/1.5;
    f32_04 =  p/1.6;
    f32_05 = -p/1.7;
    f32_06 =  p/1.8;
    f32_07 = -p/1.9;
    f32_08 =  p/2.0;
    f32_09 = -p/2.1;
    f32_10 =  p/2.2;
    f32_11 = -p/2.3;

    f32_100 =  0.00100;
    f32_101 = -0.00101;
    f32_102 =  0.00102;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p == 123456 ){
        p /= 2;
        f32_100 *= 1.045;
        f32_101 *= 1.054;
        f32_102 *= 1.067;
    }

#define F32_SUBS(_X) {f32_00 -= _X; f32_01 -= _X; f32_02 -= _X; f32_03 -= _X; f32_04 -= _X; f32_05 -= _X; f32_06 -= _X; f32_07 -= _X; f32_08 -= _X; f32_09 -= _X; f32_10 -= _X; f32_11 -= _X;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F32_SUBS(f32_100);
            F32_SUBS(f32_101);
            F32_SUBS(f32_102);
            if( p < 2 ){
                F32_SUBS(f32_00);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 12LL*3LL*N*M, (double)ev_values[0]/(12.0*3.0*N*M));

    sum_f32 += f32_00 + f32_01 + f32_02 + f32_03 + f32_04 + f32_05 + f32_06 + f32_07;
    sum_f32 += f32_08 + f32_09 + f32_10 + f32_11;

clean_up:

    return;
}

////////////////////////////////////////////////////////////////////////////////
// f32 MULTIPLICATION

void test_f32_mul(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    float f32_00, f32_01, f32_02, f32_03;

    /* Initialize the variables with values that the compiler cannot guess. */
    f32_00 =  (float)p/1.02;
    f32_01 =  1.03/(float)p;
    f32_02 =  (float)p/1.04;
    f32_03 =  1.05/(float)p;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

#define F32MUL_BLOCK() {f32_01 *= f32_00; f32_02 *= f32_01; f32_03 *= f32_02; f32_00 *= f32_03;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F32MUL_BLOCK();
            F32MUL_BLOCK();
            F32MUL_BLOCK();
            F32MUL_BLOCK();
            F32MUL_BLOCK();
            F32MUL_BLOCK();
            F32MUL_BLOCK();
            F32MUL_BLOCK();
            F32MUL_BLOCK();
            F32MUL_BLOCK();
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    long long int fp_op_count = 40LL*N*M; // There are only 50 FP operations.
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], fp_op_count, (double)ev_values[0]/(double)fp_op_count);

    sum_f32 += f32_00 + f32_01 + f32_02 + f32_03;

clean_up:

    return;
}


void test_f32_mul_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    float f32_00, f32_01, f32_02, f32_03, f32_04, f32_05, f32_06, f32_07;
    float f32_08, f32_09, f32_10, f32_11;
    float f32_100, f32_101, f32_102;

    /* Initialize the variables with values that the compiler cannot guess. */
    f32_00 =  p/431.2;
    f32_01 = -p/431.3;
    f32_02 =  p/431.4;
    f32_03 = -p/431.5;
    f32_04 =  p/431.6;
    f32_05 = -p/431.7;
    f32_06 =  p/431.8;
    f32_07 = -p/431.9;
    f32_08 =  p/432.0;
    f32_09 = -p/432.1;
    f32_10 =  p/432.2;
    f32_11 = -p/432.3;

    f32_100 =  1.00001;
    f32_101 = -1.00002;
    f32_102 =  1.00003;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p == 123456 ){
        p /= 2;
        f32_100 *= 1.0045;
        f32_101 *= 1.0054;
        f32_102 *= 1.0067;
    }

#define F32_MULS(_X) {f32_00 *= _X; f32_01 *= _X; f32_02 *= _X; f32_03 *= _X; f32_04 *= _X; f32_05 *= _X; f32_06 *= _X; f32_07 *= _X; f32_08 *= _X; f32_09 *= _X; f32_10 *= _X; f32_11 *= _X;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F32_MULS(f32_100);
            F32_MULS(f32_101);
            F32_MULS(f32_102);
            if( p < 2 ){
                F32_MULS(f32_00);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 12LL*3LL*N*M, (double)ev_values[0]/(12.0*3.0*N*M));

    sum_f32 += f32_00 + f32_01 + f32_02 + f32_03 + f32_04 + f32_05 + f32_06 + f32_07;
    sum_f32 += f32_08 + f32_09 + f32_10 + f32_11;

clean_up:

    return;
}

////////////////////////////////////////////////////////////////////////////////
// f32 DIVISION

void test_f32_div(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    float f32_00, f32_01, f32_02, f32_03;

    /* Initialize the variables with values that the compiler cannot guess. */
    f32_00 =  1.0 + 1.0/(1000.1*(float)p);
    f32_01 =  1.0 + 1.0/(1000.2*(float)p);
    f32_02 =  1.0 + 1.0/(1000.3*(float)p);
    f32_03 =  1.0 + 1.0/(1000.4*(float)p);

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

#define F32DIV_BLOCK() {f32_01 /= f32_00; f32_02 /= f32_01; f32_03 /= f32_02; f32_00 /= f32_03;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F32DIV_BLOCK();
            F32DIV_BLOCK();
            F32DIV_BLOCK();
            F32DIV_BLOCK();
            F32DIV_BLOCK();
            F32DIV_BLOCK();
            F32DIV_BLOCK();
            F32DIV_BLOCK();
            F32DIV_BLOCK();
            F32DIV_BLOCK();
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    long long int fp_op_count = 40LL*N*M; // There are only 50 FP operations.
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], fp_op_count, (double)ev_values[0]/(double)fp_op_count);

    sum_f32 += f32_00 + f32_01 + f32_02 + f32_03;

clean_up:

    return;
}


void test_f32_div_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    float f32_00, f32_01, f32_02, f32_03, f32_04, f32_05, f32_06, f32_07;
    float f32_08, f32_09, f32_10, f32_11;
    float f32_100, f32_101, f32_102;

    /* Initialize the variables with values that the compiler cannot guess. */
    f32_00 =  p/431.2;
    f32_01 = -p/431.3;
    f32_02 =  p/431.4;
    f32_03 = -p/431.5;
    f32_04 =  p/431.6;
    f32_05 = -p/431.7;
    f32_06 =  p/431.8;
    f32_07 = -p/431.9;
    f32_08 =  p/432.0;
    f32_09 = -p/432.1;
    f32_10 =  p/432.2;
    f32_11 = -p/432.3;

    f32_100 =  1.00001;
    f32_101 = -1.00002;
    f32_102 =  1.00003;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p == 123456 ){
        p /= 2;
        f32_100 *= 1.0045;
        f32_101 *= 1.0054;
        f32_102 *= 1.0067;
    }

#define F32_DIVS(_X) {f32_00 /= _X; f32_01 /= _X; f32_02 /= _X; f32_03 /= _X; f32_04 /= _X; f32_05 /= _X; f32_06 /= _X; f32_07 /= _X; f32_08 /= _X; f32_09 /= _X; f32_10 /= _X; f32_11 /= _X;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F32_DIVS(f32_100);
            F32_DIVS(f32_101);
            F32_DIVS(f32_102);
            if( p < 2 ){
                F32_DIVS(f32_00);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 12LL*3LL*N*M, (double)ev_values[0]/(12.0*3.0*N*M));

    sum_f32 += f32_00 + f32_01 + f32_02 + f32_03 + f32_04 + f32_05 + f32_06 + f32_07;
    sum_f32 += f32_08 + f32_09 + f32_10 + f32_11;

clean_up:

    return;
}


void test_f32_fma_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    float f32_00, f32_01, f32_02, f32_03, f32_04, f32_05, f32_06, f32_07;
    float f32_08, f32_09, f32_10, f32_11;
    float f32_100, f32_101, f32_102;
    float f32_B;

    /* Initialize the variables with values that the compiler cannot guess. */
    f32_00 =  p/431.2;
    f32_01 = -p/431.3;
    f32_02 =  p/431.4;
    f32_03 = -p/431.5;
    f32_04 =  p/431.6;
    f32_05 = -p/431.7;
    f32_06 =  p/431.8;
    f32_07 = -p/431.9;
    f32_08 =  p/432.0;
    f32_09 = -p/432.1;
    f32_10 =  p/432.2;
    f32_11 = -p/432.3;

    f32_100 =  1.00001;
    f32_101 = -1.00002;
    f32_102 =  1.00003;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p != 12345678 ){
        f32_100 /= 1.000045;
        f32_101 /= 1.000054;
        f32_102 /= 1.000067;
    }
    f32_B = f32_100/34567.8;

#define F32_FMAS(_A,_B) {f32_00 = _A*f32_00+_B; f32_01 = _A*f32_01+_B; f32_02 = _A*f32_02+_B; f32_03 = _A*f32_03+_B; f32_04 = _A*f32_04+_B; f32_05 = _A*f32_05+_B; f32_06 = _A*f32_06+_B; f32_07 = _A*f32_07+_B; f32_08 = _A*f32_08+_B; f32_09 = _A*f32_09+_B; f32_10 = _A*f32_10+_B; f32_11 = _A*f32_11+_B;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F32_FMAS(f32_100, f32_B);
            F32_FMAS(f32_101, f32_B);
            F32_FMAS(f32_102, f32_B);
            if( p < 2 ){
                F32_FMAS(f32_00, f32_B);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 12LL*3LL*N*M, (double)ev_values[0]/(12.0*3.0*N*M));

    sum_f32 += f32_00 + f32_01 + f32_02 + f32_03 + f32_04 + f32_05 + f32_06 + f32_07;
    sum_f32 += f32_08 + f32_09 + f32_10 + f32_11;

clean_up:

    return;
}


////////////////////////////////////////////////////////////////////////////////
// f64 ADD

void test_f64_add(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  (double)p/1.02;
    f64_01 = -(double)p/1.03;
    f64_02 =  (double)p/1.04;
    f64_03 = -(double)p/1.05;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

#define F64ADD_BLOCK() {f64_01 += f64_00; f64_02 += f64_01; f64_03 += f64_02; f64_00 += f64_03;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F64ADD_BLOCK();
            F64ADD_BLOCK();
            F64ADD_BLOCK();
            F64ADD_BLOCK();
            F64ADD_BLOCK();
            F64ADD_BLOCK();
            F64ADD_BLOCK();
            F64ADD_BLOCK();
            F64ADD_BLOCK();
            F64ADD_BLOCK();
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    long long int fp_op_count = 40LL*N*M; // There are only 50 FP operations.
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], fp_op_count, (double)ev_values[0]/(double)fp_op_count);

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03;

clean_up:

    return;
}


void test_f64_add_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03, f64_04, f64_05, f64_06, f64_07;
    double f64_08, f64_09, f64_10, f64_11;
    double f64_100, f64_101, f64_102;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  p/1.2;
    f64_01 = -p/1.3;
    f64_02 =  p/1.4;
    f64_03 = -p/1.5;
    f64_04 =  p/1.6;
    f64_05 = -p/1.7;
    f64_06 =  p/1.8;
    f64_07 = -p/1.9;
    f64_08 =  p/2.0;
    f64_09 = -p/2.1;
    f64_10 =  p/2.2;
    f64_11 = -p/2.3;

    f64_100 =  0.00100;
    f64_101 = -0.00101;
    f64_102 =  0.00102;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p == 123456 ){
        p /= 2;
        f64_100 *= 1.045;
        f64_101 *= 1.054;
        f64_102 *= 1.067;
    }

#define F64_ADDS(_X) {f64_00 += _X; f64_01 += _X; f64_02 += _X; f64_03 += _X; f64_04 += _X; f64_05 += _X; f64_06 += _X; f64_07 += _X; f64_08 += _X; f64_09 += _X; f64_10 += _X; f64_11 += _X;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F64_ADDS(f64_100);
            F64_ADDS(f64_101);
            F64_ADDS(f64_102);
            if( p < 2 ){
                F64_ADDS(f64_00);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 12LL*3LL*N*M, (double)ev_values[0]/(12.0*3.0*N*M));

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03 + f64_04 + f64_05 + f64_06 + f64_07;
    sum_f64 += f64_08 + f64_09 + f64_10 + f64_11;

clean_up:

    return;
}


////////////////////////////////////////////////////////////////////////////////
// f64 SUB

void test_f64_sub(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  (double)p/1.02;
    f64_01 = -(double)p/1.03;
    f64_02 =  (double)p/1.04;
    f64_03 = -(double)p/1.05;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

#define F64SUB_BLOCK() {f64_01 -= f64_00; f64_02 -= f64_01; f64_03 -= f64_02; f64_00 -= f64_03;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F64SUB_BLOCK();
            F64SUB_BLOCK();
            F64SUB_BLOCK();
            F64SUB_BLOCK();
            F64SUB_BLOCK();
            F64SUB_BLOCK();
            F64SUB_BLOCK();
            F64SUB_BLOCK();
            F64SUB_BLOCK();
            F64SUB_BLOCK();
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    long long int fp_op_count = 40LL*N*M; // There are only 50 FP operations.
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], fp_op_count, (double)ev_values[0]/(double)fp_op_count);

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03;

clean_up:

    return;
}


void test_f64_sub_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03, f64_04, f64_05, f64_06, f64_07;
    double f64_08, f64_09, f64_10, f64_11;
    double f64_100, f64_101, f64_102;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  p/1.2;
    f64_01 = -p/1.3;
    f64_02 =  p/1.4;
    f64_03 = -p/1.5;
    f64_04 =  p/1.6;
    f64_05 = -p/1.7;
    f64_06 =  p/1.8;
    f64_07 = -p/1.9;
    f64_08 =  p/2.0;
    f64_09 = -p/2.1;
    f64_10 =  p/2.2;
    f64_11 = -p/2.3;

    f64_100 =  0.00100;
    f64_101 = -0.00101;
    f64_102 =  0.00102;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p == 123456 ){
        p /= 2;
        f64_100 *= 1.045;
        f64_101 *= 1.054;
        f64_102 *= 1.067;
    }

#define F64_SUBS(_X) {f64_00 -= _X; f64_01 -= _X; f64_02 -= _X; f64_03 -= _X; f64_04 -= _X; f64_05 -= _X; f64_06 -= _X; f64_07 -= _X; f64_08 -= _X; f64_09 -= _X; f64_10 -= _X; f64_11 -= _X;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F64_SUBS(f64_100);
            F64_SUBS(f64_101);
            F64_SUBS(f64_102);
            if( p < 2 ){
                F64_SUBS(f64_00);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 12LL*3LL*N*M, (double)ev_values[0]/(12.0*3.0*N*M));

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03 + f64_04 + f64_05 + f64_06 + f64_07;
    sum_f64 += f64_08 + f64_09 + f64_10 + f64_11;

clean_up:

    return;
}

////////////////////////////////////////////////////////////////////////////////
// f64 MULTIPLICATION

void test_f64_mul(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  (double)p/1.02;
    f64_01 =  1.03/(double)p;
    f64_02 =  (double)p/1.04;
    f64_03 =  1.05/(double)p;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

#define F64MUL_BLOCK() {f64_01 *= f64_00; f64_02 *= f64_01; f64_03 *= f64_02; f64_00 *= f64_03;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F64MUL_BLOCK();
            F64MUL_BLOCK();
            F64MUL_BLOCK();
            F64MUL_BLOCK();
            F64MUL_BLOCK();
            F64MUL_BLOCK();
            F64MUL_BLOCK();
            F64MUL_BLOCK();
            F64MUL_BLOCK();
            F64MUL_BLOCK();
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    long long int fp_op_count = 40LL*N*M; // There are only 50 FP operations.
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], fp_op_count, (double)ev_values[0]/(double)fp_op_count);

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03;

clean_up:

    return;
}


void test_f64_mul_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03, f64_04, f64_05, f64_06, f64_07;
    double f64_08, f64_09, f64_10, f64_11;
    double f64_100, f64_101, f64_102;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  p/431.2;
    f64_01 = -p/431.3;
    f64_02 =  p/431.4;
    f64_03 = -p/431.5;
    f64_04 =  p/431.6;
    f64_05 = -p/431.7;
    f64_06 =  p/431.8;
    f64_07 = -p/431.9;
    f64_08 =  p/432.0;
    f64_09 = -p/432.1;
    f64_10 =  p/432.2;
    f64_11 = -p/432.3;

    f64_100 =  1.00001;
    f64_101 = -1.00002;
    f64_102 =  1.00003;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p == 123456 ){
        p /= 2;
        f64_100 *= 1.0045;
        f64_101 *= 1.0054;
        f64_102 *= 1.0067;
    }

#define F64_MULS(_X) {f64_00 *= _X; f64_01 *= _X; f64_02 *= _X; f64_03 *= _X; f64_04 *= _X; f64_05 *= _X; f64_06 *= _X; f64_07 *= _X; f64_08 *= _X; f64_09 *= _X; f64_10 *= _X; f64_11 *= _X;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F64_MULS(f64_100);
            F64_MULS(f64_101);
            F64_MULS(f64_102);
            if( p < 2 ){
                F64_MULS(f64_00);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 12LL*3LL*N*M, (double)ev_values[0]/(12.0*3.0*N*M));

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03 + f64_04 + f64_05 + f64_06 + f64_07;
    sum_f64 += f64_08 + f64_09 + f64_10 + f64_11;

clean_up:

    return;
}

////////////////////////////////////////////////////////////////////////////////
// f64 DIVISION

void test_f64_div(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  1.0 + 1.0/(1000.1*(double)p);
    f64_01 =  1.0 + 1.0/(1000.2*(double)p);
    f64_02 =  1.0 + 1.0/(1000.3*(double)p);
    f64_03 =  1.0 + 1.0/(1000.4*(double)p);

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

#define F64DIV_BLOCK() {f64_01 /= f64_00; f64_02 /= f64_01; f64_03 /= f64_02; f64_00 /= f64_03;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F64DIV_BLOCK();
            F64DIV_BLOCK();
            F64DIV_BLOCK();
            F64DIV_BLOCK();
            F64DIV_BLOCK();
            F64DIV_BLOCK();
            F64DIV_BLOCK();
            F64DIV_BLOCK();
            F64DIV_BLOCK();
            F64DIV_BLOCK();
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    long long int fp_op_count = 40LL*N*M; // There are only 50 FP operations.
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], fp_op_count, (double)ev_values[0]/(double)fp_op_count);

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03;

clean_up:

    return;
}


void test_f64_div_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03, f64_04, f64_05, f64_06, f64_07;
    double f64_08, f64_09, f64_10, f64_11;
    double f64_100, f64_101, f64_102;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  p/431.2;
    f64_01 = -p/431.3;
    f64_02 =  p/431.4;
    f64_03 = -p/431.5;
    f64_04 =  p/431.6;
    f64_05 = -p/431.7;
    f64_06 =  p/431.8;
    f64_07 = -p/431.9;
    f64_08 =  p/432.0;
    f64_09 = -p/432.1;
    f64_10 =  p/432.2;
    f64_11 = -p/432.3;

    f64_100 =  1.00001;
    f64_101 = -1.00002;
    f64_102 =  1.00003;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p == 123456 ){
        p /= 2;
        f64_100 *= 1.0045;
        f64_101 *= 1.0054;
        f64_102 *= 1.0067;
    }

#define F64_DIVS(_X) {f64_00 /= _X; f64_01 /= _X; f64_02 /= _X; f64_03 /= _X; f64_04 /= _X; f64_05 /= _X; f64_06 /= _X; f64_07 /= _X; f64_08 /= _X; f64_09 /= _X; f64_10 /= _X; f64_11 /= _X;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F64_DIVS(f64_100);
            F64_DIVS(f64_101);
            F64_DIVS(f64_102);
            if( p < 2 ){
                F64_DIVS(f64_00);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 12LL*3LL*N*M, (double)ev_values[0]/(12.0*3.0*N*M));

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03 + f64_04 + f64_05 + f64_06 + f64_07;
    sum_f64 += f64_08 + f64_09 + f64_10 + f64_11;

clean_up:

    return;
}


void test_f64_fma_max(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
    double f64_00, f64_01, f64_02, f64_03, f64_04, f64_05, f64_06, f64_07;
    double f64_08, f64_09, f64_10, f64_11;
    double f64_100, f64_101, f64_102;
    double f64_B;

    /* Initialize the variables with values that the compiler cannot guess. */
    f64_00 =  p/431.2;
    f64_01 = -p/431.3;
    f64_02 =  p/431.4;
    f64_03 = -p/431.5;
    f64_04 =  p/431.6;
    f64_05 = -p/431.7;
    f64_06 =  p/431.8;
    f64_07 = -p/431.9;
    f64_08 =  p/432.0;
    f64_09 = -p/432.1;
    f64_10 =  p/432.2;
    f64_11 = -p/432.3;

    f64_100 =  1.00001;
    f64_101 = -1.00002;
    f64_102 =  1.00003;

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    if( p != 12345678 ){
        f64_100 /= 1.000045;
        f64_101 /= 1.000054;
        f64_102 /= 1.000067;
    }
    f64_B = f64_100/34567.8;

#define F64_FMAS(_A,_B) {f64_00 = _A*f64_00+_B; f64_01 = _A*f64_01+_B; f64_02 = _A*f64_02+_B; f64_03 = _A*f64_03+_B; f64_04 = _A*f64_04+_B; f64_05 = _A*f64_05+_B; f64_06 = _A*f64_06+_B; f64_07 = _A*f64_07+_B; f64_08 = _A*f64_08+_B; f64_09 = _A*f64_09+_B; f64_10 = _A*f64_10+_B; f64_11 = _A*f64_11+_B;}

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            F64_FMAS(f64_100, f64_B);
            F64_FMAS(f64_101, f64_B);
            F64_FMAS(f64_102, f64_B);
            if( p < 2 ){
                F64_FMAS(f64_00, f64_B);
            }
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 12LL*3LL*N*M, (double)ev_values[0]/(12.0*3.0*N*M));

    sum_f64 += f64_00 + f64_01 + f64_02 + f64_03 + f64_04 + f64_05 + f64_06 + f64_07;
    sum_f64 += f64_08 + f64_09 + f64_10 + f64_11;

clean_up:

    return;
}

////////////////////////////////////////////////////////////////////////////////
// Vector double precision ADD

void test_f64_add_DVEC128(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE 512+2
    double a[BUFFER_SIZE];
    double b[BUFFER_SIZE];

    /* Initialize the arrays with values that the compiler cannot guess. */
    for(int i=0; i<BUFFER_SIZE; i++){
        a[i] =  p/(i+1.2);
        b[i] = -p/(i+1.3);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    long long int UB=1LL*M*N/(BUFFER_SIZE-2);
    for(int i=0; i<UB; i++){
        for(int j=2; j<BUFFER_SIZE; j++){
            a[j] = a[j-2] + b[j];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], N*M/2LL, (double)ev_values[0]/(N*M/2.0));

    for(int j=0; j<BUFFER_SIZE; j++){
        sum_f64 += a[j];
    }

clean_up:

    return;
}

void test_f64_add_DVEC256(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE 512+4
    double a[BUFFER_SIZE];
    double b[BUFFER_SIZE];

    /* Initialize the arrays with values that the compiler cannot guess. */
    for(int i=0; i<BUFFER_SIZE; i++){
        a[i] =  p/(i+1.2);
        b[i] = -p/(i+1.3);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    long long int UB=1LL*M*N/(BUFFER_SIZE-4);
    for(int i=0; i<UB; i++){
        for(int j=4; j<BUFFER_SIZE; j++){
            a[j] = a[j-4] + b[j];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], N*M/4LL, (double)ev_values[0]/(N*M/4.0));

    for(int j=0; j<BUFFER_SIZE; j++){
        sum_f64 += a[j];
    }

clean_up:

    return;
}

void test_f64_add_DVEC512(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE 512+8
    double a[BUFFER_SIZE];
    double b[BUFFER_SIZE];

    /* Initialize the arrays with values that the compiler cannot guess. */
    for(int i=0; i<BUFFER_SIZE; i++){
        a[i] =  p/(i+1.2);
        b[i] = -p/(i+1.3);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    long long int UB=1LL*M*N/(BUFFER_SIZE-8);
    for(int i=0; i<UB; i++){
        for(int j=8; j<BUFFER_SIZE; j++){
            a[j] = a[j-8] + b[j];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], N*M/8LL, (double)ev_values[0]/(N*M/8.0));

    for(int j=0; j<BUFFER_SIZE; j++){
        sum_f64 += a[j];
    }

clean_up:

    return;
}

////////////////////////////////////////////////////////////////////////////////
// Vector single precision ADD

void test_f64_add_SVEC128(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE 512+4
    float a[BUFFER_SIZE];
    float b[BUFFER_SIZE];

    /* Initialize the arrays with values that the compiler cannot guess. */
    for(int i=0; i<BUFFER_SIZE; i++){
        a[i] =  p/(i+1.2);
        b[i] = -p/(i+1.3);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    long long int UB=1LL*M*N/(BUFFER_SIZE-4);
    for(int i=0; i<UB; i++){
        for(int j=4; j<BUFFER_SIZE; j++){
            a[j] = a[j-4] + b[j];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], N*M/4LL, (double)ev_values[0]/(N*M/4.0));

    for(int j=0; j<BUFFER_SIZE; j++){
        sum_f64 += (double)a[j];
    }

clean_up:

    return;
}

void test_f64_add_SVEC256(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE 512+8
    float a[BUFFER_SIZE];
    float b[BUFFER_SIZE];

    /* Initialize the arrays with values that the compiler cannot guess. */
    for(int i=0; i<BUFFER_SIZE; i++){
        a[i] =  p/(i+1.2);
        b[i] = -p/(i+1.3);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    long long int UB=1LL*M*N/(BUFFER_SIZE-8);
    for(int i=0; i<UB; i++){
        for(int j=8; j<BUFFER_SIZE; j++){
            a[j] = a[j-8] + b[j];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], N*M/8LL, (double)ev_values[0]/(N*M/8.0));

    for(int j=0; j<BUFFER_SIZE; j++){
        sum_f64 += (double)a[j];
    }

clean_up:

    return;
}

void test_f64_add_SVEC512(int p, int M, int N, int EventSet, FILE *fp){
    int ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE 512+16
    float a[BUFFER_SIZE];
    float b[BUFFER_SIZE];

    /* Initialize the arrays with values that the compiler cannot guess. */
    for(int i=0; i<BUFFER_SIZE; i++){
        a[i] =  p/(i+1.2);
        b[i] = -p/(i+1.3);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    long long int UB=1LL*M*N/(BUFFER_SIZE-16);
    for(int i=0; i<UB; i++){
        for(int j=16; j<BUFFER_SIZE; j++){
            a[j] = a[j-16] + b[j];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], N*M/16LL, (double)ev_values[0]/(N*M/16.0));

    for(int j=0; j<BUFFER_SIZE; j++){
        sum_f64 += (double)a[j];
    }

clean_up:

    return;
}


////////////////////////////////////////////////////////////////////////////////
// MEM ops

void test_mem_ops_serial_RO(int p, int M, int N, int EventSet, FILE *fp){
    int i, ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE 256
    int buffer[BUFFER_SIZE];

    /* Initialize the buffer with values that the compiler cannot guess. */
    for(i=0; i<BUFFER_SIZE; i++){
        buffer[i] = p/(1223+i);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    uintptr_t index = buffer[0];
    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            index = buffer[index%BUFFER_SIZE];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 1LL*N*M, (double)ev_values[0]/(1.0*N*M));

    sum_i32 += index;

clean_up:

    return;
}

void test_mem_ops_serial_RW(int p, int M, int N, int EventSet, FILE *fp){
    int i, ret;
    long long int ev_values[2];
#undef BUFFER_SIZE
#define BUFFER_SIZE (256+1)
    int buffer[BUFFER_SIZE];

    /* Initialize the buffer with values that the compiler cannot guess. */
    for(i=0; i<BUFFER_SIZE; i++){
        buffer[i] = p/(1223+i);
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            uintptr_t index = j%(BUFFER_SIZE-1);
            buffer[index+1] += buffer[index];
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 2LL*N*M, (double)ev_values[0]/(2.0*N*M));

    sum_i32 += buffer[0] + buffer[BUFFER_SIZE/2] + buffer[BUFFER_SIZE-1];

clean_up:

    return;
}

void test_mem_ops_parallel_RO(int p, int M, int N, int EventSet, FILE *fp){
    int i, ret;
    long long int ev_values[2];
    int c0, c1, c2, c3;
#undef BUFFER_SIZE
#define BUFFER_SIZE (256+8)
    int buffer[BUFFER_SIZE];

    /* Initialize the buffer with values that the compiler cannot guess. */
    for(i=0; i<BUFFER_SIZE; i++){
        buffer[i] = p/(223+i);
    }

    /* Initialize the variables with values that the compiler cannot guess. */
    c0 = (int)((5+p)/(12+1));
    c1 = (int)((7+p)/(12+2));
    c2 = (int)((11+p)/(12+3));
    c3 = (int)((13+p)/(12+4));

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    for(int i=0; i<M; i++){
        // compute some junk value.
        uintptr_t base = i*(c0+c1)/(c2+c3+1);
        for(int j=0; j<N; j++){
            uintptr_t offset = (base+j)%(BUFFER_SIZE-8);

            c0 += buffer[offset+1];
            c1 += buffer[offset+2];
            c2 += buffer[offset+3];
            c3 += buffer[offset+4];

            c0 += buffer[offset+5];
            c1 += buffer[offset+6];
            c2 += buffer[offset+7];
            c3 += buffer[offset+8];

        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 8LL*N*M, (double)ev_values[0]/(8.0*N*M));

    sum_i32 += c0+c1+c2+c3;

clean_up:

    return;
}

void test_mem_ops_parallel_WO(int p, int M, int N, int EventSet, FILE *fp){
    int i, ret;
    long long int ev_values[2];
    int sum=0;
#undef BUFFER_SIZE
#define BUFFER_SIZE 256
    int buffer[BUFFER_SIZE];

    /* Initialize the buffer with values that the compiler cannot guess. */
    for(i=0; i<BUFFER_SIZE; i++){
        buffer[i] = p/(1223+i);
        sum += buffer[i];
    }

    // Start the counters.
    ret = PAPI_start(EventSet);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_start() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to run the kernel.
        goto clean_up;
    }

    for(int i=0; i<M; i++){
        for(int j=0; j<N; j++){
            buffer[j%BUFFER_SIZE] = sum;
        }
    }

    ret = PAPI_stop(EventSet, ev_values);
    if ( PAPI_OK != ret ) {
        fprintf(stderr, "PAPI_stop() error: %s\n", PAPI_strerror(ret));
        // If we can't measure events, no need to print anything.
        goto clean_up;
    }
    fprintf(fp, "%d %lld %lld %.3lf\n", N*M, ev_values[0], 1LL*N*M, (double)ev_values[0]/(1.0*N*M));

    sum_i32 += buffer[0] + buffer[BUFFER_SIZE/2] + buffer[BUFFER_SIZE-1];

clean_up:

    return;
}

////////////////////////////////////////////////////////////////////////////////
//////// Main driver

void instr_test(int EventSet, FILE *fp) {
    int i, j, M, N;
    int minM=64, minN=64;
    double f[4] = {1.0, 1.1892, 1.4142, 1.6818};
    int p = (int)getpid();


    ////////////////////////////////////////
    fprintf(fp,"# Mem_RO\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_mem_ops_serial_RO(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# Mem_RW\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_mem_ops_serial_RW(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# Mem_RO(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_mem_ops_parallel_RO(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# Mem_WO(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_mem_ops_parallel_WO(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# Int_ADD\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_int_add(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# Int_ADD(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_int_add_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# Int_MUL(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_int_mul_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# Int_DIV(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_int_div_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP32_ADD\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f32_add(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP32_ADD(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f32_add_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP32_SUB(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f32_sub(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP32_SUB(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f32_sub_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP32_MUL\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f32_mul(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP32_MUL(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f32_mul_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP32_DIV\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM/4);
            N = (int)(i*f[j]*minN/4);
            test_f32_div(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP32_DIV(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM/4);
            N = (int)(i*f[j]*minN/4);
            test_f32_div_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP64_ADD\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f64_add(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP64_ADD(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f64_add_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP64_SUB(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f64_sub(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP64_SUB(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f64_sub_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP64_MUL\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f64_mul(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP64_MUL(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM);
            N = (int)(i*f[j]*minN);
            test_f64_mul_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP64_DIV\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM/4);
            N = (int)(i*f[j]*minN/4);
            test_f64_div(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP64_DIV(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM/4);
            N = (int)(i*f[j]*minN/4);
            test_f64_div_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP32_FMA(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM/4);
            N = (int)(i*f[j]*minN/4);
            test_f32_fma_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# FP64_FMA(ILP)\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM/4);
            N = (int)(i*f[j]*minN/4);
            test_f64_fma_max(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# SVEC128\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM*2);
            N = (int)(i*f[j]*minN*2);
            test_f64_add_SVEC128(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# SVEC256\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM*2);
            N = (int)(i*f[j]*minN*2);
            test_f64_add_SVEC256(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# SVEC512\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM*2);
            N = (int)(i*f[j]*minN*2);
            test_f64_add_SVEC512(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# DVEC128\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM*2);
            N = (int)(i*f[j]*minN*2);
            test_f64_add_DVEC128(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# DVEC256\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM*2);
            N = (int)(i*f[j]*minN*2);
            test_f64_add_DVEC256(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    ////////////////////////////////////////
    fprintf(fp,"# DVEC512\n");
    for(i=16; i<50; i*=2){
        for(j=0; j<4; j++){
            M = (int)(i*f[j]*minM*2);
            N = (int)(i*f[j]*minN*2);
            test_f64_add_DVEC512(p, M, N, EventSet, fp);
        }
    }
    fprintf(fp, "\n");

    if( sum_i32 == 12345 && sum_f64 == 12.345)
        fprintf(stderr, "Side-effect to disable dead code elimination by the compiler. Please ignore.\n");
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

void instr_driver(char* papi_event_name, hw_desc_t *hw_desc, char* outdir)
{
    int retval = PAPI_OK;
    int EventSet = PAPI_NULL;
    FILE* ofp_papi;
    const char *sufx = ".instr";
    char *papiFileName;

    (void)hw_desc;

    int l = strlen(outdir)+strlen(papi_event_name)+strlen(sufx);
    if (NULL == (papiFileName = (char *)calloc( 1+l, sizeof(char)))) {
        return;
    }
    if (l != (sprintf(papiFileName, "%s%s%s", outdir, papi_event_name, sufx))) {
        goto error0;
    }
    if (NULL == (ofp_papi = fopen(papiFileName,"w"))) {
        fprintf(stderr, "Failed to open file %s.\n", papiFileName);
        goto error0;
    }
  
    retval = PAPI_create_eventset( &EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }

    retval = PAPI_add_named_event( EventSet, papi_event_name );
    if (retval != PAPI_OK ){
        goto error1;
    }

    retval = PAPI_OK;

    instr_test(EventSet, ofp_papi);

    retval = PAPI_cleanup_eventset( EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }
    retval = PAPI_destroy_eventset( &EventSet );
    if (retval != PAPI_OK ){
        goto error1;
    }

error1:
    fclose(ofp_papi);
error0:
    free(papiFileName);
    return;
}
