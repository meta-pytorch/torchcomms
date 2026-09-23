#define _GNU_SOURCE
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <papi.h>
#include "flops.h"

#define DOUBLE 2
#define SINGLE 1
#define HALF   0

#define CHOLESKY  3
#define GEMM      2
#define NORMALIZE 1

#define MAXDIM 51

#if defined(mips)
#define FMA 1
#elif (defined(sparc) && defined(sun))
#define FMA 1
#else
#define FMA 0
#endif

/* Function prototypes. */
void print_header( FILE *fp, char *prec, char *kernel );
void resultline( int i, int kernel, int EventSet, FILE *fp );
void exec_flops( int precision, int EventSet, FILE *fp );

double normalize_double( int n, double *xd );
void cholesky_double( int n, double *ld, double *ad );
void exec_double_norm( int EventSet, FILE *fp );
void exec_double_cholesky( int EventSet, FILE *fp );
void exec_double_gemm( int EventSet, FILE *fp );
void keep_double_vec_res( int n, double *xd );
void keep_double_mat_res( int n, double *ld );

float normalize_single( int n, float *xs );
void cholesky_single( int n, float  *ls, float *as );
void exec_single_norm( int EventSet, FILE *fp );
void exec_single_cholesky( int EventSet, FILE *fp );
void exec_single_gemm( int EventSet, FILE *fp );
void keep_single_vec_res( int n, float *xs );
void keep_single_mat_res( int n, float *ls );

#if defined(ARM)
half normalize_half( int n, half *xh );
void cholesky_half( int n, half *lh, half *ah );
void exec_half_norm( int EventSet, FILE *fp );
void exec_half_cholesky( int EventSet, FILE *fp );
void exec_half_gemm( int EventSet, FILE *fp );
void keep_half_vec_res( int n, half *xh );
void keep_half_mat_res( int n, half *lh );
#endif

void print_header( FILE *fp, char *prec, char *kernel ) {

    fprintf(fp, "#%s %s\n", prec, kernel);
    fprintf(fp, "#N RawEvtCnt NormdEvtCnt ExpectedAdd ExpectedSub ExpectedMul ExpectedDiv ExpectedSqrt ExpectedFMA ExpectedTotal\n");
}

void resultline( int i, int kernel, int EventSet, FILE *fp ) {

    long long flpins = 0, denom;
    long long papi, all, add, sub, mul, div, sqrt, fma;
    int retval;

    if ( (retval=PAPI_stop(EventSet, &flpins)) != PAPI_OK ) {
        return;
    }

    switch(kernel) {
      case NORMALIZE:
          all  = 3*i+1;
          denom = all;
          add  = i;
          sub  = 0;
          mul  = i;
          div  = i;
          if ( 0 == i ) {
              sqrt = 0;
          } else {
              sqrt = 1;
          }
          fma  = 0;
          break;
      case GEMM:
          all  = 2*i*i*i;
          if ( 0 == i ) {
              denom = 1;
          } else {
              denom = all;
          }
          add  = 0;
          sub  = 0;
          mul  = 0;
          div  = 0;
          sqrt = 0;
          fma  = i*i*i; // Need to derive.
          break;
      case CHOLESKY:
          all  = i*(2*i*i+9*i+1)/6.0;
          if ( 0 == i ) {
              denom = 1;
          } else {
              denom = all;
          }
          add  = i*(i-1)*(i+1)/6.0;
          sub  = i*(i+1)/2.0;
          mul  = i*(i-1)*(i+4)/6.0;
          div  = i*(i-1)/2.0;
          sqrt = i;
          fma  = 0;
          break;
      default:
          all   = -1;
          denom = -1;
          add   = -1;
          sub   = -1;
          mul   = -1;
          div   = -1;
          sqrt  = -1;
          fma   = -1;
    }

    papi = flpins << FMA;

    fprintf(fp, "%d %lld %.17g %lld %lld %lld %lld %lld %lld %lld\n", i, papi, ((double)papi)/((double)denom), add, sub, mul, div, sqrt, fma, all);
}

#if defined(ARM)

half normalize_half( int n, half *xh ) {

    if ( 0 == n )
        return 0.0;

    half aa = 0.0;
    half buff = 0.0;
    int i;

    for ( i = 0; i < n; i++ ) {
        buff = xh[i] * xh[i];
        aa += buff;
    }

    aa = SQRT_VEC_SH(aa);
    for ( i = 0; i < n; i++ )
        xh[i] = xh[i]/aa;

    return ( aa );
}

void cholesky_half( int n, half *lh, half *ah ) {

    int i, j, k;
    half sum = 0.0;
    half buff = 0.0;

    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
            sum = 0.0;
            for (k = 0; k < j; k++) {
                buff = lh[i * n + k] * lh[j * n + k];
                sum += buff;
            }

            if( i == j ) {
                buff = ah[i * n + i] - sum;
                lh[i * n + j] = SQRT_VEC_SH(buff);
            } else {
                buff = ah[i * n + i] - sum;
                sum = ((half)1.0);
                sum = sum/lh[j * n + j];
                lh[i * n + j] = sum * buff;
            }
        }
    }
}

void gemm_half( int n, half *ch, half *ah, half *bh ) {

    int i, j, k;
    half sum = 0.0;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            sum = 0.0;
            for (k = 0; k < n; k++) {
                FMA_VEC_SH(sum, ah[i * n + k], bh[k * n + j], sum);
            }
            ch[i * n + j] = sum;
        }
    }
}
#endif

float normalize_single( int n, float *xs ) {

    if ( 0 == n )
        return 0.0;

    float aa = 0.0;
    int i;

    for ( i = 0; i < n; i++ )
        aa = aa + xs[i] * xs[i];

    aa = sqrtf(aa);
    for ( i = 0; i < n; i++ )
        xs[i] = xs[i]/aa;

    return ( aa );
}

void cholesky_single( int n, float *ls, float *as ) {

    int i, j, k;
    float sum = 0.0;

    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
            sum = 0.0;
            for (k = 0; k < j; k++) {
                sum += ls[i * n + k] * ls[j * n + k];
            }

            if( i == j ) {
                ls[i * n + j] = sqrtf(as[i * n + i] - sum);
            } else {
                ls[i * n + j] = ((float)1.0)/ls[j * n + j] * (as[i * n + j] - sum);
            }
        }
    }
}

void gemm_single( int n, float *cs, float *as, float *bs ) {

    int i, j, k;
    SP_SCALAR_TYPE argI, argJ, argK;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            argK = SET_VEC_SS(0.0);
            for (k = 0; k < n; k++) {
                argI = SET_VEC_SS(as[i * n + k]);
                argJ = SET_VEC_SS(bs[k * n + j]);
                FMA_VEC_SS(argK, argI, argJ, argK);
            }
            cs[i * n + j] = ((float*)&argK)[0];
        }
    }
}

double normalize_double( int n, double *xd ) {

    if ( 0 == n )
        return 0.0;

    double aa = 0.0;
    int i;

    for ( i = 0; i < n; i++ )
        aa = aa + xd[i] * xd[i];

    aa = sqrt(aa);
    for ( i = 0; i < n; i++ )
        xd[i] = xd[i]/aa;

    return ( aa );
}

void cholesky_double( int n, double *ld, double *ad ) {

    int i, j, k;
    double sum = 0.0;

    for (i = 0; i < n; i++) {
        for (j = 0; j <= i; j++) {
            sum = 0.0;
            for (k = 0; k < j; k++) {
                sum += ld[i * n + k] * ld[j * n + k];
            }

            if( i == j ) {
                ld[i * n + j] = sqrt(ad[i * n + i] - sum);
            } else {
                ld[i * n + j] = ((double)1.0)/ld[j * n + j] * (ad[i * n + j] - sum);
            }
        }
    }
}


void gemm_double( int n, double *cd, double *ad, double *bd ) {

    int i, j, k;
    DP_SCALAR_TYPE argI, argJ, argK;

    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            argK = SET_VEC_SD(0.0);
            for (k = 0; k < n; k++) {
                argI = SET_VEC_SD(ad[i * n + k]);
                argJ = SET_VEC_SD(bd[k * n + j]);
                FMA_VEC_SD(argK, argI, argJ, argK);
            }
            cd[i * n + j] = ((double*)&argK)[0];
        }
    }
}

void exec_double_norm( int EventSet, FILE *fp ) {

    int i, n, retval;
    double *xd=NULL;

    /* Print info about the computational kernel. */
    print_header( fp, "Double-Precision", "Vector Normalization" );

    /* Allocate the linear arrays. */
    xd = malloc( MAXDIM * sizeof(double) );

    /* Step through the different array sizes. */
    for ( n = 0; n < MAXDIM; n++ ) {
        /* Initialize the needed arrays at this size. */
        for ( i = 0; i < n; i++ ) {
            xd[i] = ((double)random())/((double)RAND_MAX) * (double)1.1;
        }

        /* Reset PAPI count. */
        if ( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
            return;
        }

        /* Run the kernel. */
        normalize_double( n, xd );
        usleep(1);

        /* Stop and print count. */
        resultline( n, NORMALIZE, EventSet, fp );

        keep_double_vec_res( n, xd );
    }

    /* Free dynamically allocated memory. */
    free( xd );
}

void exec_double_cholesky( int EventSet, FILE *fp ) {

    int i, j, n, retval;
    double *ad=NULL, *ld=NULL;
    double sumd = 0.0;

    /* Print info about the computational kernel. */
    print_header( fp, "Double-Precision", "Cholesky Decomposition" );

    /* Allocate the matrices. */
    ad = malloc( MAXDIM * MAXDIM * sizeof(double) );
    ld = malloc( MAXDIM * MAXDIM * sizeof(double) );

    /* Step through the different array sizes. */
    for ( n = 0; n < MAXDIM; n++ ) {
        /* Initialize the needed arrays at this size. */
        for ( i = 0; i < n; i++ ) {
            for ( j = 0; j < i; j++ ) {
                ld[i * n + j] = 0.0;
                ld[j * n + i] = 0.0;

                ad[i * n + j] = ((double)random())/((double)RAND_MAX) * (double)1.1;
                ad[j * n + i] = ad[i * n + j];
            }
            ad[i * n + i] = 0.0;
            ld[i * n + i] = 0.0;
        }

        /* Guarantee diagonal dominance for successful Cholesky. */
        for ( i = 0; i < n; i++ ) {
            sumd = 0.0;
            for ( j = 0; j < n; j++ ) {
                sumd += fabs(ad[i * n + j]);
            }
            ad[i * n + i] = sumd + (double)1.1;
        }

        /* Reset PAPI count. */
        if ( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
            return;
        }

        /* Run the kernel. */
        cholesky_double( n, ld, ad );
        usleep(1);

        /* Stop and print count. */
        resultline( n, CHOLESKY, EventSet, fp );

        keep_double_mat_res( n, ld );
    }

    free( ad );
    free( ld );
}

void exec_double_gemm( int EventSet, FILE *fp ) {

    int i, j, n, retval;
    double *ad=NULL, *bd=NULL, *cd=NULL;

    /* Print info about the computational kernel. */
    print_header( fp, "Double-Precision", "GEMM" );

    /* Allocate the matrices. */
    ad = malloc( MAXDIM * MAXDIM * sizeof(double) );
    bd = malloc( MAXDIM * MAXDIM * sizeof(double) );
    cd = malloc( MAXDIM * MAXDIM * sizeof(double) );

    /* Step through the different array sizes. */
    for ( n = 0; n < MAXDIM; n++ ) {
        /* Initialize the needed arrays at this size. */
        for ( i = 0; i < n; i++ ) {
            for ( j = 0; j < n; j++ ) {
                cd[i * n + j] = 0.0;
                ad[i * n + j] = ((double)random())/((double)RAND_MAX) * (double)1.1;
                bd[i * n + j] = ((double)random())/((double)RAND_MAX) * (double)1.1;
            }
        }

        /* Reset PAPI count. */
        if ( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
            return;
        }

        /* Run the kernel. */
        gemm_double( n, cd, ad, bd );
        usleep(1);

        /* Stop and print count. */
        resultline( n, GEMM, EventSet, fp );

        keep_double_mat_res( n, cd );
    }

    free( ad );
    free( bd );
    free( cd );
}

void keep_double_vec_res( int n, double *xd ) {

    int i;
    double sum = 0.0;
    for( i = 0; i < n; ++i ) {
        sum += xd[i];
    }
    
    if( 1.2345 == sum ) {
        fprintf(stderr, "Side-effect to disable dead code elimination by the compiler. Please ignore.\n");
    }
}

void keep_double_mat_res( int n, double *ld ) {

    int i, j;
    double sum = 0.0;
    for( i = 0; i < n; ++i ) {
        for( j = 0; j < n; ++j ) {
            sum += ld[i * n + j];
        }
    }
    
    if( 1.2345 == sum ) {
        fprintf(stderr, "Side-effect to disable dead code elimination by the compiler. Please ignore.\n");
    }
}

void exec_single_norm( int EventSet, FILE *fp ) {

    int i, n, retval;
    float *xs=NULL;

    /* Print info about the computational kernel. */
    print_header( fp, "Single-Precision", "Vector Normalization" );

    /* Allocate the linear arrays. */
    xs = malloc( MAXDIM * sizeof(float) );

    /* Step through the different array sizes. */
    for ( n = 0; n < MAXDIM; n++ ) {
        /* Initialize the needed arrays at this size. */
        for ( i = 0; i < n; i++ ) {
            xs[i] = ((float)random())/((float)RAND_MAX) * (float)1.1;
        }

        /* Reset PAPI count. */
        if ( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
            return;
        }

        /* Run the kernel. */
        normalize_single( n, xs );
        usleep(1);

        /* Stop and print count. */
        resultline( n, NORMALIZE, EventSet, fp );

        keep_single_vec_res( n, xs );
    }

    /* Free dynamically allocated memory. */
    free( xs );
}

void exec_single_cholesky( int EventSet, FILE *fp ) {

    int i, j, n, retval;
    float *as=NULL, *ls=NULL;
    float sums = 0.0;

    /* Print info about the computational kernel. */
    print_header( fp, "Single-Precision", "Cholesky Decomposition" );

    /* Allocate the matrices. */
    as = malloc( MAXDIM * MAXDIM * sizeof(float) );
    ls = malloc( MAXDIM * MAXDIM * sizeof(float) );

    /* Step through the different array sizes. */
    for ( n = 0; n < MAXDIM; n++ ) {
        /* Initialize the needed arrays at this size. */
        for ( i = 0; i < n; i++ ) {
            for ( j = 0; j < i; j++ ) {
                ls[i * n + j] = 0.0;
                ls[j * n + i] = 0.0;

                as[i * n + j] = ((float)random())/((float)RAND_MAX) * (float)1.1;
                as[j * n + i] = as[i * n + j];
            }
            as[i * n + i] = 0.0;
            ls[i * n + i] = 0.0;
        }

        /* Guarantee diagonal dominance for successful Cholesky. */
        for ( i = 0; i < n; i++ ) {
            sums = 0.0;
            for ( j = 0; j < n; j++ ) {
                sums += fabs(as[i * n + j]);
            }
            as[i * n + i] = sums + (float)1.1;
        }

        /* Reset PAPI count. */
        if ( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
            return;
        }

        /* Run the kernel. */
        cholesky_single( n, ls, as );
        usleep(1);

        /* Stop and print count. */
        resultline( n, CHOLESKY, EventSet, fp );

        keep_single_mat_res( n, ls );
    }

    free( as );
    free( ls );
}

void exec_single_gemm( int EventSet, FILE *fp ) {

    int i, j, n, retval;
    float *as=NULL, *bs=NULL, *cs=NULL;

    /* Print info about the computational kernel. */
    print_header( fp, "Single-Precision", "GEMM" );

    /* Allocate the matrices. */
    as = malloc( MAXDIM * MAXDIM * sizeof(float) );
    bs = malloc( MAXDIM * MAXDIM * sizeof(float) );
    cs = malloc( MAXDIM * MAXDIM * sizeof(float) );

    /* Step through the different array sizes. */
    for ( n = 0; n < MAXDIM; n++ ) {
        /* Initialize the needed arrays at this size. */
        for ( i = 0; i < n; i++ ) {
            for ( j = 0; j < n; j++ ) {
                cs[i * n + j] = 0.0;
                as[i * n + j] = ((float)random())/((float)RAND_MAX) * (float)1.1;
                bs[i * n + j] = ((float)random())/((float)RAND_MAX) * (float)1.1;
            }
        }

        /* Reset PAPI count. */
        if ( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
            return;
        }

        /* Run the kernel. */
        gemm_single( n, cs, as, bs );
        usleep(1);

        /* Stop and print count. */
        resultline( n, GEMM, EventSet, fp );

        keep_single_mat_res( n, cs );
    }

    free( as );
    free( bs );
    free( cs );
}

void keep_single_vec_res( int n, float *xs ) {

    int i;
    float sum = 0.0;
    for( i = 0; i < n; ++i ) {
        sum += xs[i];
    }
    
    if( 1.2345 == sum ) {
        fprintf(stderr, "Side-effect to disable dead code elimination by the compiler. Please ignore.\n");
    }
}

void keep_single_mat_res( int n, float *ls ) {

    int i, j;
    float sum = 0.0;
    for( i = 0; i < n; ++i ) {
        for( j = 0; j < n; ++j ) {
            sum += ls[i * n + j];
        }
    }
    
    if( 1.2345 == sum ) {
        fprintf(stderr, "Side-effect to disable dead code elimination by the compiler. Please ignore.\n");
    }
}

#if defined(ARM)
void exec_half_norm( int EventSet, FILE *fp ) {

    int i, n, retval;
    half *xh=NULL;

    /* Print info about the computational kernel. */
    print_header( fp, "Half-Precision", "Vector Normalization" );

    /* Allocate the linear arrays. */
    xh = malloc( MAXDIM * sizeof(half) );

    /* Step through the different array sizes. */
    for ( n = 0; n < MAXDIM; n++ ) {
        /* Initialize the needed arrays at this size. */
        for ( i = 0; i < n; i++ ) {
            xh[i] = ((half)random())/((half)RAND_MAX) * (half)1.1;
        }

        /* Reset PAPI count. */
        if ( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
            return;
        }

        /* Run the kernel. */
        normalize_half( n, xh );
        usleep(1);

        /* Stop and print count. */
        resultline( n, NORMALIZE, EventSet, fp );

        keep_half_vec_res( n, xh );
    }

    /* Free dynamically allocated memory. */
    free( xh );
}

void exec_half_cholesky( int EventSet, FILE *fp ) {

    int i, j, n, retval;
    half *ah=NULL, *lh=NULL;
    half sumh = 0.0;

    /* Print info about the computational kernel. */
    print_header( fp, "Half-Precision", "Cholesky Decomposition" );

    /* Allocate the matrices. */
    ah = malloc( MAXDIM * MAXDIM * sizeof(half) );
    lh = malloc( MAXDIM * MAXDIM * sizeof(half) );

    /* Step through the different array sizes. */
    for ( n = 0; n < MAXDIM; n++ ) {
        /* Initialize the needed arrays at this size. */
        for ( i = 0; i < n; i++ ) {
            for ( j = 0; j < i; j++ ) {
                lh[i * n + j] = 0.0;
                lh[j * n + i] = 0.0;

                ah[i * n + j] = ((half)random())/((half)RAND_MAX) * (half)1.1;
                ah[j * n + i] = ah[i * n + j];
            }
            ah[i * n + i] = 0.0;
            lh[i * n + i] = 0.0;
        }

        /* Guarantee diagonal dominance for successful Cholesky. */
        for ( i = 0; i < n; i++ ) {
            sumh = 0.0;
            for ( j = 0; j < n; j++ ) {
                sumh += fabs(ah[i * n + j]);
            }
            ah[i * n + i] = sumh + (half)1.1;
        }

        /* Reset PAPI count. */
        if ( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
            return;
        }

        /* Run the kernel. */
        cholesky_half( n, lh, ah );
        usleep(1);

        /* Stop and print count. */
        resultline( n, CHOLESKY, EventSet, fp );

        keep_half_mat_res( n, lh );
    }

    free( ah );
    free( lh );
}

void exec_half_gemm( int EventSet, FILE *fp ) {

    int i, j, n, retval;
    half *ah=NULL, *bh=NULL, *ch=NULL;

    /* Print info about the computational kernel. */
    print_header( fp, "Half-Precision", "GEMM" );

    /* Allocate the matrices. */
    ah = malloc( MAXDIM * MAXDIM * sizeof(half) );
    bh = malloc( MAXDIM * MAXDIM * sizeof(half) );
    ch = malloc( MAXDIM * MAXDIM * sizeof(half) );

    /* Step through the different array sizes. */
    for ( n = 0; n < MAXDIM; n++ ) {
        /* Initialize the needed arrays at this size. */
        for ( i = 0; i < n; i++ ) {
            for ( j = 0; j < n; j++ ) {
                ch[i * n + j] = 0.0;
                ah[i * n + j] = ((half)random())/((half)RAND_MAX) * (half)1.1;
                bh[i * n + j] = ((half)random())/((half)RAND_MAX) * (half)1.1;
            }
        }

        /* Reset PAPI count. */
        if ( (retval = PAPI_start( EventSet )) != PAPI_OK ) {
            return;
        }

        /* Run the kernel. */
        gemm_half( n, ch, ah, bh );
        usleep(1);

        /* Stop and print count. */
        resultline( n, GEMM, EventSet, fp );

        keep_half_mat_res( n, ch );
    }

    free( ah );
    free( bh );
    free( ch );
}

void keep_half_vec_res( int n, half *xh ) {

    int i;
    half sum = 0.0;
    for( i = 0; i < n; ++i ) {
        sum += xh[i];
    }
    
    if( 1.2345 == sum ) {
        fprintf(stderr, "Side-effect to disable dead code elimination by the compiler. Please ignore.\n");
    }
}

void keep_half_mat_res( int n, half *lh ) {

    int i, j;
    half sum = 0.0;
    for( i = 0; i < n; ++i ) {
        for( j = 0; j < n; ++j ) {
            sum += lh[i * n + j];
        }
    }
    
    if( 1.2345 == sum ) {
        fprintf(stderr, "Side-effect to disable dead code elimination by the compiler. Please ignore.\n");
    }
}
#endif

void exec_flops( int precision, int EventSet, FILE *fp ) {

    /* Vector Normalization and Cholesky Decomposition tests. */
    switch(precision) {
      case DOUBLE:
          exec_double_norm(EventSet, fp);
          exec_double_cholesky(EventSet, fp);
          exec_double_gemm(EventSet, fp);
          break;
      case SINGLE:
          exec_single_norm(EventSet, fp);
          exec_single_cholesky(EventSet, fp);
          exec_single_gemm(EventSet, fp);
          break;
      case HALF:
#if defined(ARM)
          exec_half_norm(EventSet, fp);
          exec_half_cholesky(EventSet, fp);
          exec_half_gemm(EventSet, fp);
#endif
          break;
      default:
          ;
    }

    return;
}

void flops_driver( char* papi_event_name, hw_desc_t *hw_desc, char* outdir ) {
    int retval = PAPI_OK;
    int EventSet = PAPI_NULL;
    FILE* ofp_papi;
    const char *sufx = ".flops";
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

    exec_flops(HALF,   EventSet, ofp_papi);
    exec_flops(SINGLE, EventSet, ofp_papi);
    exec_flops(DOUBLE, EventSet, ofp_papi);

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
