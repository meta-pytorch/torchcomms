#define NUM_WORK_SECONDS 2
#define NUM_FLOPS  20000000
#define NUM_MISSES 2000000
#define NUM_READS  20000
#define SUCCESS 1
#define FAILURE 0
#define MAX_THREADS 256
#define NUM_THREADS 4
#define NUM_ITERS  1000000
#define THRESHOLD   1000000
#define L1_MISS_BUFFER_SIZE_INTS 128*1024
#define CACHE_FLUSH_BUFFER_SIZE_INTS 16*1024*1024
#define TOLERANCE   .2
#define OVR_TOLERANCE .75
#define MPX_TOLERANCE .20
#define TIME_LIMIT_IN_US 60*1000000    /* Run for about 1 minute or 60000000 us */

void do_reads( int n );
void fdo_reads( int *n );
void fdo_reads_( int *n );
void fdo_reads__( int *n );
void FDO_READS( int *n );
void _FDO_READS( int *n );
void do_flops( int n );
/* export the next symbol as 'end' address of do_flops for profiling */
void fdo_flops( int *n );
void fdo_flops_( int *n );
void fdo_flops__( int *n );
void FDO_FLOPS( int *n );
void _FDO_FLOPS( int *n );
void do_misses( int n, int bytes );
void fdo_misses( int *n, int *size );
void fdo_misses_( int *n, int *size );
void fdo_misses__( int *n, int *size );
void FDO_MISSES( int *n, int *size );
void _FDO_MISSES( int *n, int *size );
void do_flush( void );
void fdo_flush( void );
void fdo_flush_( void );
void fdo_flush__( void );
void FDO_FLUSH( void );
void _FDO_FLUSH( void );
void do_l1misses( int n );
void fdo_l1misses( int *n );
void fdo_l1misses_( int *n );
void fdo_l1misses__( int *n );
void FDO_L1MISSES( int *n );
void _FDO_L1MISSES( int *n );
void do_stuff( void );
void do_stuff_( void );
void do_stuff__( void );
void DO_STUFF( void );
void _DO_STUFF( void );

void dummy( void *array );
void dummy_( void *array );
void dummy__( void *array );
void DUMMY( void *array );
void _DUMMY( void *array );
void touch_dummy( double *array, int size );



