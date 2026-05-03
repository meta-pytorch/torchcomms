#define ALL_OK              0
#define CODE_UNIMPLEMENTED -1
#define ERROR_RESULT       -2

/* instructions_testcode.c */
int instructions_million(void);
int instructions_fldcw(void);
int instructions_rep(void);

/* branches_testcode.c */
int branches_testcode(void);
int random_branches_testcode(int number, int quiet);

/* flops_testcode.c */
int flops_float_init_matrix(void);
float flops_float_matrix_matrix_multiply(void);
float flops_float_swapped_matrix_matrix_multiply(void);
int flops_double_init_matrix(void);
double flops_double_matrix_matrix_multiply(void);
double flops_double_swapped_matrix_matrix_multiply(void);
double do_flops3( double x, int iters, int quiet );
double do_flops( int n, int quiet );

/* cache_testcode.c */
int cache_write_test(double *array, int size);
double cache_read_test(double *array, int size);
int cache_random_write_test(double *array, int size, int count);
double cache_random_read_test(double *array, int size, int count);

/* busy_work.c */
double do_cycles( int minimum_time );
