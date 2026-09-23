#include "eventstock.h"
#include "dcache.h"
#include "branch.h"
#include "icache.h"
#include "flops.h"
#include "vec.h"
#include "instr.h"
#include "hw_desc.h"
#include "params.h"

#define USE_ALL_EVENTS 0x0
#define READ_FROM_FILE 0x1

#define BENCH_FLOPS        0x01
#define BENCH_BRANCH       0x02
#define BENCH_DCACHE_READ  0x04
#define BENCH_DCACHE_WRITE 0x08
#define BENCH_ICACHE_READ  0x10
#define BENCH_VEC          0x20
#define BENCH_INSTR        0x40

int parseArgs(int argc, char **argv, cat_params_t *params);
int setup_evts(char* inputfile, char*** basenames, int** cards);
unsigned long int omp_get_thread_num_wrapper();
int check_cards(cat_params_t mode, int** indexmemo, char** basenames, int* cards, int ct, int nevts, evstock* data);
void combine_qualifiers(int n, int pk, int ct, char** list, char* name, char** allevts, int* track, int flag, int* bitmap);
void trav_evts(evstock* stock, int pk, int* cards, int nevts, int selexnsize, int mode, char** allevts, int* track, int* indexmemo, char** basenames);
int perm(int n, int k);
int comb(int n, int k);
void testbench(char** allevts, int cmbtotal, hw_desc_t *hw_desc, cat_params_t params, int myid, int nprocs);
void print_usage();
static int parse_line(FILE *input, char **key, long long *value);
static void read_conf_file(char *conf_file, hw_desc_t *hw_desc);
static hw_desc_t *obtain_hardware_description(char *conf_file_name);
