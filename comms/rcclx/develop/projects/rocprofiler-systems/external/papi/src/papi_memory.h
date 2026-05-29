#ifndef _PAPI_MALLOC
#define _PAPI_MALLOC

#include <stdlib.h>

#define DEBUG_FILE_LEN  20

typedef struct pmem
{
	void *ptr;
	int size;
#ifdef DEBUG
	char file[DEBUG_FILE_LEN];
	int line;
#endif
	struct pmem *next;
	struct pmem *prev;
} pmem_t;

#ifndef IN_MEM_FILE
#ifdef PAPI_NO_MEMORY_MANAGEMENT
#define papi_malloc(a) malloc(a)
#define papi_free(a)   free(a)
#define papi_realloc(a,b) realloc(a,b)
#define papi_calloc(a,b) calloc(a,b)
#define papi_valid_free(a) 1
#define papi_strdup(a) strdup(a)
#define papi_mem_cleanup_all() ;
#define papi_mem_print_info(a) ;
#define papi_mem_print_stats() ;
#define papi_mem_overhead(a) ;
#define papi_mem_check_all_overflow() ;
#else
#define papi_malloc(a) _papi_malloc(__FILE__,__LINE__, a)
#define papi_free(a) _papi_free(__FILE__,__LINE__, a)
#define papi_realloc(a,b) _papi_realloc(__FILE__,__LINE__,a,b)
#define papi_calloc(a,b) _papi_calloc(__FILE__,__LINE__,a,b)
#define papi_valid_free(a) _papi_valid_free(__FILE__,__LINE__,a)
#define papi_strdup(a) _papi_strdup(__FILE__,__LINE__,a)
#define papi_mem_cleanup_all _papi_mem_cleanup_all
#define papi_mem_print_info(a) _papi_mem_print_info(a)
#define papi_mem_print_stats _papi_mem_print_stats
#define papi_mem_overhead(a) _papi_mem_overhead(a)
#define papi_mem_check_all_overflow _papi_mem_check_all_overflow
#endif
#endif

void *_papi_malloc( char *, int, size_t );
void _papi_free( char *, int, void * );
void *_papi_realloc( char *, int, void *, size_t );
void *_papi_calloc( char *, int, size_t, size_t );
int _papi_valid_free( char *, int, void * );
char *_papi_strdup( char *, int, const char *s );
void _papi_mem_cleanup_all(  );
void _papi_mem_print_info( void *ptr );
void _papi_mem_print_stats(  );
int _papi_mem_overhead( int );
int _papi_mem_check_all_overflow(  );

#define PAPI_MEM_LIB_OVERHEAD	1	/* PAPI Library Overhead */
#define PAPI_MEM_OVERHEAD	2	/* Memory Overhead */
#endif
