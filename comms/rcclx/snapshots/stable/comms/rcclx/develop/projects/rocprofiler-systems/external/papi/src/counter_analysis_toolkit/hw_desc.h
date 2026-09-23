#ifndef _HW_DESC_
#define _HW_DESC_

#define _MAX_SUPPORTED_CACHE_LEVELS 16

typedef struct _hw_desc{
  int numcpus;
  int cache_levels;
  int maxPPB;
  int mmsplit;
  int pts_per_mm;
  int split[_MAX_SUPPORTED_CACHE_LEVELS];
  int pts_per_reg[_MAX_SUPPORTED_CACHE_LEVELS];
  long long dcache_line_size[_MAX_SUPPORTED_CACHE_LEVELS];
  long long dcache_size[_MAX_SUPPORTED_CACHE_LEVELS];
  int dcache_assoc[_MAX_SUPPORTED_CACHE_LEVELS];
  long long icache_line_size[_MAX_SUPPORTED_CACHE_LEVELS];
  long long icache_size[_MAX_SUPPORTED_CACHE_LEVELS];
  int icache_assoc[_MAX_SUPPORTED_CACHE_LEVELS];
} hw_desc_t;

#endif
