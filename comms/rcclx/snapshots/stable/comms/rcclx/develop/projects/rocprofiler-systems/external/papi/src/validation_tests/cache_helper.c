#include <stdio.h>

#include "cache_helper.h"

#include "papi.h"
#include "papi_test.h"

const PAPI_hw_info_t *hw_info=NULL;

struct cache_info_t {
	int wpolicy;
	int replace;
	int size;
	int entries;
	int ways;
	int linesize;
};

static struct cache_info_t cache_info[MAX_CACHE];

static int check_if_cache_info_available(void) {

	int cache_type,level,j;

	/* Get PAPI Hardware Info */
	hw_info=PAPI_get_hardware_info();
	if (hw_info==NULL) {
		return -1;
	}

	/* Iterate down the levels (L1, L2, L3) */
	for(level=0;level<hw_info->mem_hierarchy.levels;level++) {
		for(j=0;j<2;j++) {
			cache_type=PAPI_MH_CACHE_TYPE(
				hw_info->mem_hierarchy.level[level].cache[j].type);
			if (cache_type==PAPI_MH_TYPE_EMPTY) continue;

			if (level==0) {
				if (cache_type==PAPI_MH_TYPE_DATA) {
					cache_info[L1D_CACHE].size=hw_info->mem_hierarchy.level[level].cache[j].size;
					cache_info[L1D_CACHE].linesize=hw_info->mem_hierarchy.level[level].cache[j].line_size;
					cache_info[L1D_CACHE].ways=hw_info->mem_hierarchy.level[level].cache[j].associativity;
					cache_info[L1D_CACHE].entries=cache_info[L1D_CACHE].size/cache_info[L1D_CACHE].linesize;
					cache_info[L1D_CACHE].wpolicy=PAPI_MH_CACHE_WRITE_POLICY(hw_info->mem_hierarchy.level[level].cache[j].type);
					cache_info[L1D_CACHE].replace=PAPI_MH_CACHE_REPLACEMENT_POLICY(hw_info->mem_hierarchy.level[level].cache[j].type);
				}
				else if (cache_type==PAPI_MH_TYPE_INST) {
					cache_info[L1I_CACHE].size=hw_info->mem_hierarchy.level[level].cache[j].size;
					cache_info[L1I_CACHE].linesize=hw_info->mem_hierarchy.level[level].cache[j].line_size;
					cache_info[L1I_CACHE].ways=hw_info->mem_hierarchy.level[level].cache[j].associativity;
					cache_info[L1I_CACHE].entries=cache_info[L1I_CACHE].size/cache_info[L1I_CACHE].linesize;
					cache_info[L1I_CACHE].wpolicy=PAPI_MH_CACHE_WRITE_POLICY(hw_info->mem_hierarchy.level[level].cache[j].type);
					cache_info[L1I_CACHE].replace=PAPI_MH_CACHE_REPLACEMENT_POLICY(hw_info->mem_hierarchy.level[level].cache[j].type);
				}
			}
			else if (level==1) {
				cache_info[L2_CACHE].size=hw_info->mem_hierarchy.level[level].cache[j].size;
				cache_info[L2_CACHE].linesize=hw_info->mem_hierarchy.level[level].cache[j].line_size;
				cache_info[L2_CACHE].ways=hw_info->mem_hierarchy.level[level].cache[j].associativity;
				cache_info[L2_CACHE].entries=cache_info[L2_CACHE].size/cache_info[L2_CACHE].linesize;
				cache_info[L2_CACHE].wpolicy=PAPI_MH_CACHE_WRITE_POLICY(hw_info->mem_hierarchy.level[level].cache[j].type);
				cache_info[L2_CACHE].replace=PAPI_MH_CACHE_REPLACEMENT_POLICY(hw_info->mem_hierarchy.level[level].cache[j].type);
			}
			else if (level==2) {
				cache_info[L3_CACHE].size=hw_info->mem_hierarchy.level[level].cache[j].size;
				cache_info[L3_CACHE].linesize=hw_info->mem_hierarchy.level[level].cache[j].line_size;
				cache_info[L3_CACHE].ways=hw_info->mem_hierarchy.level[level].cache[j].associativity;
				cache_info[L3_CACHE].entries=cache_info[L3_CACHE].size/cache_info[L3_CACHE].linesize;
				cache_info[L3_CACHE].wpolicy=PAPI_MH_CACHE_WRITE_POLICY(hw_info->mem_hierarchy.level[level].cache[j].type);
				cache_info[L3_CACHE].replace=PAPI_MH_CACHE_REPLACEMENT_POLICY(hw_info->mem_hierarchy.level[level].cache[j].type);
			}

		}
	}
	return 0;
}

long long get_cachesize(int type) {

	int result;

	result=check_if_cache_info_available();
	if (result<0) return result;

	if (type>=MAX_CACHE) {
		printf("Errror!\n");
		return -1;
	}

	return cache_info[type].size;
}


long long get_entries(int type) {

	int result;

	result=check_if_cache_info_available();
	if (result<0) return result;

	if (type>=MAX_CACHE) {
		printf("Errror!\n");
		return -1;
	}

	return cache_info[type].entries;
}


long long get_linesize(int type) {

	int result;

	result=check_if_cache_info_available();
	if (result<0) return result;

	if (type>=MAX_CACHE) {
		printf("Errror!\n");
		return -1;
	}

	return cache_info[type].linesize;
}
