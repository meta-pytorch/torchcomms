#define L1I_CACHE 0
#define L1D_CACHE 1
#define L2_CACHE  2
#define L3_CACHE  3
#define MAX_CACHE (L3_CACHE+1)

long long get_cachesize(int type);
long long get_entries(int type);
long long get_linesize(int type);
