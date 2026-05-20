#ifndef __PAPI_COST_UTILS_H__
#define __PAPI_COST_UTILS_H__
extern int num_iters;

double	do_stats(long long*, long long*, long long *, double *);
void	do_std_dev( long long*, int*, double, double );
void	do_dist( long long*, long long, long long, int, int*);
int do_percentile(long long *a, long long *percent25, long long *percent50,
		long long *percent75, long long *percent99);

#endif /* __PAPI_COST_UTILS_H__ */
