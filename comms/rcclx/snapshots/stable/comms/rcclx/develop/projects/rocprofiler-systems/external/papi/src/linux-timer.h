long long _linux_get_real_cycles( void );

long long _linux_get_virt_usec_pttimer( void );
long long _linux_get_virt_usec_gettime( void );
long long _linux_get_virt_usec_times( void );
long long _linux_get_virt_usec_rusages( void );

long long _linux_get_real_usec_gettime( void );
long long _linux_get_real_usec_gettimeofday( void );
long long _linux_get_real_usec_cycles( void );

long long _linux_get_real_nsec_gettime( void );
long long _linux_get_virt_nsec_gettime( void );

int mmtimer_setup(void);
int init_proc_thread_timer( hwd_context_t *thr_ctx );
