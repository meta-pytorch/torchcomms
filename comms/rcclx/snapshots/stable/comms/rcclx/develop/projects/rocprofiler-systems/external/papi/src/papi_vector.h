/**
  * @file papi_vector.h
  */
#ifndef _PAPI_VECTOR_H
#define _PAPI_VECTOR_H

/** Sizes of structure private to each component 
  */
typedef struct cmp_struct_sizes {
    int		context;
    int		control_state;
    int		reg_value;
    int		reg_alloc;
} cmp_struct_sizes_t;

/** Vector Table Stuff 
 *	@internal */
typedef struct papi_vectors {
/** Component specific data structure @see papi.h */
    PAPI_component_info_t   cmp_info;

/** Component specific structure sizes*/
    cmp_struct_sizes_t size;

/* List of exposed function pointers for this component */
	void ( *dispatch_timer ) ( int, hwd_siginfo_t *, void * );
    void *	(*get_overflow_address)	(int, char *, int);						/**< */
    int		(*start)		(hwd_context_t *, hwd_control_state_t *);		/**< */
    int		(*stop)			(hwd_context_t *, hwd_control_state_t *);		/**< */
    int		(*read)			(hwd_context_t *, hwd_control_state_t *, long long **, int);	/**< */
    int		(*reset)		(hwd_context_t *, hwd_control_state_t *);		/**< */
    int		(*write)		(hwd_context_t *, hwd_control_state_t *, long long[]);			/**< */
	int			(*cleanup_eventset)	( hwd_control_state_t * );				/**< */
    int		(*stop_profiling)	(ThreadInfo_t *, EventSetInfo_t *);			/**< */
    int		(*init_component)	(int);										/**< */
    int		(*init_thread)		 (hwd_context_t *);								/**< */
    int		(*init_control_state)	(hwd_control_state_t * ptr);			/**< */
    int		(*update_control_state)	(hwd_control_state_t *, NativeInfo_t *, int, hwd_context_t *);	/**< */
    int		(*ctl)			(hwd_context_t *, int , _papi_int_option_t *);	/**< */
    int		(*set_overflow)		(EventSetInfo_t *, int, int);				/**< */
    int		(*set_profile)		(EventSetInfo_t *, int, int);				/**< */
    int		(*set_domain)		(hwd_control_state_t *, int);				/**< */
    int		(*ntv_enum_events)	(unsigned int *, int);						/**< */
    int		(*ntv_name_to_code)	(const char *, unsigned int *);					/**< */
    int		(*ntv_code_to_name)	(unsigned int, char *, int);				/**< */
    int		(*ntv_code_to_descr)	(unsigned int, char *, int);			/**< */
    int		(*ntv_code_to_bits)	(unsigned int, hwd_register_t *);			/**< */
    int         (*ntv_code_to_info)     (unsigned int, PAPI_event_info_t *);
    int		(*allocate_registers)	(EventSetInfo_t *);
		/**< called when an event is added.  Should make
                     sure the new EventSet can map to hardware and
                     any conflicts are addressed */
    int		(*shutdown_thread)	(hwd_context_t *);								/**< */
    int		(*shutdown_component)	(void);									/**< */
    int		(*user)			(int, void *, void *);							/**< */
}papi_vector_t;

extern papi_vector_t *_papi_hwd[];

typedef struct papi_os_vectors {
  long long   (*get_real_cycles)      (void);                   /**< */
  long long   (*get_virt_cycles)      (void);                   /**< */
  long long   (*get_real_usec)        (void);                   /**< */
  long long   (*get_virt_usec)        (void);                   /**< */
  long long   (*get_real_nsec)        (void);                   /**< */
  long long   (*get_virt_nsec)        (void);                   /**< */
  int         (*update_shlib_info)    (papi_mdi_t * mdi);       /**< */
  int         (*get_system_info)      (papi_mdi_t * mdi);       /**< */
  int         (*get_memory_info)      (PAPI_hw_info_t *, int);  /**< */
  int         (*get_dmem_info)        (PAPI_dmem_info_t *);     /**< */
} papi_os_vector_t;

extern papi_os_vector_t _papi_os_vector;


/* Prototypes */
int _papi_hwi_innoculate_vector( papi_vector_t * v );
int _papi_hwi_innoculate_os_vector( papi_os_vector_t * v );

#endif /* _PAPI_VECTOR_H */
