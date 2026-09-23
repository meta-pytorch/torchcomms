/*
* File:    pe_libpfm4_events.h
*/

/* Prototypes for libpfm name library access */

int _pe_libpfm4_setup_presets( char *name, int type, int cidx );
int _pe_libpfm4_ntv_enum_events( unsigned int *EventCode, int modifier, int cidx,
		       struct native_event_table_t *event_table);
int _pe_libpfm4_ntv_name_to_code( const char *ntv_name,
				    unsigned int *EventCode, int cidx,
		       struct native_event_table_t *event_table);
int _pe_libpfm4_ntv_code_to_name( unsigned int EventCode, char *name,
				    int len,
		       struct native_event_table_t *event_table);
int _pe_libpfm4_ntv_code_to_descr( unsigned int EventCode, char *name,
				     int len,
		       struct native_event_table_t *event_table);
int _pe_libpfm4_shutdown(papi_vector_t *my_vector,
		       struct native_event_table_t *event_table);

int _pe_libpfm4_ntv_code_to_info(unsigned int EventCode,
				   PAPI_event_info_t *info,
		       struct native_event_table_t *event_table);

int _pe_libpfm4_init(papi_vector_t *my_vector, int cidx,
		       struct native_event_table_t *event_table,
		       int pmu_type);

int _peu_libpfm4_init(papi_vector_t *my_vector, int cidx,
		       struct native_event_table_t *event_table,
		       int pmu_type);
