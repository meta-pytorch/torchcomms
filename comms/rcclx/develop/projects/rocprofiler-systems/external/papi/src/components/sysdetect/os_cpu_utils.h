#ifndef __OS_CPU_UTILS_H__
#define __OS_CPU_UTILS_H__

#include "cpu_utils.h"

int os_cpu_get_vendor( char *vendor );
int os_cpu_get_name( char *name );
int os_cpu_get_attribute( CPU_attr_e attr, int *value );
int os_cpu_get_attribute_at( CPU_attr_e attr, int loc, int *value );
int os_cpu_set_affinity( int cpu );
int os_cpu_get_num_supported( void );
int os_cpu_store_affinity( void );
int os_cpu_load_affinity( void );

#endif /* End of __OS_CPU_UTILS_H__ */
