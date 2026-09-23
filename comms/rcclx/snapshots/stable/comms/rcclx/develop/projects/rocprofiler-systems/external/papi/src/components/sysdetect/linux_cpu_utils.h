#ifndef __LINUX_CPU_UTIL_H__
#define __LINUX_CPU_UTIL_H__

#include "cpu_utils.h"

int linux_cpu_get_vendor( char *vendor );
int linux_cpu_get_name( char *name );
int linux_cpu_get_attribute( CPU_attr_e attr, int *value );
int linux_cpu_get_attribute_at( CPU_attr_e attr, int loc, int *value );
int linux_cpu_set_affinity( int cpu );
int linux_cpu_get_num_supported( void );
int linux_cpu_store_affinity( void );
int linux_cpu_load_affinity( void );

#endif /* End of __LINUX_CPU_UTIL_H__ */
