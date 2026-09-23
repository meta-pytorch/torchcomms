#ifndef __POWERPC_UTIL_H__
#define __POWERPC_UTIL_H__

#include "cpu_utils.h"

int powerpc_cpu_init( void );
int powerpc_cpu_finalize( void );
int powerpc_cpu_get_vendor( char *vendor );
int powerpc_cpu_get_name( char *name );
int powerpc_cpu_get_attribute( CPU_attr_e attr, int *value );
int powerpc_cpu_get_attribute_at( CPU_attr_e attr, int loc, int *value );

#endif /* End of __POWERPC_UTIL_H__ */
