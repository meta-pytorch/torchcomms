#ifndef __X86_UTIL_H__
#define __X86_UTIL_H__

#include "cpu_utils.h"

int x86_cpu_init( void );
int x86_cpu_finalize( void );
int x86_cpu_get_vendor( char *vendor );
int x86_cpu_get_name( char *name );
int x86_cpu_get_attribute( CPU_attr_e attr, int  *value );
int x86_cpu_get_attribute_at( CPU_attr_e attr, int loc, int *value );

#endif /* End of __X86_UTIL_H__ */
