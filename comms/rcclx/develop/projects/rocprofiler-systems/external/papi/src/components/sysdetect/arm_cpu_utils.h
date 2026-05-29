#ifndef __ARM_UTIL_H__
#define __ARM_UTIL_H__

#include "cpu_utils.h"

int arm_cpu_init( void );
int arm_cpu_finalize( void );
int arm_cpu_get_vendor( char *vendor );
int arm_cpu_get_name( char *name );
int arm_cpu_get_attribute( CPU_attr_e attr, int *value );
int arm_cpu_get_attribute_at( CPU_attr_e attr, int loc, int *value );

#endif /* End of __ARM_UTIL_H__ */
