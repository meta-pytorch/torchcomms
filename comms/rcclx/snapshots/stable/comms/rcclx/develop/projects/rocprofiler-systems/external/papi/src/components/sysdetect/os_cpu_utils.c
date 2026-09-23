#include "sysdetect.h"
#include "os_cpu_utils.h"
#include "linux_cpu_utils.h"

int
os_cpu_get_vendor( char *vendor )
{
#if defined(__linux__)
    return linux_cpu_get_vendor(vendor);
#elif defined(__APPLE__) || defined(__MACH__)
    #warning "WARNING! Darwin support of " __func__ " not yet implemented."
    return CPU_ERROR;
#endif
    return CPU_ERROR;
}

int
os_cpu_get_name( char *name )
{
#if defined(__linux__)
    return linux_cpu_get_name(name);
#elif defined(__APPLE__) || defined(__MACH__)
    #warning "WARNING! Darwin support of " __func__ " not yet implemented."
    return CPU_ERROR;
#endif
    return CPU_ERROR;
}

int
os_cpu_get_attribute( CPU_attr_e attr, int *value )
{
#if defined(__linux__)
    return linux_cpu_get_attribute(attr, value);
#elif defined(__APPLE__) || defined(__MACH__)
    #warning "WARNING! Darwin support of " __func__ " not yet implemented."
    return CPU_ERROR;
#endif
    return CPU_ERROR;
}

int
os_cpu_get_attribute_at( CPU_attr_e attr, int loc, int *value )
{
#if defined(__linux__)
    return linux_cpu_get_attribute_at(attr, loc, value);
#elif defined(__APPLE__) || defined(__MACH__)
    #warning "WARNING! Darwin support of " __func__ " not yet implemented."
    return CPU_ERROR;
#endif
    return CPU_ERROR;
}

int
os_cpu_set_affinity( int cpu )
{
#if defined(__linux__)
    return linux_cpu_set_affinity(cpu);
#elif defined(__APPLE__) || defined(__MACH__)
    #warning "WARNING! Darwin support of " __func__ " not yet implemented."
    return CPU_ERROR;
#endif
    return CPU_ERROR;
}

int
os_cpu_get_num_supported( void )
{
#if defined(__linux__)
    return linux_cpu_get_num_supported();
#elif defined(__APPLE__) || defined(__MACH__)
    #warning "WARNING! Darwin support of " __func__ " not yet implemented."
    return CPU_ERROR;
#endif
    return CPU_ERROR;
}

int
os_cpu_store_affinity( void )
{
#if defined(__linux__)
    return linux_cpu_store_affinity();
#elif defined(__APPLE__) || defined(__MACH__)
    #warning "WARNING! Darwin support of " __func__ " not yet implemented."
    return CPU_ERROR;
#endif
    return CPU_ERROR;
}

int
os_cpu_load_affinity( void )
{
#if defined(__linux__)
    return linux_cpu_load_affinity();
#elif defined(__APPLE__) || defined(__MACH__)
    #warning "WARNING! Darwin support of " __func__ " not yet implemented."
    return CPU_ERROR;
#endif
    return CPU_ERROR;
}
