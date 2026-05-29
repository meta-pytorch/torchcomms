/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

/** 
* @file    papi.h
*
* @author  Philip Mucci
*          mucci@cs.utk.edu
* @author  dan terpstra
*          terpstra@cs.utk.edu
* @author  Haihang You
*	       you@cs.utk.edu
* @author  Kevin London
*	       london@cs.utk.edu
* @author  Maynard Johnson
*          maynardj@us.ibm.com
*
* @brief Return codes and api definitions.
*/

#ifndef _PAPI
#define _PAPI

#pragma GCC visibility push(default)

/**
 * @mainpage PAPI
 *  
 * @section papi_intro Introduction
 * The PAPI Performance Application Programming Interface provides machine and 
 * operating system independent access to hardware performance counters found 
 * on most modern processors. 
 * Any of over 100 preset events can be counted through either a simple high 
 * level programming interface or a more complete low level interface from 
 * either C or Fortran. 
 * A list of the function calls in these interfaces is given below, 
 * with references to other pages for more complete details. 
 *
 * @section papi_high_api High Level Functions
 * A simple interface for instrumenting end-user applications. 
 * Fully supported on both C and Fortran.
 * See individual functions for details on usage.
 * 
 *	@ref high_api
 * 
 * Note that the high-level interface is self-initializing. 
 * You can mix high and low level calls, but you @b must call either 
 * @ref PAPI_library_init() or a high level routine before calling a low level routine.
 *
 * @section papi_low_api Low Level Functions
 * Advanced interface for all applications and performance tools.
 * Some functions may be implemented only for C or Fortran.
 * See individual functions for details on usage and support.
 * 
 * @ref low_api
 *
 * @section papi_Fortran Fortran API
 * The Fortran interface has some unique features and entry points.
 * See individual functions for details.
 * 
 * @ref PAPIF
 *
 * @ref PAPIF-HL
 *
 * @section Components 
 *
 *	Components provide access to hardware information on specific subsystems.
 *
 *	Components can be found under the components directory or @ref papi_components "here"
 *	and included in a build as an argument to configure,\n
 *	'--with-components=< comma_seperated_list_of_components_to_build >'.
 * 
 * @section papi_util PAPI Utility Commands
 * <ul> 
 *		<li> @ref papi_avail - provides availability and detail information for PAPI preset events
 *		<li> @ref papi_clockres - provides availability and detail information for PAPI preset events
 *		<li> @ref papi_cost - provides availability and detail information for PAPI preset events
 *		<li> @ref papi_command_line - executes PAPI preset or native events from the command line
 *		<li> @ref papi_decode -	decodes PAPI preset events into a csv format suitable for 
 *							PAPI_encode_events
 *		<li> @ref papi_event_chooser -	given a list of named events, lists other events 
 *										that can be counted with them
 *		<li> @ref papi_mem_info -	provides information on the memory architecture 
									of the current processor
 *		<li> @ref papi_native_avail - provides detailed information for PAPI native events 
 * </ul>
 * @see The PAPI Website http://icl.cs.utk.edu/papi
 */

/** \htmlonly
  * @page CDI PAPI Component Development Interface
  * @par \em Introduction
  *		PAPI-C consists of a Framework and between 1 and 16 Components. 
  *		The Framework is platform independent and exposes the PAPI API to end users. 
  *		The Components provide access to hardware information on specific subsystems. 
  *		By convention, Component 0 is always a CPU Component. 
  *		This allows default behavior for legacy code, and provides a universal 
  *		place to define system-wide operations and parameters, 
  *		like clock rates and interrupt structures. 
  *		Currently only a single CPU Component can exist at a time. 
  *
  * @par No CPU
  *		In certain cases it can be desirable to use a generic CPU component for 
  *		testing instrumentation or for operation on systems that don't provide 
  *		the proper patches for accessing cpu counters. 
  *		For such a case, the configure option: 
  *	@code
  *		configure --with-no-cpu-counters = yes
  *	@endcode 
  *	is provided to build PAPI with an "empty" cpu component.
  *
  *	@par Exposed Interface
  *		A Component for PAPI-C typically consists of a single header file and a 
  *		single (or small number of) source file(s). 
  *		All of the information for a Component needed by PAPI-C is exposed through 
  *		a single data structure that is declared and initialized at the bottom 
  *		of the main source file. 
  *		This structure, @ref papi_vector_t , is defined in @ref papi_vector.h .
  *	
  *	@par Compiling With an Existing Component 
  *		Components provided with the PAPI source distribution all appear in the 
  *		src/components directory. 
  *		Each component exists in its own directory, named the same as the component itself. 
  *		To include a component in a PAPI build, use the configure command line as shown:
  *	
  *	@code
  *		configure --with-components="component list"
  *	@endcode
  *	
  * Replace the "component list" argument with either the name of a specific 
  *	component directory or multiple component names separated by spaces and 
  *	enclosed in quotes as shown below:
  *
  *	\c configure --with-components="acpi lustre infiniband"
  *
  *	In some cases components themselves require additional configuration. 
  *	In these cases an error message will be produced when you run @code make @endcode . 
  *	To fix this, run the configure script found in the component directory.
  * 
  *	@par Adding a New Component 
  *	The mechanics of adding a new component to the PAPI 4.1 build are relatively straight-forward.
  *	Add a directory to the papi/src/components directory that is named with 
  *	the base name of the component. 
  *	This directory will contain the source files and build files for the new component. 
  *	If configuration of the component is necessary, 
  *	additional configure and make files will be needed. 
  *	The /example directory can be cloned and renamed as a starting point. 
  *	Other components can be used as examples. 
  *	This is described in more detail in /components/README.
  *
  *	@par Developing a New Component 
  *		A PAPI-C component generally consists of a header file and one or a 
  *		small number of source files. 
  *		The source file must contain a @ref papi_vector_t structure that 
  *		exposes the internal data and entry points of the component to the PAPI-C Framework. 
  *		This structure must have a unique name that is exposed externally and 
  *		contains the name of the directory containing the component source code.
  *
  *	Three types of information are exposed in the @ref papi_vector_t structure:
  *		Configuration parameters are contained in the @ref PAPI_component_info_t structure;
  *		Sizes of opaque data structures necessary for memory management are in the @ref cmp_struct_sizes_t structure;
  *		An array of function entry points which, if implemented, provide access to the functionality of the component.
  *
  *	If a function is not implemented in a given component its value in the structure can be left unset. 
  *	In this case it will be initialized to NULL, and result (generally) in benign, although unproductive, behavior.
  *
  *	During the development of a component, functions can be implemented and tested in blocks. 
  *	Further information about an appropriate order for developing these functions 
  *	can be found in the Component Development Cookbook .
  *
  * @par PAPI-C Open Research Issues:
  *	<ul>
  *	<li> Support for non-standard data types: 
  *		Currently PAPI supports returned data values expressed as unsigned 64-bit integers. 
  *		This is appropriate for counting events, but may not be as appropriate 
  *		for expressing other values. 
  *		Examples of some other possible data types are shown below. 
  *		Data type might be expressed as a flag in the event definition.
  *	<li> Signed Integer
  *		<ul>
  *		<li>Float: 64-bit IEEE double precision
  *		<li>Fixed Point: 32-bit integer and 32-bit fraction
  *		<li>Ratios: 32 bit numerator and 32 bit denominator
  *		</ul>
  *	<li> Synchronization:
  *		Components might report values with widely different time scales and 
  *		remote measurements may be significantly skewed in time from local measurements. 
  *		It would be desirable to have a mechanism to synchronize these values in time.
  *	<li> Dynamic Component Discovery:
  *		Components currently must be included statically in the PAPI library build. 
  *		This minimizes startup disruption and time lag, particularly for large parallel systems. 
  *		In some instances it would also be desirable to support a run-time 
  *		discovery process for components, possibly by searching a specific 
  *		location for dynamic libraries.
  *	<li> Component Repository:
  *		A small collection of components are currently maintained and 
  *		supported inside the PAPI source distribution. 
  *		It would be desirable to create a public component repository where 3rd 
  *		parties could submit components for the use and benefit of the larger community.
  *	<li> Multiple CPU Components:
  *		With the rise in popularity of heterogeneous computing systems, it may 
  *		become desirable to have more than one CPU component. 
  *		Issues must then be resolved relating to which cpu time-base is used, 
  *		how are interrupts handled, etc. 
  *	</ul>
  * \endhtmlonly
  */

/* Definition of PAPI_VERSION format.  Note that each of the four 
 * components _must_ be less than 256.  Also, the PAPI_VER_CURRENT
 * masks out the revision and increment.  Any revision change is supposed 
 * to be binary compatible between the user application code and the 
 * run-time library. Any modification that breaks this compatibility 
 * _should_ modify the minor version number as to force user applications 
 * to re-compile.
 */
#define PAPI_VERSION_NUMBER(maj,min,rev,inc) (((maj)<<24) | ((min)<<16) | ((rev)<<8) | (inc))
#define PAPI_VERSION_MAJOR(x)   	(((x)>>24)    & 0xff)
#define PAPI_VERSION_MINOR(x)		(((x)>>16)    & 0xff)
#define PAPI_VERSION_REVISION(x)	(((x)>>8)     & 0xff)
#define PAPI_VERSION_INCREMENT(x)((x)          & 0xff)

/* This is the official PAPI version */
/* The final digit represents the patch count */
#define PAPI_VERSION  			PAPI_VERSION_NUMBER(7,2,0,0)
#define PAPI_VER_CURRENT 		(PAPI_VERSION & 0xffff0000)

  /* Tests for checking event code type */
#define IS_NATIVE( EventCode ) ( ( EventCode & PAPI_NATIVE_MASK ) && !(EventCode & PAPI_PRESET_MASK) )
#define IS_PRESET( EventCode ) ( ( EventCode & PAPI_PRESET_MASK ) && !(EventCode & PAPI_NATIVE_MASK) )
#define IS_USER_DEFINED( EventCode ) ( ( EventCode & PAPI_PRESET_MASK ) && (EventCode & PAPI_NATIVE_MASK) )

#ifdef __cplusplus
extern "C"
{
#endif

/* Include files */

#include <sys/types.h>
#include <limits.h>
#include "papiStdEventDefs.h"

/** \internal 
@defgroup ret_codes Return Codes
Return Codes
All of the functions contained in the PerfAPI return standardized error codes.
Values greater than or equal to zero indicate success, less than zero indicates
failure. 
@{
*/

#define PAPI_OK          0     /**< No error */
#define PAPI_EINVAL     -1     /**< Invalid argument */
#define PAPI_ENOMEM     -2     /**< Insufficient memory */
#define PAPI_ESYS       -3     /**< A System/C library call failed */
#define PAPI_ECMP       -4     /**< Not supported by component */
#define PAPI_ESBSTR     -4     /**< Backwards compatibility */
#define PAPI_ECLOST     -5     /**< Access to the counters was lost or interrupted */
#define PAPI_EBUG       -6     /**< Internal error, please send mail to the developers */
#define PAPI_ENOEVNT    -7     /**< Event does not exist */
#define PAPI_ECNFLCT    -8     /**< Event exists, but cannot be counted due to counter resource limitations */
#define PAPI_ENOTRUN    -9     /**< EventSet is currently not running */
#define PAPI_EISRUN     -10    /**< EventSet is currently counting */
#define PAPI_ENOEVST    -11    /**< No such EventSet Available */
#define PAPI_ENOTPRESET -12    /**< Event in argument is not a valid preset */
#define PAPI_ENOCNTR    -13    /**< Hardware does not support performance counters */
#define PAPI_EMISC      -14    /**< Unknown error code */
#define PAPI_EPERM      -15    /**< Permission level does not permit operation */
#define PAPI_ENOINIT    -16    /**< PAPI hasn't been initialized yet */
#define PAPI_ENOCMP     -17    /**< Component Index isn't set */
#define PAPI_ENOSUPP    -18    /**< Not supported */
#define PAPI_ENOIMPL    -19    /**< Not implemented */
#define PAPI_EBUF       -20    /**< Buffer size exceeded */
#define PAPI_EINVAL_DOM -21    /**< EventSet domain is not supported for the operation */
#define PAPI_EATTR		-22    /**< Invalid or missing event attributes */
#define PAPI_ECOUNT		-23    /**< Too many events or attributes */
#define PAPI_ECOMBO		-24    /**< Bad combination of features */
#define PAPI_ECMP_DISABLED	-25    /**< Component containing event is disabled */
#define PAPI_EDELAY_INIT -26   /**< Delayed initialization component */
#define PAPI_EMULPASS   -27    /**< Event exists, but cannot be counted due to multiple passes required by hardware */
#define PAPI_NUM_ERRORS	 28    /**< Number of error messages specified in this API */

#define PAPI_NOT_INITED		0
#define PAPI_LOW_LEVEL_INITED 	1       /* Low level has called library init */
#define PAPI_HIGH_LEVEL_INITED 	2       /* High level has called library init */
#define PAPI_THREAD_LEVEL_INITED 4      /* Threads have been inited */

/** @} */

/** @internal 
@defgroup consts Constants
All of the functions in the PerfAPI should use the following set of constants.
@{
*/

#define PAPI_NULL       -1      /**<A nonexistent hardware event used as a placeholder */

/** @internal  
	@defgroup domain_defns Domain definitions 
 	@{ */

#define PAPI_DOM_USER    0x1    /**< User context counted */
#define PAPI_DOM_MIN     PAPI_DOM_USER
#define PAPI_DOM_KERNEL	 0x2    /**< Kernel/OS context counted */
#define PAPI_DOM_OTHER	 0x4    /**< Exception/transient mode (like user TLB misses ) */
#define PAPI_DOM_SUPERVISOR 0x8 /**< Supervisor/hypervisor context counted */
#define PAPI_DOM_ALL	 (PAPI_DOM_USER|PAPI_DOM_KERNEL|PAPI_DOM_OTHER|PAPI_DOM_SUPERVISOR) /**< All contexts counted */
/* #define PAPI_DOM_DEFAULT PAPI_DOM_USER NOW DEFINED BY COMPONENT */
#define PAPI_DOM_MAX     PAPI_DOM_ALL
#define PAPI_DOM_HWSPEC  0x80000000     /**< Flag that indicates we are not reading CPU like stuff.
                                           The lower 31 bits can be decoded by the component into something
                                           meaningful. i.e. SGI HUB counters */
/** @} */

/** @internal 
 *	@defgroup thread_defns Thread Definitions 
 *		We define other levels in papi_internal.h
 *		for internal PAPI use, so if you change anything
 *		make sure to look at both places -KSL
 *	@{ */
#define PAPI_USR1_TLS		0x0
#define PAPI_USR2_TLS		0x1
#define PAPI_TLS_HIGH_LEVEL    0x2
#define PAPI_NUM_TLS		0x3
#define PAPI_TLS_USR1		PAPI_USR1_TLS
#define PAPI_TLS_USR2		PAPI_USR2_TLS
#define PAPI_TLS_NUM		PAPI_NUM_TLS
#define PAPI_TLS_ALL_THREADS	0x10
/** @} */

/** @internal 
 *	@defgroup locking_defns Locking Mechanisms defines 
 *	@{ */
#define PAPI_USR1_LOCK          	0x0    /**< User controlled locks */
#define PAPI_USR2_LOCK          	0x1    /**< User controlled locks */
#define PAPI_NUM_LOCK           	0x2    /**< Used with setting up array */
#define PAPI_LOCK_USR1          	PAPI_USR1_LOCK
#define PAPI_LOCK_USR2          	PAPI_USR2_LOCK
#define PAPI_LOCK_NUM			PAPI_NUM_LOCK
/** @} */

/* Remove this!  If it breaks userspace we might have to add it back :( */
/* #define PAPI_MPX_DEF_DEG 32			                        */

/**	@internal 
	@defgroup papi_vendors  Vendor definitions 
	@{ */
#define PAPI_VENDOR_UNKNOWN 0
#define PAPI_VENDOR_INTEL   1
#define PAPI_VENDOR_AMD     2
#define PAPI_VENDOR_IBM     3
#define PAPI_VENDOR_CRAY    4
#define PAPI_VENDOR_SUN     5
#define PAPI_VENDOR_FREESCALE 6
#define PAPI_VENDOR_ARM     7
#define PAPI_VENDOR_MIPS    8
#define PAPI_VENDOR_ARM_ARM       0x41
#define PAPI_VENDOR_ARM_BROADCOM  0x42
#define PAPI_VENDOR_ARM_CAVIUM    0x43
#define PAPI_VENDOR_ARM_FUJITSU   0x46
#define PAPI_VENDOR_ARM_HISILICON 0x48
#define PAPI_VENDOR_ARM_APM       0x50
#define PAPI_VENDOR_ARM_QUALCOMM  0x51
/** @} */

/** @internal 
 *	@defgroup granularity_defns Granularity definitions 
 *	@{ */

#define PAPI_GRN_THR     0x1    /**< PAPI counters for each individual thread */
#define PAPI_GRN_MIN     PAPI_GRN_THR
#define PAPI_GRN_PROC    0x2    /**< PAPI counters for each individual process */
#define PAPI_GRN_PROCG   0x4    /**< PAPI counters for each individual process group */
#define PAPI_GRN_SYS     0x8    /**< PAPI counters for the current CPU, are you bound? */
#define PAPI_GRN_SYS_CPU 0x10   /**< PAPI counters for all CPUs individually */
#define PAPI_GRN_MAX     PAPI_GRN_SYS_CPU
/** @} */

/** @internal 
	@defgroup evt_states States of an EventSet 
	@{ */
#define PAPI_STOPPED      0x01  /**< EventSet stopped */
#define PAPI_RUNNING      0x02  /**< EventSet running */
#define PAPI_PAUSED       0x04  /**< EventSet temp. disabled by the library */
#define PAPI_NOT_INIT     0x08  /**< EventSet defined, but not initialized */
#define PAPI_OVERFLOWING  0x10  /**< EventSet has overflowing enabled */
#define PAPI_PROFILING    0x20  /**< EventSet has profiling enabled */
#define PAPI_MULTIPLEXING 0x40  /**< EventSet has multiplexing enabled */
#define PAPI_ATTACHED	  0x80  /**< EventSet is attached to another thread/process */
#define PAPI_CPU_ATTACHED 0x100 /**< EventSet is attached to a specific cpu (not counting thread of execution) */
/** @} */

/** @internal 
	@defgroup error_predef Error predefines 
	@{ */
#define PAPI_QUIET       0      /**< Option to turn off automatic reporting of return codes < 0 to stderr. */
#define PAPI_VERB_ECONT  1      /**< Option to automatically report any return codes < 0 to stderr and continue. */
#define PAPI_VERB_ESTOP  2      /**< Option to automatically report any return codes < 0 to stderr and exit. */
/** @} */

/** @internal 
	@defgroup profile_defns Profile definitions 
	@{ */
#define PAPI_PROFIL_POSIX     0x0        /**< Default type of profiling, similar to 'man profil'. */
#define PAPI_PROFIL_RANDOM    0x1        /**< Drop a random 25% of the samples. */
#define PAPI_PROFIL_WEIGHTED  0x2        /**< Weight the samples by their value. */
#define PAPI_PROFIL_COMPRESS  0x4        /**< Ignore samples if hash buckets get big. */
#define PAPI_PROFIL_BUCKET_16 0x8        /**< Use 16 bit buckets to accumulate profile info (default) */
#define PAPI_PROFIL_BUCKET_32 0x10       /**< Use 32 bit buckets to accumulate profile info */
#define PAPI_PROFIL_BUCKET_64 0x20       /**< Use 64 bit buckets to accumulate profile info */
#define PAPI_PROFIL_FORCE_SW  0x40       /**< Force Software overflow in profiling */
#define PAPI_PROFIL_DATA_EAR  0x80       /**< Use data address register profiling */
#define PAPI_PROFIL_INST_EAR  0x100      /**< Use instruction address register profiling */
#define PAPI_PROFIL_BUCKETS   (PAPI_PROFIL_BUCKET_16 | PAPI_PROFIL_BUCKET_32 | PAPI_PROFIL_BUCKET_64)
/** @} */

/* @defgroup overflow_defns Overflow definitions 
   @{ */
#define PAPI_OVERFLOW_FORCE_SW 0x40	/**< Force using Software */
#define PAPI_OVERFLOW_HARDWARE 0x80	/**< Using Hardware */
/** @} */

/** @internal 
  *	@defgroup mpx_defns Multiplex flags definitions 
  * @{ */
#define PAPI_MULTIPLEX_DEFAULT	0x0	/**< Use whatever method is available, prefer kernel of course. */
#define PAPI_MULTIPLEX_FORCE_SW 0x1	/**< Force PAPI multiplexing instead of kernel */
/** @} */

/** @internal 
	@defgroup option_defns Option definitions 
	@{ */
#define PAPI_INHERIT_ALL  1     /**< The flag to this to inherit all children's counters */
#define PAPI_INHERIT_NONE 0     /**< The flag to this to inherit none of the children's counters */


#define PAPI_DETACH			1		/**< Detach */
#define PAPI_DEBUG          2       /**< Option to turn on  debugging features of the PAPI library */
#define PAPI_MULTIPLEX 		3       /**< Turn on/off or multiplexing for an eventset */
#define PAPI_DEFDOM  		4       /**< Domain for all new eventsets. Takes non-NULL option pointer. */
#define PAPI_DOMAIN  		5       /**< Domain for an eventset */
#define PAPI_DEFGRN  		6       /**< Granularity for all new eventsets */
#define PAPI_GRANUL  		7       /**< Granularity for an eventset */
#define PAPI_DEF_MPX_NS     8       /**< Multiplexing/overflowing interval in ns, same as PAPI_DEF_ITIMER_NS */
#define PAPI_MAX_MPX_CTRS	11      /**< Maximum number of counters we can multiplex */
#define PAPI_PROFIL  		12      /**< Option to turn on the overflow/profil reporting software [not implemented] */
#define PAPI_PRELOAD 		13      /**< Option to find out the environment variable that can preload libraries */
#define PAPI_CLOCKRATE  	14      /**< Clock rate in MHz */
#define PAPI_MAX_HWCTRS 	15      /**< Number of physical hardware counters */
#define PAPI_HWINFO  		16      /**< Hardware information */
#define PAPI_EXEINFO  		17      /**< Executable information */
#define PAPI_MAX_CPUS 		18      /**< Number of ncpus we can talk to from here */
#define PAPI_ATTACH			19      /**< Attach to a another tid/pid instead of ourself */
#define PAPI_SHLIBINFO      20      /**< Shared Library information */
#define PAPI_LIB_VERSION    21      /**< Option to find out the complete version number of the PAPI library */
#define PAPI_COMPONENTINFO  22      /**< Find out what the component supports */
/* Currently the following options are only available on Itanium; they may be supported elsewhere in the future */
#define PAPI_DATA_ADDRESS   23      /**< Option to set data address range restriction */
#define PAPI_INSTR_ADDRESS  24      /**< Option to set instruction address range restriction */
#define PAPI_DEF_ITIMER		25		/**< Option to set the type of itimer used in both software multiplexing, overflowing and profiling */
#define PAPI_DEF_ITIMER_NS	26		/**< Multiplexing/overflowing interval in ns, same as PAPI_DEF_MPX_NS */
/* Currently the following options are only available on systems using the perf_events component within papi */
#define PAPI_CPU_ATTACH		27      /**< Specify a cpu number the event set should be tied to */
#define PAPI_INHERIT		28      /**< Option to set counter inheritance flag */
#define PAPI_USER_EVENTS_FILE 29	/**< Option to set file from where to parse user defined events */

#define PAPI_INIT_SLOTS    64     /*Number of initialized slots in
                                   DynamicArray of EventSets */

#define PAPI_MIN_STR_LEN        64      /* For small strings, like names & stuff */
#define PAPI_MAX_STR_LEN       128      /* For average run-of-the-mill strings */
#define PAPI_2MAX_STR_LEN      256      /* For somewhat longer run-of-the-mill strings */
#define PAPI_HUGE_STR_LEN     1024      /* This should be defined in terms of a system parameter */

#define PAPI_PMU_MAX           80      /* maximum number of pmu's supported by one component */
#define PAPI_DERIVED           0x1      /* Flag to indicate that the event is derived */
/** @} */

/** Possible values for the 'modifier' parameter of the PAPI_enum_event call.
   A value of 0 (PAPI_ENUM_EVENTS) is always assumed to enumerate ALL 
   events on every platform.
   PAPI PRESET events are broken into related event categories.
   Each supported component can have optional values to determine how 
   native events on that component are enumerated.
*/
enum {
   PAPI_ENUM_EVENTS = 0,		/**< Always enumerate all events */
   PAPI_ENUM_FIRST,				/**< Enumerate first event (preset or native) */
   PAPI_PRESET_ENUM_AVAIL, 		/**< Enumerate events that exist here */

   /* PAPI PRESET section */
   PAPI_PRESET_ENUM_MSC,		/**< Miscellaneous preset events */
   PAPI_PRESET_ENUM_INS,		/**< Instruction related preset events */
   PAPI_PRESET_ENUM_IDL,		/**< Stalled or Idle preset events */
   PAPI_PRESET_ENUM_BR,			/**< Branch related preset events */
   PAPI_PRESET_ENUM_CND,		/**< Conditional preset events */
   PAPI_PRESET_ENUM_MEM,		/**< Memory related preset events */
   PAPI_PRESET_ENUM_CACH,		/**< Cache related preset events */
   PAPI_PRESET_ENUM_L1,			/**< L1 cache related preset events */
   PAPI_PRESET_ENUM_L2,			/**< L2 cache related preset events */
   PAPI_PRESET_ENUM_L3,			/**< L3 cache related preset events */
   PAPI_PRESET_ENUM_TLB,		/**< Translation Lookaside Buffer events */
   PAPI_PRESET_ENUM_FP,			/**< Floating Point related preset events */

   /* PAPI native event related section */
   PAPI_NTV_ENUM_UMASKS,		/**< all individual bits for given group */
   PAPI_NTV_ENUM_UMASK_COMBOS,	/**< all combinations of mask bits for given group */
   PAPI_NTV_ENUM_IARR,			/**< Enumerate events that support IAR (instruction address ranging) */
   PAPI_NTV_ENUM_DARR,			/**< Enumerate events that support DAR (data address ranging) */
   PAPI_NTV_ENUM_OPCM,			/**< Enumerate events that support OPC (opcode matching) */
   PAPI_NTV_ENUM_IEAR,			/**< Enumerate IEAR (instruction event address register) events */
   PAPI_NTV_ENUM_DEAR,			/**< Enumerate DEAR (data event address register) events */
   PAPI_NTV_ENUM_GROUPS			/**< Enumerate groups an event belongs to (e.g. POWER5) */
};

#define PAPI_ENUM_ALL PAPI_ENUM_EVENTS

#define PAPI_PRESET_BIT_MSC		(1 << PAPI_PRESET_ENUM_MSC)	/* Miscellaneous preset event bit */
#define PAPI_PRESET_BIT_INS		(1 << PAPI_PRESET_ENUM_INS)	/* Instruction related preset event bit */
#define PAPI_PRESET_BIT_IDL		(1 << PAPI_PRESET_ENUM_IDL)	/* Stalled or Idle preset event bit */
#define PAPI_PRESET_BIT_BR		(1 << PAPI_PRESET_ENUM_BR)	/* branch related preset events */
#define PAPI_PRESET_BIT_CND		(1 << PAPI_PRESET_ENUM_CND)	/* conditional preset events */
#define PAPI_PRESET_BIT_MEM		(1 << PAPI_PRESET_ENUM_MEM)	/* memory related preset events */
#define PAPI_PRESET_BIT_CACH	(1 << PAPI_PRESET_ENUM_CACH)	/* cache related preset events */
#define PAPI_PRESET_BIT_L1		(1 << PAPI_PRESET_ENUM_L1)	/* L1 cache related preset events */
#define PAPI_PRESET_BIT_L2		(1 << PAPI_PRESET_ENUM_L2)	/* L2 cache related preset events */
#define PAPI_PRESET_BIT_L3		(1 << PAPI_PRESET_ENUM_L3)	/* L3 cache related preset events */
#define PAPI_PRESET_BIT_TLB		(1 << PAPI_PRESET_ENUM_TLB)	/* Translation Lookaside Buffer events */
#define PAPI_PRESET_BIT_FP		(1 << PAPI_PRESET_ENUM_FP)	/* Floating Point related preset events */

#define PAPI_NTV_GROUP_AND_MASK		0x00FF0000	/* bits occupied by group number */
#define PAPI_NTV_GROUP_SHIFT		16			/* bit shift to encode group number */
/** @} */

/* 
The Low Level API

The following functions represent the low level portion of the
PerfAPI. These functions provide greatly increased efficiency and
functionality over the high level API presented in the next
section. All of the following functions are callable from both C and
Fortran except where noted. As mentioned in the introduction, the low
level API is only as powerful as the component upon which it is
built. Thus some features may not be available on every platform. The
converse may also be true, that more advanced features may be
available and defined in the header file.  The user is encouraged to
read the documentation carefully.  */


#include <signal.h>

/*  Earlier versions of PAPI define a special long_long type to mask
	an incompatibility between the Windows compiler and gcc-style compilers.
	That problem no longer exists, so long_long has been purged from the source.
	The defines below preserve backward compatibility. Their use is deprecated,
	but will continue to be supported in the near term.
*/
#define long_long long long
#define u_long_long unsigned long long

/** @defgroup papi_data_structures PAPI Data Structures */

	typedef unsigned long PAPI_thread_id_t;

	/** @ingroup papi_data_structures */
	typedef struct _papi_all_thr_spec {
     int num;
     PAPI_thread_id_t *id;
     void **data;
   } PAPI_all_thr_spec_t;

  typedef void (*PAPI_overflow_handler_t) (int EventSet, void *address,
                                long long overflow_vector, void *context);

typedef void *vptr_t;

	/** @ingroup papi_data_structures */
   typedef struct _papi_sprofil {
      void *pr_base;          /**< buffer base */
      unsigned pr_size;       /**< buffer size */
      vptr_t pr_off;         /**< pc start address (offset) */
      unsigned pr_scale;      /**< pc scaling factor: 
                                 fixed point fraction
                                 0xffff ~= 1, 0x8000 == .5, 0x4000 == .25, etc.
                                 also, two extensions 0x1000 == 1, 0x2000 == 2 */
   } PAPI_sprofil_t;

/** @ingroup papi_data_structures */
   typedef struct _papi_itimer_option {
     int itimer_num;
     int itimer_sig;
     int ns;
     int flags;
   } PAPI_itimer_option_t;

/** @ingroup papi_data_structures */
   typedef struct _papi_inherit_option {
      int eventset;
      int inherit;
   } PAPI_inherit_option_t;

/** @ingroup papi_data_structures */
   typedef struct _papi_domain_option {
      int def_cidx; /**< this structure requires a component index to set default domains */
      int eventset;
      int domain;
   } PAPI_domain_option_t;

/**  @ingroup papi_data_structures*/
   typedef struct _papi_granularity_option {
      int def_cidx; /**< this structure requires a component index to set default granularity */
      int eventset;
      int granularity;
   } PAPI_granularity_option_t;

/** @ingroup papi_data_structures */
   typedef struct _papi_preload_option {
      char lib_preload_env[PAPI_MAX_STR_LEN];   
      char lib_preload_sep;
      char lib_dir_env[PAPI_MAX_STR_LEN];
      char lib_dir_sep;
   } PAPI_preload_info_t;

/** @ingroup papi_data_structures */
   typedef struct _papi_component_option {
     char name[PAPI_MAX_STR_LEN];            /**< Name of the component we're using */
     char short_name[PAPI_MIN_STR_LEN];      /**< Short name of component,
						to be prepended to event names */
     char description[PAPI_MAX_STR_LEN];     /**< Description of the component */
     char version[PAPI_MIN_STR_LEN];         /**< Version of this component */
     char support_version[PAPI_MIN_STR_LEN]; /**< Version of the support library */
     char kernel_version[PAPI_MIN_STR_LEN];  /**< Version of the kernel PMC support driver */
     char disabled_reason[PAPI_HUGE_STR_LEN]; /**< Reason for failure of initialization */
     int disabled;   /**< 0 if enabled, otherwise error code from initialization */
     int initialized;                        /**< Component is ready to use */
     int CmpIdx;				/**< Index into the vector array for this component; set at init time */
     int num_cntrs;               /**< Number of hardware counters the component supports */
     int num_mpx_cntrs;           /**< Number of hardware counters the component or PAPI can multiplex supports */
     int num_preset_events;       /**< Number of preset events the component supports */
     int num_native_events;       /**< Number of native events the component supports */
     int default_domain;          /**< The default domain when this component is used */
     int available_domains;       /**< Available domains */ 
     int default_granularity;     /**< The default granularity when this component is used */
     int available_granularities; /**< Available granularities */
     int hardware_intr_sig;       /**< Signal used by hardware to deliver PMC events */
     int component_type;          /**< Type of component */
     char *pmu_names[PAPI_PMU_MAX];         /**< list of pmu names supported by this component */
     int reserved[8];             /* */
     unsigned int hardware_intr:1;         /**< hw overflow intr, does not need to be emulated in software*/
     unsigned int precise_intr:1;          /**< Performance interrupts happen precisely */
     unsigned int posix1b_timers:1;        /**< Using POSIX 1b interval timers (timer_create) instead of setitimer */
     unsigned int kernel_profile:1;        /**< Has kernel profiling support (buffered interrupts or sprofil-like) */
     unsigned int kernel_multiplex:1;      /**< In kernel multiplexing */
     unsigned int fast_counter_read:1;     /**< Supports a user level PMC read instruction */
     unsigned int fast_real_timer:1;       /**< Supports a fast real timer */
     unsigned int fast_virtual_timer:1;    /**< Supports a fast virtual timer */
     unsigned int attach:1;                /**< Supports attach */
     unsigned int attach_must_ptrace:1;	   /**< Attach must first ptrace and stop the thread/process*/
     unsigned int cntr_umasks:1;           /**< counters have unit masks */
     /* This should be a granularity option */
     unsigned int cpu:1;                   /**< Supports specifying cpu number to use with event set */
     unsigned int inherit:1;               /**< Supports child processes inheriting parents counters */
     unsigned int reserved_bits:19;
   } PAPI_component_info_t;

/**  @ingroup papi_data_structures*/
   typedef struct _papi_mpx_info {
     int timer_sig;				/**< Signal number used by the multiplex timer, 0 if not: PAPI_SIGNAL */
     int timer_num;				/**< Number of the itimer or POSIX 1 timer used by the multiplex timer: PAPI_ITIMER */
     int timer_us;				/**< uS between switching of sets: PAPI_MPX_DEF_US */
   } PAPI_mpx_info_t;

   typedef int (*PAPI_debug_handler_t) (int code);

   /** @ingroup papi_data_structures */
   typedef struct _papi_debug_option {
      int level;
      PAPI_debug_handler_t handler;
   } PAPI_debug_option_t;

/** @ingroup papi_data_structures
	@brief get the executable's address space info */
   typedef struct _papi_address_map {
      char name[PAPI_HUGE_STR_LEN];
      vptr_t text_start;       /**< Start address of program text segment */
      vptr_t text_end;         /**< End address of program text segment */
      vptr_t data_start;       /**< Start address of program data segment */
      vptr_t data_end;         /**< End address of program data segment */
      vptr_t bss_start;        /**< Start address of program bss segment */
      vptr_t bss_end;          /**< End address of program bss segment */
   } PAPI_address_map_t;

/** @ingroup papi_data_structures
	@brief get the executable's info */
   typedef struct _papi_program_info {
      char fullname[PAPI_HUGE_STR_LEN];  /**< path + name */
      PAPI_address_map_t address_info;	 /**< executable's address space info */
   } PAPI_exe_info_t;

   /** @ingroup papi_data_structures */
   typedef struct _papi_shared_lib_info {
      PAPI_address_map_t *map;
      int count;
   } PAPI_shlib_info_t;

/** Specify the file containing user defined events. */
typedef char* PAPI_user_defined_events_file_t;

   /* The following defines and next for structures define the memory hierarchy */
   /* All sizes are in BYTES */
   /* Associativity:
		0: Undefined;
		1: Direct Mapped
		SHRT_MAX: Full
		Other values == associativity
   */
#define PAPI_MH_TYPE_EMPTY    0x0
#define PAPI_MH_TYPE_INST     0x1
#define PAPI_MH_TYPE_DATA     0x2
#define PAPI_MH_TYPE_VECTOR   0x4
#define PAPI_MH_TYPE_TRACE    0x8
#define PAPI_MH_TYPE_UNIFIED  (PAPI_MH_TYPE_INST|PAPI_MH_TYPE_DATA)
#define PAPI_MH_CACHE_TYPE(a) (a & 0xf)
#define PAPI_MH_TYPE_WT       0x00	   /* write-through cache */
#define PAPI_MH_TYPE_WB       0x10	   /* write-back cache */
#define PAPI_MH_CACHE_WRITE_POLICY(a) (a & 0xf0)
#define PAPI_MH_TYPE_UNKNOWN  0x000
#define PAPI_MH_TYPE_LRU      0x100
#define PAPI_MH_TYPE_PSEUDO_LRU 0x200
#define PAPI_MH_TYPE_FIFO     0x400
#define PAPI_MH_CACHE_REPLACEMENT_POLICY(a) (a & 0xf00)
#define PAPI_MH_TYPE_TLB       0x1000  /* tlb, not memory cache */
#define PAPI_MH_TYPE_PREF      0x2000  /* prefetch buffer */
#define PAPI_MH_TYPE_RD_ALLOC       0x10000	/* read-allocation cache */
#define PAPI_MH_TYPE_WR_ALLOC       0x20000	/* write-allocation cache */
#define PAPI_MH_TYPE_RW_ALLOC       0x40000	/* read-write-allocation cache */
#define PAPI_MH_CACHE_ALLOCATION_POLICY(a) (a & 0xf0000)
#define PAPI_MH_MAX_LEVELS    6		   /* # descriptors for each TLB or cache level */
#define PAPI_MAX_MEM_HIERARCHY_LEVELS 	  4

/** @ingroup papi_data_structures */
   typedef struct _papi_mh_tlb_info {
      int type; /**< Empty, instr, data, vector, unified */
      int num_entries;
      int page_size;
      int associativity;
   } PAPI_mh_tlb_info_t;

/** @ingroup papi_data_structures */
   typedef struct _papi_mh_cache_info {
      int type; /**< Empty, instr, data, vector, trace, unified */
      int size;
      int line_size;
      int num_lines;
      int associativity;
   } PAPI_mh_cache_info_t;

/** @ingroup papi_data_structures */
   typedef struct _papi_mh_level_info {
      PAPI_mh_tlb_info_t   tlb[PAPI_MH_MAX_LEVELS];
      PAPI_mh_cache_info_t cache[PAPI_MH_MAX_LEVELS];
   } PAPI_mh_level_t;

/**  @ingroup papi_data_structures
  *	 @brief mh for mem hierarchy maybe? */
   typedef struct _papi_mh_info { 
      int levels;
      PAPI_mh_level_t level[PAPI_MAX_MEM_HIERARCHY_LEVELS];
   } PAPI_mh_info_t;

/**  @ingroup papi_data_structures
  *  @brief Hardware info structure */
   typedef struct _papi_hw_info {
      int ncpu;                     /**< Number of CPUs per NUMA Node */
      int threads;                  /**< Number of hdw threads per core */
      int cores;                    /**< Number of cores per socket */
      int sockets;                  /**< Number of sockets */
      int nnodes;                   /**< Total Number of NUMA Nodes */
      int totalcpus;                /**< Total number of CPUs in the entire system */
      int vendor;                   /**< Vendor number of CPU */
      char vendor_string[PAPI_MAX_STR_LEN];     /**< Vendor string of CPU */
      int model;                    /**< Model number of CPU */
      char model_string[PAPI_MAX_STR_LEN];      /**< Model string of CPU */
      float revision;               /**< Revision of CPU */
      int cpuid_family;             /**< cpuid family */
      int cpuid_model;              /**< cpuid model */
      int cpuid_stepping;           /**< cpuid stepping */

      int cpu_max_mhz;              /**< Maximum supported CPU speed */
      int cpu_min_mhz;              /**< Minimum supported CPU speed */

      PAPI_mh_info_t mem_hierarchy; /**< PAPI memory hierarchy description */
      int virtualized;              /**< Running in virtual machine */
      char virtual_vendor_string[PAPI_MAX_STR_LEN]; 
                                    /**< Vendor for virtual machine */
      char virtual_vendor_version[PAPI_MAX_STR_LEN];
                                    /**< Version of virtual machine */

      /* Legacy Values, do not use */
      float mhz;                    /**< Deprecated */
      int clock_mhz;                /**< Deprecated */

      /* For future expansion */
      int reserved[8];

   } PAPI_hw_info_t;

/** @ingroup papi_data_structures */
   typedef struct _papi_attach_option {
      int eventset;
      unsigned long tid;
   } PAPI_attach_option_t;

/**  @ingroup papi_data_structures*/
      typedef struct _papi_cpu_option {
         int eventset;
         unsigned int cpu_num;
      } PAPI_cpu_option_t;

/** @ingroup papi_data_structures */
   typedef struct _papi_multiplex_option {
      int eventset;
      int ns;
      int flags;
   } PAPI_multiplex_option_t;

   /** @ingroup papi_data_structures 
	 *  @brief address range specification for range restricted counting if both are zero, range is disabled  */
   typedef struct _papi_addr_range_option { 
      int eventset;           /**< eventset to restrict */
      vptr_t start;          /**< user requested start address of an address range */
      vptr_t end;            /**< user requested end address of an address range */
      int start_off;          /**< hardware specified offset from start address */
      int end_off;            /**< hardware specified offset from end address */
   } PAPI_addr_range_option_t;

/** @ingroup papi_data_structures 
  *	@union PAPI_option_t
  *	@brief A pointer to the following is passed to PAPI_set/get_opt() */

	typedef union
	{
		PAPI_preload_info_t preload;
		PAPI_debug_option_t debug;
		PAPI_inherit_option_t inherit;
		PAPI_granularity_option_t granularity;
		PAPI_granularity_option_t defgranularity;
		PAPI_domain_option_t domain;
		PAPI_domain_option_t defdomain;
		PAPI_attach_option_t attach;
		PAPI_cpu_option_t cpu;
		PAPI_multiplex_option_t multiplex;
		PAPI_itimer_option_t itimer;
		PAPI_hw_info_t *hw_info;
		PAPI_shlib_info_t *shlib_info;
		PAPI_exe_info_t *exe_info;
		PAPI_component_info_t *cmp_info;
		PAPI_addr_range_option_t addr;
		PAPI_user_defined_events_file_t events_file;
	} PAPI_option_t;

/** @ingroup papi_data_structures
  *	@brief A pointer to the following is passed to PAPI_get_dmem_info() */
	typedef struct _dmem_t {
	  long long peak;
	  long long size;
	  long long resident;
	  long long high_water_mark;
	  long long shared;
	  long long text;
	  long long library;
	  long long heap;
	  long long locked;
	  long long stack;
	  long long pagesize;
	  long long pte;
	} PAPI_dmem_info_t;

/* Fortran offsets into PAPI_dmem_info_t structure. */

#define PAPIF_DMEM_VMPEAK     1
#define PAPIF_DMEM_VMSIZE     2
#define PAPIF_DMEM_RESIDENT   3
#define PAPIF_DMEM_HIGH_WATER 4
#define PAPIF_DMEM_SHARED     5
#define PAPIF_DMEM_TEXT       6
#define PAPIF_DMEM_LIBRARY    7
#define PAPIF_DMEM_HEAP       8
#define PAPIF_DMEM_LOCKED     9
#define PAPIF_DMEM_STACK      10
#define PAPIF_DMEM_PAGESIZE   11
#define PAPIF_DMEM_PTE        12
#define PAPIF_DMEM_MAXVAL     12

#define PAPI_MAX_INFO_TERMS  12		   /* should match PAPI_EVENTS_IN_DERIVED_EVENT defined in papi_internal.h */


/** @ingroup papi_data_structures 
  *	@brief This structure is the event information that is exposed to the user through the API.

   The same structure is used to describe both preset and native events.
   WARNING: This structure is very large. With current definitions, it is about 2660 bytes.
   Unlike previous versions of PAPI, which allocated an array of these structures within
   the library, this structure is carved from user space. It does not exist inside the library,
   and only one copy need ever exist.
   The basic philosophy is this:
   - each preset consists of a code, some descriptors, and an array of native events;
   - each native event consists of a code, and an array of register values;
   - fields are shared between preset and native events, and unused where not applicable;
   - To completely describe a preset event, the code must present all available
      information for that preset, and then walk the list of native events, retrieving
      and presenting information for each native event in turn.
   The various fields and their usage is discussed below.
*/


/** Enum values for event_info location field */
enum {
   PAPI_LOCATION_CORE = 0,			/**< Measures local to core      */
   PAPI_LOCATION_CPU,				/**< Measures local to CPU (HT?) */
   PAPI_LOCATION_PACKAGE,			/**< Measures local to package   */
   PAPI_LOCATION_UNCORE,			/**< Measures uncore             */
};

/** Enum values for event_info data_type field */
enum {
   PAPI_DATATYPE_INT64 = 0,			/**< Default: Data is a signed 64-bit int   */
   PAPI_DATATYPE_UINT64,			/**< Data is a unsigned 64-bit int */
   PAPI_DATATYPE_FP64,				/**< Data is 64-bit floating point */
   PAPI_DATATYPE_BIT64,				/**< Data is 64-bit binary */
};

/** Enum values for event_info value_type field */
enum {
   PAPI_VALUETYPE_RUNNING_SUM = 0,	/**< Data is running sum from start */
   PAPI_VALUETYPE_ABSOLUTE,	        /**< Data is from last read */
};

/** Enum values for event_info timescope field */
enum {
   PAPI_TIMESCOPE_SINCE_START = 0,	/**< Data is cumulative from start */
   PAPI_TIMESCOPE_SINCE_LAST,		/**< Data is from last read */
   PAPI_TIMESCOPE_UNTIL_NEXT,		/**< Data is until next read */
   PAPI_TIMESCOPE_POINT,			/**< Data is an instantaneous value */
};

/** Enum values for event_info update_type field */
enum {
   PAPI_UPDATETYPE_ARBITRARY = 0,	/**< Data is cumulative from start */
   PAPI_UPDATETYPE_PUSH,	        /**< Data is pushed */
   PAPI_UPDATETYPE_PULL,	        /**< Data is pulled */
   PAPI_UPDATETYPE_FIXEDFREQ,	    /**< Data is read periodically */
};


   typedef struct event_info {
      unsigned int event_code;             /**< preset (0x8xxxxxxx) or 
                                                native (0x4xxxxxxx) event code */
      char symbol[PAPI_HUGE_STR_LEN];      /**< name of the event */
      char short_descr[PAPI_MIN_STR_LEN];  /**< a short description suitable for 
                                                use as a label */
      char long_descr[PAPI_HUGE_STR_LEN];  /**< a longer description:
                                                typically a sentence for presets,
                                                possibly a paragraph from vendor
                                                docs for native events */

      int component_index;           /**< component this event belongs to */
      char units[PAPI_MIN_STR_LEN];  /**< units event is measured in */
      int location;                  /**< location event applies to */
      int data_type;                 /**< data type returned by PAPI */
      int value_type;                /**< sum or absolute */
      int timescope;                 /**< from start, etc. */
      int update_type;               /**< how event is updated */
      int update_freq;               /**< how frequently event is updated */

     /* PRESET SPECIFIC FIELDS FOLLOW */



      unsigned int count;                /**< number of terms (usually 1) 
                                              in the code and name fields 
                                              - presets: these are native events
                                              - native: these are unused */

      unsigned int event_type;           /**< event type or category 
                                              for preset events only */

      char derived[PAPI_MIN_STR_LEN];    /**< name of the derived type
                                              - presets: usually NOT_DERIVED
                                              - native: empty string */
      char postfix[PAPI_2MAX_STR_LEN];   /**< string containing postfix 
                                              operations; only defined for 
                                              preset events of derived type 
                                              DERIVED_POSTFIX */

      unsigned int code[PAPI_MAX_INFO_TERMS]; /**< array of values that further 
                                              describe the event:
                                              - presets: native event_code values
                                              - native:, register values(?) */

      char name[PAPI_MAX_INFO_TERMS]         /**< names of code terms: */
               [PAPI_2MAX_STR_LEN];          /**< - presets: native event names,
                                                  - native: descriptive strings 
						  for each register value(?) */

     char note[PAPI_HUGE_STR_LEN];          /**< an optional developer note 
                                                supplied with a preset event
                                                to delineate platform specific 
						anomalies or restrictions */

   } PAPI_event_info_t;


/** @ingroup papi_data_structures
 * PAPI_dev_type_id_e - enum device types
 *
 * Device types are defined, in most cases, by the device runtime used
 * to access its attributes. For devices that expose their attributes
 * through the OS interfaces only the device name is used (e.g., CPU).
 */
typedef enum {
    PAPI_DEV_TYPE_ID__CPU,
    PAPI_DEV_TYPE_ID__CUDA,
    PAPI_DEV_TYPE_ID__ROCM,
    PAPI_DEV_TYPE_ID__MAX_NUM,
} PAPI_dev_type_id_e;

/** @ingroup papi_data_structures
 * enum of device types.
 *
 * Device type are identified, most of the times, by the runtime used
 * to access them (e.g., CUDA, ROCM, L0, etc). For devices that expose
 * their attributes through the operating system interfaces the identification
 * is the device name (e.g., CPU).
 */
enum {
    PAPI_DEV_TYPE_ENUM__FIRST= (0                          ),
    PAPI_DEV_TYPE_ENUM__CPU  = (1 << PAPI_DEV_TYPE_ID__CPU ),
    PAPI_DEV_TYPE_ENUM__CUDA = (1 << PAPI_DEV_TYPE_ID__CUDA),
    PAPI_DEV_TYPE_ENUM__ROCM = (1 << PAPI_DEV_TYPE_ID__ROCM),
    PAPI_DEV_TYPE_ENUM__ALL  = (1 << PAPI_DEV_TYPE_ID__MAX_NUM) - 1,
};

/** @ingroup papi_data_structures
 * PAPI_dev_type_attr_e - enum device type attributes.
 *
 */
typedef enum {
    PAPI_DEV_TYPE_ATTR__INT_PAPI_ID,
    PAPI_DEV_TYPE_ATTR__INT_VENDOR_ID,
    PAPI_DEV_TYPE_ATTR__CHAR_NAME,
    PAPI_DEV_TYPE_ATTR__INT_COUNT,
    PAPI_DEV_TYPE_ATTR__CHAR_STATUS,
} PAPI_dev_type_attr_e;

/** @ingroup papi_data_structures
 * PAPI_dev_attr_e - enum device attributes
 *
 * As the PAPI_get_dev_attr interface returns a pointer to void as attribute
 * value, the attribute name has the following format:
 *
 *   PAPI_DEV_ATTR__<DEVICE>_<ATTRIBUTE_TYPE>_<ATTRIBUTE_NAME>
 *
 * This identifies, in order, the device for which the attribute is being
 * queried, the type of the attribute returned, and the attribute name.
 */
typedef enum {
    PAPI_DEV_ATTR__CPU_CHAR_NAME,
    PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_SIZE,
    PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_SIZE,
    PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_SIZE,
    PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_SIZE,
    PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_LINE_SIZE,
    PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_LINE_SIZE,
    PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_LINE_SIZE,
    PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_LINE_SIZE,
    PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_LINE_COUNT,
    PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_LINE_COUNT,
    PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_LINE_COUNT,
    PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_LINE_COUNT,
    PAPI_DEV_ATTR__CPU_UINT_L1I_CACHE_ASSOC,
    PAPI_DEV_ATTR__CPU_UINT_L1D_CACHE_ASSOC,
    PAPI_DEV_ATTR__CPU_UINT_L2U_CACHE_ASSOC,
    PAPI_DEV_ATTR__CPU_UINT_L3U_CACHE_ASSOC,
    PAPI_DEV_ATTR__CPU_UINT_SOCKET_COUNT,
    PAPI_DEV_ATTR__CPU_UINT_NUMA_COUNT,
    PAPI_DEV_ATTR__CPU_UINT_CORE_COUNT,
    PAPI_DEV_ATTR__CPU_UINT_THREAD_COUNT,
    PAPI_DEV_ATTR__CPU_UINT_FAMILY,
    PAPI_DEV_ATTR__CPU_UINT_MODEL,
    PAPI_DEV_ATTR__CPU_UINT_STEPPING,
    PAPI_DEV_ATTR__CPU_UINT_NUMA_MEM_SIZE,
    PAPI_DEV_ATTR__CPU_UINT_THR_NUMA_AFFINITY,
    PAPI_DEV_ATTR__CPU_UINT_THR_PER_NUMA,
    PAPI_DEV_ATTR__CUDA_ULONG_UID,
    PAPI_DEV_ATTR__CUDA_CHAR_DEVICE_NAME,
    PAPI_DEV_ATTR__CUDA_UINT_WARP_SIZE,
    PAPI_DEV_ATTR__CUDA_UINT_SHM_PER_BLK,
    PAPI_DEV_ATTR__CUDA_UINT_SHM_PER_SM,
    PAPI_DEV_ATTR__CUDA_UINT_BLK_DIM_X,
    PAPI_DEV_ATTR__CUDA_UINT_BLK_DIM_Y,
    PAPI_DEV_ATTR__CUDA_UINT_BLK_DIM_Z,
    PAPI_DEV_ATTR__CUDA_UINT_GRD_DIM_X,
    PAPI_DEV_ATTR__CUDA_UINT_GRD_DIM_Y,
    PAPI_DEV_ATTR__CUDA_UINT_GRD_DIM_Z,
    PAPI_DEV_ATTR__CUDA_UINT_THR_PER_BLK,
    PAPI_DEV_ATTR__CUDA_UINT_SM_COUNT,
    PAPI_DEV_ATTR__CUDA_UINT_MULTI_KERNEL,
    PAPI_DEV_ATTR__CUDA_UINT_MAP_HOST_MEM,
    PAPI_DEV_ATTR__CUDA_UINT_MEMCPY_OVERLAP,
    PAPI_DEV_ATTR__CUDA_UINT_UNIFIED_ADDR,
    PAPI_DEV_ATTR__CUDA_UINT_MANAGED_MEM,
    PAPI_DEV_ATTR__CUDA_UINT_COMP_CAP_MAJOR,
    PAPI_DEV_ATTR__CUDA_UINT_COMP_CAP_MINOR,
    PAPI_DEV_ATTR__CUDA_UINT_BLK_PER_SM,
    PAPI_DEV_ATTR__ROCM_ULONG_UID,
    PAPI_DEV_ATTR__ROCM_CHAR_DEVICE_NAME,
    PAPI_DEV_ATTR__ROCM_UINT_WAVEFRONT_SIZE,
    PAPI_DEV_ATTR__ROCM_UINT_WORKGROUP_SIZE,
    PAPI_DEV_ATTR__ROCM_UINT_WAVE_PER_CU,
    PAPI_DEV_ATTR__ROCM_UINT_SHM_PER_WG,
    PAPI_DEV_ATTR__ROCM_UINT_WG_DIM_X,
    PAPI_DEV_ATTR__ROCM_UINT_WG_DIM_Y,
    PAPI_DEV_ATTR__ROCM_UINT_WG_DIM_Z,
    PAPI_DEV_ATTR__ROCM_UINT_GRD_DIM_X,
    PAPI_DEV_ATTR__ROCM_UINT_GRD_DIM_Y,
    PAPI_DEV_ATTR__ROCM_UINT_GRD_DIM_Z,
    PAPI_DEV_ATTR__ROCM_UINT_CU_COUNT,
    PAPI_DEV_ATTR__ROCM_UINT_SIMD_PER_CU,
    PAPI_DEV_ATTR__ROCM_UINT_COMP_CAP_MAJOR,
    PAPI_DEV_ATTR__ROCM_UINT_COMP_CAP_MINOR,
} PAPI_dev_attr_e;


/** \internal
  * @defgroup low_api The Low Level API 
  @{ */
   int   PAPI_accum(int EventSet, long long * values); /**< accumulate and reset hardware events from an event set */
   int   PAPI_add_event(int EventSet, int Event); /**< add single PAPI preset or native hardware event to an event set */
   int   PAPI_add_named_event(int EventSet, const char *EventName); /**< add an event by name to a PAPI event set */
   int   PAPI_add_events(int EventSet, int *Events, int number); /**< add array of PAPI preset or native hardware events to an event set */
   int   PAPI_assign_eventset_component(int EventSet, int cidx); /**< assign a component index to an existing but empty eventset */
   int   PAPI_attach(int EventSet, unsigned long tid); /**< attach specified event set to a specific process or thread id */
   int   PAPI_cleanup_eventset(int EventSet); /**< remove all PAPI events from an event set */
   int   PAPI_create_eventset(int *EventSet); /**< create a new empty PAPI event set */
   int   PAPI_detach(int EventSet); /**< detach specified event set from a previously specified process or thread id */
   int   PAPI_destroy_eventset(int *EventSet); /**< deallocates memory associated with an empty PAPI event set */
   int   PAPI_enum_event(int *EventCode, int modifier); /**< return the event code for the next available preset or natvie event */
   int   PAPI_enum_cmp_event(int *EventCode, int modifier, int cidx); /**< return the event code for the next available component event */
   int   PAPI_event_code_to_name(int EventCode, char *out); /**< translate an integer PAPI event code into an ASCII PAPI preset or native name */
   int   PAPI_event_name_to_code(const char *in, int *out); /**< translate an ASCII PAPI preset or native name into an integer PAPI event code */
   int  PAPI_get_dmem_info(PAPI_dmem_info_t *dest); /**< get dynamic memory usage information */
   int   PAPI_get_event_info(int EventCode, PAPI_event_info_t * info); /**< get the name and descriptions for a given preset or native event code */
   const PAPI_exe_info_t *PAPI_get_executable_info(void); /**< get the executable's address space information */
   const PAPI_hw_info_t *PAPI_get_hardware_info(void); /**< get information about the system hardware */
   const PAPI_component_info_t *PAPI_get_component_info(int cidx); /**< get information about the component features */
   int   PAPI_get_multiplex(int EventSet); /**< get the multiplexing status of specified event set */
   int   PAPI_get_opt(int option, PAPI_option_t * ptr); /**< query the option settings of the PAPI library or a specific event set */
   int   PAPI_get_cmp_opt(int option, PAPI_option_t * ptr,int cidx); /**< query the component specific option settings of a specific event set */
   long long PAPI_get_real_cyc(void); /**< return the total number of cycles since some arbitrary starting point */
   long long PAPI_get_real_nsec(void); /**< return the total number of nanoseconds since some arbitrary starting point */
   long long PAPI_get_real_usec(void); /**< return the total number of microseconds since some arbitrary starting point */
   const PAPI_shlib_info_t *PAPI_get_shared_lib_info(void); /**< get information about the shared libraries used by the process */
   int   PAPI_get_thr_specific(int tag, void **ptr); /**< return a pointer to a thread specific stored data structure */
   int   PAPI_get_overflow_event_index(int Eventset, long long overflow_vector, int *array, int *number); /**< # decomposes an overflow_vector into an event index array */
   long long PAPI_get_virt_cyc(void); /**< return the process cycles since some arbitrary starting point */
   long long PAPI_get_virt_nsec(void); /**< return the process nanoseconds since some arbitrary starting point */
   long long PAPI_get_virt_usec(void); /**< return the process microseconds since some arbitrary starting point */
   int   PAPI_is_initialized(void); /**< return the initialized state of the PAPI library */
   int   PAPI_library_init(int version); /**< initialize the PAPI library */
   int   PAPI_list_events(int EventSet, int *Events, int *number); /**< list the events that are members of an event set */
   int   PAPI_list_threads(unsigned long *tids, int *number); /**< list the thread ids currently known to PAPI */
   int   PAPI_lock(int); /**< lock one of two PAPI internal user mutex variables */
   int   PAPI_multiplex_init(void); /**< initialize multiplex support in the PAPI library */
   int   PAPI_num_cmp_hwctrs(int cidx); /**< return the number of hardware counters for a specified component */
   int   PAPI_num_events(int EventSet); /**< return the number of events in an event set */
   int   PAPI_overflow(int EventSet, int EventCode, int threshold,
                     int flags, PAPI_overflow_handler_t handler); /**< set up an event set to begin registering overflows */
   void  PAPI_perror(const char *msg ); /**< Print a PAPI error message */
   int   PAPI_profil(void *buf, unsigned bufsiz, vptr_t offset,
					 unsigned scale, int EventSet, int EventCode,
					 int threshold, int flags); /**< generate PC histogram data where hardware counter overflow occurs */
   int   PAPI_query_event(int EventCode); /**< query if a PAPI event exists */
   int   PAPI_query_named_event(const char *EventName); /**< query if a named PAPI event exists */
   int   PAPI_read(int EventSet, long long * values); /**< read hardware events from an event set with no reset */
   int   PAPI_read_ts(int EventSet, long long * values, long long *cyc); /**< read from an eventset with a real-time cycle timestamp */
   int   PAPI_register_thread(void); /**< inform PAPI of the existence of a new thread */
   int   PAPI_remove_event(int EventSet, int EventCode); /**< remove a hardware event from a PAPI event set */
   int   PAPI_remove_named_event(int EventSet, const char *EventName); /**< remove a named event from a PAPI event set */
   int   PAPI_remove_events(int EventSet, int *Events, int number); /**< remove an array of hardware events from a PAPI event set */
   int   PAPI_reset(int EventSet); /**< reset the hardware event counts in an event set */
   int   PAPI_set_debug(int level); /**< set the current debug level for PAPI */
   int   PAPI_set_cmp_domain(int domain, int cidx); /**< set the component specific default execution domain for new event sets */
   int   PAPI_set_domain(int domain); /**< set the default execution domain for new event sets  */
   int   PAPI_set_cmp_granularity(int granularity, int cidx); /**< set the component specific default granularity for new event sets */
   int   PAPI_set_granularity(int granularity); /**<set the default granularity for new event sets */
   int   PAPI_set_multiplex(int EventSet); /**< convert a standard event set to a multiplexed event set */
   int   PAPI_set_opt(int option, PAPI_option_t * ptr); /**< change the option settings of the PAPI library or a specific event set */
   int   PAPI_set_thr_specific(int tag, void *ptr); /**< save a pointer as a thread specific stored data structure */
   void  PAPI_shutdown(void); /**< finish using PAPI and free all related resources */
   int   PAPI_sprofil(PAPI_sprofil_t * prof, int profcnt, int EventSet, int EventCode, int threshold, int flags); /**< generate hardware counter profiles from multiple code regions */
   int   PAPI_start(int EventSet); /**< start counting hardware events in an event set */
   int   PAPI_state(int EventSet, int *status); /**< return the counting state of an event set */
   int   PAPI_stop(int EventSet, long long * values); /**< stop counting hardware events in an event set and return current events */
   char *PAPI_strerror(int); /**< return a pointer to the error name corresponding to a specified error code */
   unsigned long PAPI_thread_id(void); /**< get the thread identifier of the current thread */
   int   PAPI_thread_init(unsigned long (*id_fn) (void)); /**< initialize thread support in the PAPI library */
   int   PAPI_unlock(int); /**< unlock one of two PAPI internal user mutex variables */
   int   PAPI_unregister_thread(void); /**< inform PAPI that a previously registered thread is disappearing */
   int   PAPI_write(int EventSet, long long * values); /**< write counter values into counters */
   int   PAPI_get_event_component(int EventCode);  /**< return which component an EventCode belongs to */
   int   PAPI_get_eventset_component(int EventSet);  /**< return which component an EventSet is assigned to */
   int   PAPI_get_component_index(const char *name); /**< Return component index for component with matching name */
   int   PAPI_disable_component(int cidx); /**< Disables a component before init */
   int	 PAPI_disable_component_by_name(const char *name ); /**< Disable, before library init, a component by name. */
   int PAPI_num_components(void); /**< get the number of components available on the system */

   int PAPI_flips_rate(int event, float *rtime, float *ptime, long long *flpins, float *mflips); /**< simplified call to get Mflips/s (floating point instruction rate), real and processor time */
   int PAPI_flops_rate(int event, float *rtime, float *ptime, long long * flpops, float *mflops); /**< simplified call to get Mflops/s (floating point operation rate), real and processor time */
   int PAPI_ipc(float *rtime, float *ptime, long long * ins, float *ipc); /**< gets instructions per cycle, real and processor time */
   int PAPI_epc(int event, float *rtime, float *ptime, long long *ref, long long *core, long long *evt, float *epc); /**< gets (named) events per cycle, real and processor time, reference and core cycles */
   int PAPI_rate_stop(void); /**< stops a running event set of a rate function */

   int PAPI_enum_dev_type(int enum_modifier, void **handle); /**< return the handler for the next device type available */
   int PAPI_get_dev_type_attr(void *handle, PAPI_dev_type_attr_e attr, void *value); /**< return the value of the queried attribute for the device type handle */
   int PAPI_get_dev_attr(void *handle, int id, PAPI_dev_attr_e attr, void *value); /**< return the value of the queried attribute for the device handle */

   /** @} */

/** \internal
  @defgroup high_api  The High Level API 

   The simple interface implemented by the following routines allows the user to record hardware events inside instrumented regions from both C and Fortran.
   @{ */

   int PAPI_hl_region_begin(const char* region); /**< read performance events at the beginning of a region */
   int PAPI_hl_read(const char* region); /**< read performance events inside of a region and store the difference to the corresponding beginning of the region */
   int PAPI_hl_region_end(const char* region); /**< read performance events at the end of a region and store the difference to the corresponding beginning of the region */
   int PAPI_hl_stop(void); /**< stops a running high-level event set */
/** @} */


/* Backwards compatibility hacks.  Remove eventually? */
int   PAPI_num_hwctrs(void); /**< return the number of hardware counters for the cpu. for backward compatibility. Don't use! */
#define PAPI_COMPONENT_INDEX(a) PAPI_get_event_component(a)
#define PAPI_descr_error(a) PAPI_strerror(a)

#ifdef __cplusplus
}
#endif

#pragma GCC visibility pop

#endif
