//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// This file contains the functions needed to interface PAPI with PCP, the 
// Performance Co-Pilot package. A manual for pmXXX commands is online at 
// https://pcp.io/man/
// Performance: As tested on the ICL Saturn system, round-trip communications 
// with the PCP Daemon cost us 8-10ms, so every pm___ call is a stall. We have
// to create our event list when initialized, but we can do that with some 
// 'batch' reads to minimize overhead. 
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
// #define DEBUG /* To enable SUBDBG messages */
// see also _papi_hwi_debug = DEBUG_SUBSTRATE; below, to enable xxxDBG macros.

#include <unistd.h>
#include <errno.h>
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <ctype.h>
#include <math.h>
#include <sys/time.h>

// Headers required by PAPI.
#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "papi_debug.h" // for FUNC macro in error messages, keep even if we have no xxxDBG messages.
#include <dlfcn.h>      // Dynamic lib routines; especially dlsym to get func ptrs.

#ifndef FUNC            /* Not defined if not DEBUG... */
#define  FUNC __func__  /* force it. */ 
#endif 

// Event Name filters, used in init_component for the routine pmTraversePMNS(). Differs by system.
//#define AGENT_NAME "xfs"          /* Saturn PCP. */
//#define AGENT_NAME "mem"          /* Saturn PCP. */
  #define AGENT_NAME "perfevent"    /* Power9 PCP. */
//#define AGENT_NAME ""             /* Get it all! */

/* To remove redefined warnings */
#undef PACKAGE_BUGREPORT
#undef PACKAGE_TARNAME
#undef PACKAGE_NAME
#undef PACKAGE_STRING
#undef PACKAGE_VERSION

// PCP include directory (defaults to /usr/include/pcp; see README for PAPI_PCP_INC.)
#include <pmapi.h> // See https://pcp.io/man/man3/pmapi.3.html for routines.
#include <impl.h>  // also a PCP file.  

#define   PM_OPTFLAG_EXIT     (1<<5)
#define   PM_CONTEXT_UNDEF    -1
#define   PM_CONTEXT_HOST     1
#define   PM_CONTEXT_ARCHIVE  2
#define   PM_CONTEXT_LOCAL    3

// The following macro follows if a string function has an error. It should 
// never happen; but it is necessary to prevent compiler warnings. We print 
// something just in case there is programmer error in invoking the function.
#define HANDLE_STRING_ERROR {fprintf(stderr,"%s:%i unexpected string function error.\n",__FILE__,__LINE__); exit(-1);}

//-----------------------------------------------------------------------------
// Union to convert pointers and avoid warnings. Plug in one, pull out the other.
//-----------------------------------------------------------------------------
typedef union  
{
   void                 *vPtr;
   int                  *iPtr;
   unsigned int         *uiPtr;
   long                 *lPtr;
   long long            *llPtr;
   unsigned long long   *ullPtr;
   float                *fPtr;
   double               *dPtr;
   char                 *cPtr;
} uPointer_t;

typedef union
{
   long long ll;
   unsigned long long ull;
   double    d;
   void *vp;
   unsigned char ch[8];
} convert_64_t;

//-----------------------------------------------------------------------------
// Structure to store private information on each Event. 
//-----------------------------------------------------------------------------
typedef struct _pcp_register  
{
   unsigned int selector;                    // indexed from 1, index of counter slot, but 0 means none.
} _pcp_register_t;


//-----------------------------------------------------------------------------
// WARNING: Do NOT expect pointers into the pcp_event_info[] array to remain
// valid during processing; the list is realloc() to make room and this can
// i_nvalidate the pointer. (Hard won knowledge). 
//-----------------------------------------------------------------------------
typedef struct _pcp_event_info               // an array of these is populated by our pcp create event routine.
{
   char      name[PAPI_MAX_STR_LEN];         // name (we never copy more than PAPI_MAX_STR_LEN-1, so always z-terminated).
   pmID      pmid;                           // PCP's unique id (gets only base name).
   pmDesc    desc;                           // Description (not text, just var type etc). desc.pmid = PM_ID_NULL if not read yet.
   int       valfmt;                         // Value format from Fetch (PM_VAL_INSITU, PM_VAL_DPTR [dynamic], PM_VAL_SPTR [static]
   int       valType;                        // Type of value (PM_TYPE_[32,U32,64,U64,FLOAT,DOUBLE,STRING).
   char      domainDesc[PAPI_MAX_STR_LEN];   // Domain description if not null.
   int       numVal;                         // number of values in array.
   int       idx;                            // idx into vlist array.
   unsigned long long zeroValue;             // Value that counts as zero.
} _pcp_event_info_t;


//-----------------------------------------------------------------------------
// This structure is used when doing register allocation it possibly is not
// necessary when there are no register constraints.
//-----------------------------------------------------------------------------
typedef struct _pcp_reg_alloc  
{
   _pcp_register_t ra_bits;
} _pcp_reg_alloc_t;


//-----------------------------------------------------------------------------
// Holds control flags. There's one of these per event-set. Use this to hold
// data specific to the EventSet. We hold for each event the index into the
// pcp_event_info[] array, and a corresponding value; they must grow in
// lockstep; to be NULL together or [maxAllocated] together. We cannot create a
// structure to make that automatic; we need to point to an array of long long
// values alone after a read.
//-----------------------------------------------------------------------------

typedef struct _pcp_control_state  
{
   int numEvents;                                  // The number of events we have.
   int maxAllocated;                               // The most ever allocated.
   int *pcpIndex;                                  // array of indices into pcp_event_info[].
   unsigned long long *pcpValue;                   // corresponding value read.
} _pcp_control_state_t;


//-----------------------------------------------------------------------------
// Holds per-thread information. We don't have any.
//-----------------------------------------------------------------------------
typedef struct _pcp_context  
{
   int initDone;                                   // does nothing.
} _pcp_context_t;

//-----------------------------------------------------------------------------
// Our hash table entry. We use a hash table to help lookup event names. Each
// entry contains an index into the pcp_event_info table, and a chain pointer
// to the next hash entry if the same hash results for several event names.
// See the routines addNameHash(), freeNameHash(), findNameHash().  
//-----------------------------------------------------------------------------
typedef struct _pcp_hash                           // hash table entry.'
{
   int idx;                                        // The entry that matches this hash.
   void *next;                                     // next entry that matches this hash, or NULL.
}_pcp_hash_t;

//-----------------------------------------------------------------------------
// We cache our domains to save costly reads from the daemon, using a table of
// this structure per entry. The domain id is not a counter, but there haven't
// been many of them, so we do a sequential search of the table.
//-----------------------------------------------------------------------------
typedef struct 
{
   pmInDom domainId;                               // The id.
   int     numInstances;                           // size of the arrays below.
   int     *instances;                             // the instances. cleanup must free(instances).
   char    **names;                                // The names. cleanup (must free(names).
} _pcp_domain_cache_t;


papi_vector_t _pcp_vector;                         // What we expose to PAPI, routine ptrs and other values.


// -------------------------- GLOBAL SECTION ---------------------------------

       int  _papi_hwi_debug = DEBUG_SUBSTRATE;                          // Bit flags to enable xxxDBG; SUBDBG for Substrate. Overrides weak global in papi.c.
static int  sEventInfoSize=0;                                           // total size of pcp_event_info.
static int  sEventInfoBlock = ((8*1024) / sizeof(_pcp_event_info_t));   // add about 8K at a time.
static      _pcp_event_info_t * pcp_event_info = NULL;                  // our array of created pcp events.
static int  sEventCount = 0;                                            // count of events seen by pmTraversePMNS().
static int  ctxHandle = -1;                                             // context handle. (-1 is invalid).
static char *cachedGetInDom(pmInDom indom, int inst);                   // cache all reads of pcp_pmGetInDom, to save time.
#define     HASH_SIZE 512                                               /* very roughly in the range of total events. full Saturn test, had ~ 11,000 events.*/
static      _pcp_hash_t sNameHash[HASH_SIZE];                           // hash table into pcp_event_info by event name.

#define COUNT_ROUTINES 1                                                /* Change to zero to stop counting. */
#if (COUNT_ROUTINES == 1)
enum {
ctr_pcp_init_thread,                               // counter 0
ctr_pcp_init_component,                            // counter 1
ctr_pcp_init_control_state,                        // counter 2
ctr_pcp_start,                                     // counter 3
ctr_pcp_stop,                                      // counter 4
ctr_pcp_read,                                      // counter 5
ctr_pcp_shutdown_thread,                           // counter 6
ctr_pcp_shutdown_component,                        // counter 7
ctr_pcp_ctl,                                       // counter 8
ctr_pcp_update_control_state,                      // counter 9
ctr_pcp_set_domain,                                // counter 10
ctr_pcp_reset,                                     // counter 11
ctr_pcp_ntv_enum_events,                           // counter 12
ctr_pcp_ntv_name_to_code,                          // counter 13
ctr_pcp_ntv_code_to_name,                          // counter 14
ctr_pcp_ntv_code_to_descr,                         // counter 15
ctr_pcp_ntv_code_to_info};                         // counter 16

static int cnt[ctr_pcp_ntv_code_to_info+1] = {0};  // counters for the following macro.

#define mRtnCnt(funcname) \
   if (COUNT_ROUTINES) {            /* Note if (0) optimized out completely even if -O0. */  \
      cnt[ctr##funcname]++;         /* Increment counter. */                                 \
      if (cnt[ctr##funcname] == 1)  /* If first time entering a new function, report it. */  \
         _prog_fprintf(stderr, "Entered " TOSTRING(funcname) "\n");                          \
   }
#else                                                                   /* If COUNT_ROUTINES != 1, */
#define mRtnCnt(funcname)                                               /* .. make macro do nothing. */
#endif /* if/else for COUNT_ROUTINES handled. */

//--------------------------------------------------------------------
// Timing of routines and blocks. Typical usage;
// _time_gettimeofday(&t1, NULL);                  // starting point.
// ... some code to execute ...
// _time_gettimeofday(&t2, NULL);                  // finished timing.
// _time_fprintf(stderr, "routine took %li uS.\n", // report time.
//                       (mConvertUsec(t2)-mConvertUsec(t1)));
//--------------------------------------------------------------------
static struct timeval t1, t2;                                           // used in timing routines to measure performance.
#define mConvertUsec(timeval_) \
        (timeval_.tv_sec*1000000+timeval_.tv_usec)                      /* avoid typos. */

#define _prog_fprintf if (0) fprintf                                    /* change to 1 to enable printing of progress debug messages.     */
#define _time_fprintf if (0) fprintf                                    /* change to 1 to enable printing of performance timings.         */
#define _time_gettimeofday if (0) gettimeofday                          /* change to 1 to enable gettimeofday for performance timings.    */


// file handle used to access pcp library with dlopen
static void *dl1 = NULL;

// string macro defined within Rules.pcp
static char pcp_main[]=PAPI_PCP_MAIN;

//-----------------------------------------------------------------------------
// Using weak symbols (global declared without a value, so it defers to any
// other global declared in another file WITH a value) allows PAPI to be built
// with the component, but PAPI can still be installed in a system without the
// required library.
//-----------------------------------------------------------------------------

void (*_dl_non_dynamic_init)(void) __attribute__((weak));               // declare a weak dynamic-library init routine pointer.

// ------------------------ LINK DYNAMIC LIBRARIES ---------------------------
// Function pointers for the PCP lib; begin with pm (performance monitor).
// Commented out functions are ones we do not use; so we don't waste space.
static int     (*pmLookupName_ptr)     (int numpid, char **namelist,pmID *pmidlist);
static char*   (*pmErrStr_ptr)         (int code);
static int     (*pmTraversePMNS_ptr)   (const char *name, void(*func)(const char *));
static void    (*pmFreeResult_ptr)     (pmResult *result);
static int     (*pmNewContext_ptr)     (int type, const char *name);
static int     (*pmDestroyContext_ptr) (int handle);
static int     (*pmFetch_ptr)          (int numpid, pmID *pmidlist, pmResult **result);
static int     (*pmLookupDesc_ptr)     (pmID pmid, pmDesc *desc);
static int     (*pmGetInDom_ptr)       (pmInDom indom, int **instlist, char ***namelist);
static int     (*pmLookupText_ptr)     (pmID pmid, int level, char **buffer);
static char *  (*pmUnitsStr_r_ptr)     (const pmUnits *pu, char *buf, int buflen); 

// -------------------- LOCAL WRAPPERS FOR LIB FUNCTIONS ---------------------
static int     pcp_pmLookupName (int numpid, char **namelist, pmID *pmidlist) 
                  { return ((*pmLookupName_ptr) (numpid, namelist, pmidlist)); }

static char*   pcp_pmErrStr (int code) 
                  { return ((*pmErrStr_ptr) (code)); }

static int     pcp_pmTraversePMNS (const char *name, void(*func)(const char *)) 
                  { return ((*pmTraversePMNS_ptr) (name, func)); }

static void    pcp_pmFreeResult (pmResult *result) 
                  {(*pmFreeResult_ptr) (result); return;}

static int     pcp_pmNewContext (int type, const char *name) 
                  { return ((*pmNewContext_ptr) (type,name)); }

static int     pcp_pmDestroyContext(int handle) 
                  { return ((*pmDestroyContext_ptr) (handle));}

static int     pcp_pmFetch (int numpid, pmID *pmidlist, pmResult **result) 
                  { return ((*pmFetch_ptr) (numpid,pmidlist,result)); }

static int     pcp_pmLookupDesc (pmID pmid, pmDesc *desc) 
                  { return ((*pmLookupDesc_ptr) (pmid,desc)); }

static int     pcp_pmGetInDom (pmInDom indom, int **instlist, char ***namelist) 
                  { return ((*pmGetInDom_ptr) (indom,instlist,namelist)); }

static int     pcp_pmLookupText(pmID pmid, int level, char **buffer) 
                  { return ((*pmLookupText_ptr) (pmid, level, buffer)); }

static char*   pcp_pmUnitsStr_r (const pmUnits *pu, char *buf, int buflen) 
                  {return ((*pmUnitsStr_r_ptr) (pu, buf, buflen)); }


//-----------------------------------------------------------------------------
// stringHash: returns unsigned long value for hashed string.  See djb2, Dan
// Bernstein, http://www.cse.yorku.ca/~oz/hash.html Empirically a fast well
// distributed hash, not theoretically explained.  On a test system with 1857
// events, this gets about a 65% density in a 2000 element table; 35% of slots
// have dups; max dups was 4.
//-----------------------------------------------------------------------------

static unsigned int stringHash(char *str, unsigned int tableSize) 
{
  unsigned long hash = 5381;                             // seed value.
  int c;
  while ((c = (*str++))) {                               // ends when c == 0.
     hash = ((hash << 5) + hash) + c;                    // hash * 33 + c.
  }

  return (hash % tableSize);                             // compute index and exit.
} // end function.


//-----------------------------------------------------------------------------
// addNameHash: Given a string, hash it, and add to sNameHash[]. 
//-----------------------------------------------------------------------------

static unsigned int addNameHash(char *key, int idx) 
{
   unsigned int slot = stringHash(key, HASH_SIZE);       // compute hash code.
   if (sNameHash[slot].idx < 0) {                        // If not occupied,
      sNameHash[slot].idx = idx;                         // ..Now it is.
      return(slot);                                      // and we are done.
   }

   // slot was occupied (collision).
   _pcp_hash_t *newGuy = calloc(1, sizeof(_pcp_hash_t));                // make a new entry.
   newGuy->idx = sNameHash[slot].idx;                                   // copy the idx sitting in table.
   newGuy->next = sNameHash[slot].next;                                 // copy the chain pointer.
   sNameHash[slot].idx = idx;                                           // this one goes into table.
   sNameHash[slot].next = (void*) newGuy;                               // and chains to that new guy.
   return(slot);                                                        // and we are done.
} // end routine.


//-----------------------------------------------------------------------------
// freeNameHash: delete any allocated for collisions.
//-----------------------------------------------------------------------------

static void freeNameHash(void) 
{
   int i;
   for (i=0; i<HASH_SIZE; i++) {                                         // loop through table.
      void *next = sNameHash[i].next;                                    // Get any pointer.
      while (next != NULL) {                                             // follow the chain.
         void *tofree = next;                                            // remember what we have to free.
         next = ((_pcp_hash_t*) next)->next;                             // follow the chain.
         free(tofree);                                                   // free the one we are standing on.
      } 
   }
} // end routine.


//-----------------------------------------------------------------------------
// findNameHash: Returns the idx into pcp_event_info, or -1 if not found.
// avg over 1857 lookups, Saturn [Intel XEON 2.0GHz) was 120ns per lookup.
//-----------------------------------------------------------------------------

static int findNameHash(char *key) 
{
   int idx;
   unsigned int slot = stringHash(key, HASH_SIZE);                      // compute hash code.
   idx = sNameHash[slot].idx;                                           // get the index.
   if (idx < 0) return(-1);                                             // No good slot for it.
   if (strcmp(key, pcp_event_info[idx].name) == 0) {                    // If we found it straight away,
      return(idx);                                                      // .. quick return.
   }

   _pcp_hash_t *next = (_pcp_hash_t*) sNameHash[slot].next;             // get the next guy.

   while (next != NULL) {                                               // follow the chain.
      idx = next->idx;                                                  // .. get the idx.
      if (strcmp(key, pcp_event_info[idx].name) == 0) {                 // If we found a match,
         return(idx);                                                   // .. return with answer.
      }
      
      next = next->next;                                                // get the next guy in the link.
   } // end chain follow for collisions.

   return(-1);                                                          // did not find it.
} // end routine.


//-----------------------------------------------------------------------------
// Get all needed function pointers from the Dynamic Link Library. 
//-----------------------------------------------------------------------------

// Simplify routine below; relies on ptr names being same as func tags.
#define STRINGIFY(x) #x 
#define TOSTRING(x) STRINGIFY(x)
#define mGet_DL_FPtr(Name)                                                 \
   Name##_ptr = dlsym(dl1, TOSTRING(Name));                                \
   if (dlerror() != NULL) {  /* If we had an error, */                     \
      int strErr=snprintf(_pcp_vector.cmp_info.disabled_reason, PAPI_HUGE_STR_LEN,     \
         "PCP library function %s not found in lib.", TOSTRING(Name));     \
      _pcp_vector.cmp_info.disabled_reason[PAPI_HUGE_STR_LEN-1]=0;          \
      if (strErr > PAPI_HUGE_STR_LEN) HANDLE_STRING_ERROR;                  \
      return(PAPI_ENOSUPP);                                                \
   } /* end of macro. */

static int _local_linkDynamicLibraries(void) 
{
   if ( _dl_non_dynamic_init != NULL ) {  // If weak var present, statically linked insted of dynamic.
       char *strCpy=strncpy( _pcp_vector.cmp_info.disabled_reason, "The pcp component REQUIRES dynamic linking capabilities.", PAPI_HUGE_STR_LEN);
       _pcp_vector.cmp_info.disabled_reason[PAPI_HUGE_STR_LEN-1]=0;
       if (strCpy == NULL) HANDLE_STRING_ERROR;
       return PAPI_ENOSUPP;               // EXIT not supported.
   }

   char path_name[1024];
   char *pcp_root = getenv("PAPI_PCP_ROOT"); 
   
   dl1 = NULL;
   // Step 1: Process override if given.   
   if (strlen(pcp_main) > 0) {                                  // If override given, it has to work.
      dl1 = dlopen(pcp_main, RTLD_NOW | RTLD_GLOBAL);           // Try to open that path.
      if (dl1 == NULL) {
         int strErr=snprintf(_pcp_vector.cmp_info.disabled_reason, PAPI_HUGE_STR_LEN, "PAPI_PCP_MAIN override '%s' given in Rules.pcp not found.", pcp_main);
         _pcp_vector.cmp_info.disabled_reason[PAPI_HUGE_STR_LEN-1]=0;
         if (strErr > PAPI_HUGE_STR_LEN) HANDLE_STRING_ERROR;
         return(PAPI_ENOSUPP);   // Override given but not found.
      }
   }

   // Step 2: Try system paths, will work with Spack, LD_LIBRARY_PATH, default paths.
   if (dl1 == NULL) {                                           // No override,
      dl1 = dlopen("libpcp.so", RTLD_NOW | RTLD_GLOBAL);        // Try system paths.
   }

   // Step 3: Try the explicit install default. 
   if (dl1 == NULL && pcp_root != NULL) {                          // if root given, try it.
      int strErr=snprintf(path_name, 1024, "%s/lib64/libpcp.so", pcp_root);   // PAPI Root check.
      path_name[1024-1]=0;
      if (strErr > PAPI_MAX_STR_LEN) HANDLE_STRING_ERROR;
      dl1 = dlopen(path_name, RTLD_NOW | RTLD_GLOBAL);             // Try to open that path.
   }

   // Check for failure.
   if (dl1 == NULL) {
      int strErr=snprintf(_pcp_vector.cmp_info.disabled_reason, PAPI_HUGE_STR_LEN, "libpcp.so not found.");
      _pcp_vector.cmp_info.disabled_reason[PAPI_HUGE_STR_LEN-1]=0;
      if (strErr > PAPI_HUGE_STR_LEN) HANDLE_STRING_ERROR;
      return(PAPI_ENOSUPP);
   }

   // We have dl1. 

//-----------------------------------------------------------------------------
// Collect pointers for routines in shared library.  All below will abort this
// routine with PAPI_ENOSUPP, if the routine is not found in the dynamic
// library.
//-----------------------------------------------------------------------------

   mGet_DL_FPtr(pmLookupName);
   mGet_DL_FPtr(pmErrStr);
   mGet_DL_FPtr(pmTraversePMNS);
   mGet_DL_FPtr(pmFreeResult);
   mGet_DL_FPtr(pmNewContext);
   mGet_DL_FPtr(pmDestroyContext);
   mGet_DL_FPtr(pmFetch);
   mGet_DL_FPtr(pmLookupDesc);
   mGet_DL_FPtr(pmGetInDom);
   mGet_DL_FPtr(pmLookupText);
   mGet_DL_FPtr(pmUnitsStr_r);
   return PAPI_OK;   // If we get here, all above succeeded. 
} // end routine.


//-----------------------------------------------------------------------------
// qsort comparison routine, for pcp_event_info.
//-----------------------------------------------------------------------------
static int qsPMID(const void *arg1, const void* arg2) 
{
   _pcp_event_info_t *p1 = (_pcp_event_info_t*) arg1;
   _pcp_event_info_t *p2 = (_pcp_event_info_t*) arg2;

   if (p1->pmid < p2->pmid) return (-1);                                // 1 comes before 2.
   if (p1->pmid > p2->pmid) return ( 1);                                // 1 comes after 2.
   if (p1->idx  < p2->idx ) return (-1);                                // same pmid, try idx into vlist.
   if (p1->idx  > p2->idx ) return ( 1);                                // 1 comes after 2. 
   return (strcmp(p1->name, p2->name));                                 // sort by name if same PMID and idx.
} // end routine.


//-----------------------------------------------------------------------------
// cbPopulateNameOnly: This is a callback routine, called by pmTraversePMNS.  That
// routine iterates through the PM name space and calls this routine once per
// name. We increment sEventCount as we go, this will be the final count of valid
// array entries. sEventInfoSize will be >= sEventCount.
// WARNING: May realloc() pcp_event_info[], invalidating pointers into it.
//-----------------------------------------------------------------------------

static void cbPopulateNameOnly(const char *name) 
{
   if (sEventCount >= sEventInfoSize) {                                 // If we must realloc, 
      sEventInfoSize += sEventInfoBlock;                                // .. Add another page.
      pcp_event_info = realloc(pcp_event_info,                          // .. do realloc.
                        sEventInfoSize*sizeof(_pcp_event_info_t));      // ..
      memset(&pcp_event_info[sEventCount], 0,                           // .. clear to zeros.
             sEventInfoBlock*sizeof(_pcp_event_info_t));                // .. 
   }

   strncpy(pcp_event_info[sEventCount].name, name, PAPI_MAX_STR_LEN-1); // copy name.
   sEventCount++;                                                       // increment our count of events.
} // end routine.


//-----------------------------------------------------------------------------
// makeQualifiedEvent: Create a new event as a copy of the old, plus a tag at
// the end.  This appends to the pcp_event_info[] array, which may be realloced
// (which CAN invalidate any pointers into it).
//-----------------------------------------------------------------------------

static void makeQualifiedEvent(int baseEvent, int idx, char *qualifier) 
{
   int prevSize;
   if (sEventCount >= sEventInfoSize) {                                 // If we must realloc, 
      prevSize = sEventInfoSize;
      sEventInfoSize += sEventInfoBlock;                                // .. Add another block.
      pcp_event_info = realloc(pcp_event_info,                          // .. attempt reallocation.
                       sEventInfoSize*sizeof(_pcp_event_info_t));       // .. ..
      if (pcp_event_info == NULL) {                                     // If realloc failed, report it.
         fprintf(stderr, "%s:%i:%s realloc denied; "
                 "pcp_event_info=%p at size=%i.\n",
                 __FILE__, __LINE__, __func__, 
                pcp_event_info, sEventInfoSize);
         exit(-1);
      } // end if realloc failed.

      memset(&pcp_event_info[prevSize], 0,                              // .. clear the new block to zeros..
             sEventInfoBlock*sizeof(_pcp_event_info_t));                // .. 
   } // end if realloc needed.

   pcp_event_info[sEventCount] = pcp_event_info[baseEvent];             // copy the structure.
   pcp_event_info[sEventCount].numVal = 1;                              // Just one value.
   pcp_event_info[sEventCount].idx = idx;                               // Set the right index.
   pcp_event_info[sEventCount].zeroValue = 0;                           // Set the base value.  
   int slen = strlen(pcp_event_info[sEventCount].name);                 // get length of user name.
   char *c = qualifier;                                                 // point at qualifier.
   pcp_event_info[sEventCount].name[slen++] = ':';                      // place a colon.

   while ( (*c) != 0 && slen < PAPI_MAX_STR_LEN-1) {                    // .. appending qualifier,
      char v=*c++;                                                      // .. what we intend to append, point at next.
      // your chance to invalidate any chars, right here!
      pcp_event_info[sEventCount].name[slen++] = v;                     // .. add to name, inc slen.
   }
   
   pcp_event_info[sEventCount].name[slen] = 0;                          // ensure z-terminator. 
   sEventCount++;                                                       // increment our count of events.
} // end routine.


//-----------------------------------------------------------------------------
// Helper; reads the description if it has not already been read for a given
// index. Presumes pmid is already present.
//-----------------------------------------------------------------------------

static void getPMDesc(int pcpIdx) {                                     // Reads the variable descriptor.
   int ret;
   if (pcp_event_info[pcpIdx].pmid == PM_ID_NULL) return;               // Already have it.
   ret = pcp_pmLookupDesc(pcp_event_info[pcpIdx].pmid,                  // Get the event descriptor.
         &pcp_event_info[pcpIdx].desc);                                 // .. into the table; will set desc.pmid to not null.
   if (ret == PM_ERR_PMID) {                                            // If we failed for PMID,
      fprintf(stderr, "%s:%i:%s Invalid PMID.\n",
              __FILE__, __LINE__, __func__); 
      exit(-1);
   } // end if realloc failed.
      
   if (ret == PM_ERR_NOAGENT) {                                         // If we failed for agent,
      fprintf(stderr, "%s:%i:%s PMDA Agent not available to respond..\n",
              __FILE__, __LINE__, __func__); 
      exit(-1);
   } // end if realloc failed.

   if (ret != 0) {                                                      // Unknown error, 
      fprintf(stderr, "%s:%i:%s Unknown error code ret=%i.\n",
              __FILE__, __LINE__, __func__, ret); 
      exit(-1);
   } // end if realloc failed.

   pcp_event_info[pcpIdx].valType = pcp_event_info[pcpIdx].desc.type;   // Always copy type over.
   return;                                                              // No error. 
} // END code.


//-----------------------------------------------------------------------------
// We cache all domains we read; to save the round-trip cost to the daemon;
// on Saturn Test 8ms (too much delay invoked over thousands of events). 
// Use inst = -1 to free all malloced memory. 
// WARNING: realloc() cachedDomains[] can relocate the entire cache in memory,
// INVALIDATING ALL POINTERS into it, causing segfaults or anomolous results.
//-----------------------------------------------------------------------------

static char *cachedGetInDom(pmInDom indom, int inst) 
{
   static int domains=0;                                                // None cached to begin.
   static _pcp_domain_cache_t* cachedDomains = NULL;                    // Empty pointer.
   int i, domIdx;

   if (inst == -1) {                                                    // If we are shutting down,
      if (cachedDomains == NULL) return(NULL);                          // exit if we never built an array.
      for (i=0; i<domains; i++) {                                       // for every one cached,
         free(cachedDomains[i].instances);                              // .. free malloced memory.
         free(cachedDomains[i].names);                                  // .. free malloced memory.
      }
      
      free(cachedDomains);                                              // discard our table.
      domains = 0;                                                      // reset.
      cachedDomains = NULL;                                             // never free twice.
      return(NULL);                                                     // exit.
   } // end if shutting down.

   // Check if we have it already.
   for (i=0; i<domains; i++) {
      if (indom == cachedDomains[i].domainId) break;                    // Exit loop if found.
   } 

   domIdx = i;                                                          // The domain index.
   if (i == domains) {                                                  // If not found; make a new one and read it.
      domains++;                                                        // .. add one to count.

      if (domains == 1) {                                               // for first domain,
         cachedDomains = malloc(sizeof(_pcp_domain_cache_t));           // ..malloc.
      } else {                                                          // for subsequent domains,
         cachedDomains = realloc(cachedDomains, 
                        domains*sizeof(_pcp_domain_cache_t));           // realloc, retain first.
      }

      if (cachedDomains == NULL) {                                      // If we failed malloc/realloc, 
         fprintf(stderr, "%s:%i:%s malloc/realloc denied for "
                 "cachedDomains; size=%i.\n", 
                 __FILE__, __LINE__, __func__, domains);
         exit(-1);
      } // end if realloc failed.
         
      cachedDomains[domIdx].domainId = indom;                           // .. The domain we are getting.
      cachedDomains[domIdx].numInstances = pcp_pmGetInDom(indom, 
         &cachedDomains[domIdx].instances,                              // .. store pointer lists in struct too.
         &cachedDomains[domIdx].names);                                 // .. 
      for (i=0; i<cachedDomains[domIdx].numInstances; i++) {            // DEBUG, vet the strings.
         if (cachedDomains[domIdx].names[i] == NULL || 
             strlen(cachedDomains[domIdx].names[i]) == 0 || 
             strlen(cachedDomains[domIdx].names[i]) >= PAPI_MAX_STR_LEN) {
            fprintf(stderr, "%s:%i:%s ERROR: cachedGetInDom: domain=%u, domIdx=%i, name idx %i invalid string.\n",
               __FILE__, __LINE__, FUNC, indom, domIdx, i);
            exit(-1);
         } // end if domain string is nonsense.
      }           
   } // end if we need to cache a new domain.
      
   // We got the domain index, Now we can try to look up the 
   // instance name.

   for (i=0; i < cachedDomains[domIdx].numInstances; i++) {             // look through all instances. 
      if (cachedDomains[domIdx].instances[i] == inst)                   // .. If found,
        return cachedDomains[domIdx].names[i];                          // .. .. return matching name.
   } // end search for inst.

   fprintf(stderr, "%s:%i:%s ERROR: cachedGetInDom: domain=%u, domIdx=%i, numInstances=%i, failed to find inst=%i.\n",
      __FILE__, __LINE__, FUNC, indom, domIdx, cachedDomains[domIdx].numInstances, inst);
   exit(-1);                                                            // Cannot continue; should not have happened.

   return NULL;                                                         // Code cannot be reached. Couldn't find it.
} // end routine.


//-----------------------------------------------------------------------------
// Helper routine, returns a ull value from a value set pointer. Automatically
// does conversions from 32 bit to 64 bit (int32, uint32, fp32).  
//-----------------------------------------------------------------------------
static unsigned long long getULLValue(pmValueSet *vset, int value_index) 
{
   unsigned long long value;                                         // our return value.
   convert_64_t convert;                                             // union for conversion.
   uPointer_t myPtr;                                                 // a union helper to avoid warnings.

   if (vset->valfmt == PM_VAL_INSITU) {                              // If the value format is in situ; a 32 bit value.
      convert.ll = vset->vlist[value_index].value.lval;              // .. we can just collect the value immediately.
      value = convert.ull;                                           // .. 
   } else {                                                          // If it is either static or dynamic alloc table,
      pmValueBlock *pmvb = vset->vlist[value_index].value.pval;      // .. value given is a pointer to value block.
      myPtr.cPtr = pmvb->vbuf;                                       // .. use cPtr because vbuf defined as char[1] in pmValueBlock.
      switch (pmvb->vtype) {                                         // Note we restricted the types in init; these cases should agree.
         case  PM_TYPE_32:       // 32-bit signed integer
            convert.ll = myPtr.iPtr[0];
            value = convert.ull;
            break;

         case  PM_TYPE_U32:      // 32-bit unsigned integer
            value =  myPtr.uiPtr[0];
            break;

         case  PM_TYPE_64:       // 64-bit signed integer
            convert.ll = myPtr.llPtr[0];
            value = convert.ull;
            break;

         case  PM_TYPE_U64:      // 64-bit unsigned integer
            value = myPtr.ullPtr[0];
            break;

         case  PM_TYPE_FLOAT:    // 32-bit floating point
            convert.d = myPtr.fPtr[0];  // convert first.
            value = convert.ull;
            break;

         case  PM_TYPE_DOUBLE:   // 64-bit floating point
            convert.d = myPtr.dPtr[0];
            value = convert.ull;
            break;

         case  PM_TYPE_STRING:   // array of char 
            convert.vp = myPtr.cPtr;
            value = convert.ull;            
            break;

         default:
            fprintf(stderr, "%s:%i pmValueBlock (from PCP) contains an unrecognized value type=%i.\n",
               __FILE__, __LINE__,  pmvb->vtype);
            convert.ll = -1;                                            // A flag besides zero
            value = convert.ull;
      } // end switch on type. 
   } // if pmValueBlock value.

   return(value);                                                       // exit with result.
} // end routine. 


//----------------------------------------------------------------------------
// Helper routine to subtract zero value from an event value. Although we
// store an unsigned long long value for both our zero value and a read-in
// value, it can be recast to signed or double. So zeroing on those must be
// done in the type intended, and we have to recast to do it. Note that 32
// bit signed int or float were saved and cast as 64 bit int or double,
// respectively.
//
// There are three types of 'semantics' in a description; only one is a 
// counter; the other two are instantaneous values. We do not subtract a
// zero reference value from instantaneous values. 
// PM_SEM_COUNTER    // cumulative counter (monotonic increasing)
// PM_SEM_INSTANT    // instantaneous value, continuous domain
// PM_SEM_DISCRETE   // instantaneous value, discrete domain
//----------------------------------------------------------------------------

static void subZero(_pcp_control_state_t *myCtl, int event)
{
   int k = myCtl->pcpIndex[event];                                      // get pcp_event_info[] index.
   if (pcp_event_info[k].desc.sem != PM_SEM_COUNTER) return;            // Don't subtract from instantaneous values.

   convert_64_t zero, rawval;
   rawval.ull = myCtl->pcpValue[event];                                 // collect the raw value.
   zero.ull = pcp_event_info[k].zeroValue;                              // collect the zero (base) value.
   switch (pcp_event_info[k].valType) {                                 // Note we restricted the types in init; these cases should agree.
      case  PM_TYPE_32:                                                 // 32 bit was converted to long long.
      case  PM_TYPE_64:                                                 // long long.
         rawval.ll -= zero.ll;                                          // converted.
         break;

      case  PM_TYPE_U32:                                                // 32-bit was converted to 64 bit.
      case  PM_TYPE_U64:                                                // 64-bit unsigned integer
         rawval.ull -= zero.ull;                                        // converted.
         break;

      case  PM_TYPE_FLOAT:                                              // 32-bit was converted to double.
      case  PM_TYPE_DOUBLE:                                             // 64-bit floating point
         rawval.d -= zero.d;                                            // converted.
         break;

      case  PM_TYPE_STRING:                                             // array of char, do nothing for pointer.
         break;

      default:
         fprintf(stderr, "%s:%i pcp_event_info[%s] contains an unrecognized value type=%i.\n",
            __FILE__, __LINE__,  pcp_event_info[k].name, pcp_event_info[k].valType);
         exit(-1);                                                      // Quit, this shouldn't happen; something needs updating.
         break;
   } // end switch on type. 
   
   myCtl->pcpValue[event] = rawval.ull;                                 // The adjusted value.
} // end routine.


//-----------------------------------------------------------------------------
// Helper routine for _pcp_ntv_code_to_descr to retrieve pm event text (a
// description of the event) given a pcp_event_info[] index.
// There are two options, a PM_TEXT_ONELINE is a single line description and
// PM_TEXT_HELP is a longer help description. The PAPI code that calls this 
// stores the description in PAPI_event_info_t->long_descr[PAPI_HUGE_STR_LEN];
// currently 1024 characters. So we use the PM_TEXT_HELP version.
//
// pcpIdx MUST BE pre-validated as in range [0, sEventCount-1].
// *helpText may be allocated; caller must free(helpText).
// RETURNS PAPI_OK or PAPI error.
// NOTE: There is also a pmLookupInDomText() that returns a description of a
// domain; if you want that, you need a pmInDom and a very similar routine. 
//-----------------------------------------------------------------------------
static int getHelpText(unsigned int pcpIdx, char **helpText) 
{
   char    *p;
   int     ret;
   char    errMsg[]="Help Text is not available for this event.";       // for an error we have seen.

   pmID myPMID = pcp_event_info[pcpIdx].pmid;                           // collect the pmid.
   ret = pcp_pmLookupText(myPMID, PM_TEXT_HELP, helpText);              // collect a line of text, routine mallocs helpText.
   if (ret != 0) {                                                      // If larger help is not available, Try oneline.
      if (*helpText != NULL) free(*helpText);                           // .. Free anything allocated. 
      *helpText = NULL;                                                 // .. start it as null.
      ret = pcp_pmLookupText(myPMID, PM_TEXT_ONELINE, helpText);        // .. collect a line of text, routine mallocs helpText.
   } 

   if (ret == PM_ERR_TEXT) {                                            // If not available, 
      *helpText = strdup(errMsg);                                       // duplicate this error message.
   } else if (ret != 0) {                                               // If PCP has any other error, report and exit.
      fprintf(stderr, "%s:%i:%s pmLookupText failed; return=%s.\n", 
         __FILE__, __LINE__, FUNC, pcp_pmErrStr(ret));
      return PAPI_EATTR;                                                // .. invalid or missing event attribute.
   }

   // Replace all /n with '|'.
   for (p=(*helpText); p[0] != 0; p++) {                                // loop through string routine allocated,
      if (p[0] == '\n') p[0] = '|';                                     // .. If we found a \n, replace with '|'.
   } // end scan for \n.

   return PAPI_OK;                                                      // Presumably all went well.    
} // end routine.


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// PAPI FUNCTIONS. 
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// _pcp_init_component is called when PAPI is initialized (during 
// PAPI_library_init). The argument is the component index. 
//---------------------------------------------------------------------------
static int _pcp_init_component(int cidx) 
{
    int retval = PAPI_OK;
   char *reason = _papi_hwd[cidx]->cmp_info.disabled_reason;            // For error messages.
   int rLen = PAPI_HUGE_STR_LEN;                                        // Most I will print.
   reason[rLen-1]=0;                                                    // last resort terminator.

   mRtnCnt(_pcp_init_component);                                        // count the routine.
   #define hostnameLen 512 /* constant used multiple times. */
   char hostname[hostnameLen];                                          // host name.
   int  ret;

   ret = _local_linkDynamicLibraries();
   if ( ret != PAPI_OK ) {                                              // Failure to get lib.
       retval = PAPI_ESYS;
       goto fn_fail;
   }

   ret = gethostname(hostname, hostnameLen);                            // Try to get the host hame.
   if (ret != 0) {                                                      // If we can't get the hostname, 
      int strErr=snprintf(reason, rLen, "Failed system call, gethostname() "
            "returned %i.", ret);   
      reason[rLen-1]=0;
      if (strErr > rLen) HANDLE_STRING_ERROR;
      retval = PAPI_ESYS;
      goto fn_fail;
   }
   #undef hostnameLen /* done with it. */

   _prog_fprintf(stderr, "%s:%i retrieved hostname='%s'\n", __FILE__, __LINE__, hostname); // show progress.

   ctxHandle = pcp_pmNewContext(PM_CONTEXT_HOST, "local:");             // Set the new context to local host.
   if (ctxHandle < 0) {
      int strErr=snprintf(reason, rLen, "Cannot connect to PM Daemon on \"%s\".\n "
         "(Is the Performance Co-Pilot running?)\n", hostname);
      reason[rLen-1]=0;
      if (strErr > rLen) SUBDBG("%s:%i Warning! Error's 'reason' string truncated:\n%s\n",__FILE__,__LINE__,reason);
      retval = PAPI_ESYS;
      goto fn_fail;
   }

   _prog_fprintf(stderr, "%s:%i Found ctxHandle=%i\n", __FILE__, __LINE__, ctxHandle); // show progress.

   sEventInfoSize = sEventInfoBlock;                                    // first allocation.   
   pcp_event_info = (_pcp_event_info_t*) 
      calloc(sEventInfoSize, sizeof(_pcp_event_info_t));                // Make room for all events.
   if (pcp_event_info == NULL) {
      int strErr=snprintf(_pcp_vector.cmp_info.disabled_reason, PAPI_HUGE_STR_LEN,
          "Could not allocate %lu bytes of memory for pcp_event_info.", sEventInfoSize*sizeof(_pcp_event_info_t));
      _pcp_vector.cmp_info.disabled_reason[PAPI_HUGE_STR_LEN-1]=0;    // force null termination.
      if (strErr > PAPI_HUGE_STR_LEN) HANDLE_STRING_ERROR;
      retval = PAPI_ENOMEM;
      goto fn_fail;
   }

   sEventCount = 0;                                                     // begin at zero.
   _time_gettimeofday(&t1, NULL);
   ret = pcp_pmTraversePMNS(AGENT_NAME, cbPopulateNameOnly);            // Timed on Saturn [Intel Xeon 2.0GHz]; typical 9ms, range 8.5-10.5ms.
   if (ret < 0) {                                                       // Failure...
      int strErr=snprintf(reason, rLen, "pmTraversePMNS failed; ret=%i [%s]\n", 
         ret, pcp_pmErrStr(ret));
      if (ret == PM_ERR_NAME) {                                         // We know what this one is,
         strErr=snprintf(reason, rLen, "pmTraversePMNS ret=PM_ERR_NAME: "
            "Occurs if event filter '%s' unknown to PCP Daemon.\n", AGENT_NAME);
      }
      reason[rLen-1]=0;
      if (strErr >rLen) HANDLE_STRING_ERROR;
      retval = PAPI_ENOIMPL;
      goto fn_fail;
   }      
      
   _time_gettimeofday(&t2, NULL);
   _time_fprintf(stderr, "pmTraversePMNS PopulateNameOnly took %li uS.\n", 
      (mConvertUsec(t2)-mConvertUsec(t1)));
   _time_fprintf(stderr, "Final sEventCount=%i, sEventInfoSize=%i, "
               "sEventInfoBlock=%i.\n", 
      sEventCount, sEventInfoSize, sEventInfoBlock);
  
   if (sEventCount < 1) {                                               // Failure...
      int strErr=snprintf(reason, rLen, "pmTraversePMNS returned zero events "
         "for AGENT=\"%s\".\n", AGENT_NAME);
      reason[rLen-1]=0;
      if (strErr >rLen) HANDLE_STRING_ERROR;
      retval = PAPI_ENOIMPL;
      goto fn_fail;
   }

   int i,j,k;
   char **allNames=calloc(sEventCount, sizeof(char*));                  // Make an array for all names. 
   for (i=0; i<sEventCount; i++) {                                      // .. 
      allNames[i] = pcp_event_info[i].name;                             // copy pointer into array. 
   } // end for each event.

   pmID *allPMID=calloc(sEventCount, sizeof(pmID));                     // Make an array for results.
   if (allPMID == NULL) {                                               // If we failed,
      snprintf(reason, rLen, "memory alloc denied for allPMID; "
            "size=%i.\n", sEventCount);
      int strErr=snprintf(reason, rLen, "Could not allocate %lu bytes of memory for allPMID.", sEventCount*sizeof(pmID));
      free(allNames);
      if (strErr > rLen) HANDLE_STRING_ERROR;
      retval = PAPI_ENOMEM;
      goto fn_fail;
   } // end if calloc failed.

   //----------------------------------------------------------------
   // Unlike Saturn, on the Power9 we get an 'IPC protocol failure' 
   // if we try to read more than 946 names at a time. This is some
   // limitation on a communication packet size. On our test system
   // Power9; the maximum number we can read is 946. To allow leeway
   // for other possible values; we read in blocks of 256.
   //----------------------------------------------------------------
   #define LNBLOCK 256                                                  /* Power9 gets IPC errors if read block is too large. */
   k = (__LINE__)-1;                                                    // where LNBLOCK is defined.   

   _time_gettimeofday(&t1, NULL);

   i=0;                                                                 // Starting index for allPMID.
   while (i<sEventCount) {                                              // read in blocks of LNBLOCK.
      j = sEventCount-i;                                                // .. presume we can read the rest.
      if (j > LNBLOCK) j=LNBLOCK;                                       // .. reduce if we cannot.
      ret = pcp_pmLookupName(j, allNames+i, allPMID+i);                 // .. Get a block of PMIDs for a block of names.
      if (ret < 0) {                                                    // .. Failure...
         int strErr=snprintf(reason, rLen, "pmLookupName for %i names failed; ret=%i [%s].\n", 
            sEventCount, ret, pcp_pmErrStr(ret));
         reason[rLen-1]=0;
         if (strErr > rLen) HANDLE_STRING_ERROR;

         if (ret == PM_ERR_IPC) {                                       // .. If we know it, rewrite.
            strErr=snprintf(reason, rLen, "pmLookupName ret=PM_ERR_IPC: one known cause is a readblock too large; reduce LNBLOCK (%s:%i).\n",
                  __FILE__,k);
            reason[rLen-1]=0;
            if (strErr > rLen) HANDLE_STRING_ERROR;
            retval = PAPI_EBUF;
            goto fn_fail;
         }

         retval = PAPI_ESYS;
         goto fn_fail;
      }

      i+=j;                                                             // .. Adjust the pointer forward by what we read.
   } // end while to read names in chunks, and avoid IPC error. 
   #undef LNBLOCK                                                       /* Discard constant; no further use. */  

   _time_gettimeofday(&t2, NULL);
   _time_fprintf(stderr, "pmLookupName for all took %li uS, ret=%i.\n", 
      (mConvertUsec(t2)-mConvertUsec(t1)), ret );

   for (i=0; i<sEventCount; i++) pcp_event_info[i].pmid = allPMID[i];   // copy all the pmid over to array.

   pmResult *allFetch = NULL;                                           // result of pmFetch. 
   _time_gettimeofday(&t1, NULL);
   ret = pcp_pmFetch(sEventCount, allPMID, &allFetch);                  // Fetch (read) all the events.
   _time_gettimeofday(&t2, NULL);
   if (ret != 0) {
      int strErr=snprintf(reason, rLen, "pcp_pmFetch failed, retcode=%d.", ret);
      reason[rLen-1]=0;
      if (strErr > rLen) HANDLE_STRING_ERROR;
      retval = PAPI_ENOSUPP;
      goto fn_fail;
   }

   _time_fprintf(stderr, "pmFetch for all took %li uS, for %i events; ret=%i.\n", 
      (mConvertUsec(t2)-mConvertUsec(t1)), sEventCount, ret);

   //-------------------------------------------------------------------
   // In processing fetches, if we find a multi-valued event, we need
   // to force an event for every value (PAPI only returns 1 value per
   // event; not an array). In experiments thus far, all multi-valued
   // events have had domain descriptor names; so we just concat with
   // the name and make a new Event. We use a helper for that,
   // afterward we set the number of values in the original 'base name'
   // to zero; so it will be deleted by the cleanup routine.                                        
   //-------------------------------------------------------------------

   _time_gettimeofday(&t1, NULL);                                       // time this index explosion.
   int origEventCount = sEventCount;                                    // sEventCount may change in below routine.

   for (i=0; i<origEventCount; i++) {                                   // process all the fetch results.
      pcp_event_info[i].desc.pmid = PM_ID_NULL;                         // This indicates the description is NOT loaded yet.
      pmValueSet *vset = allFetch->vset[i];                             // get the result for event[i].

      // On Saturn test system, never saw this happen.
      if (vset == NULL) {                                               // .. should not happen. leave numVal=0 for deletion.
         fprintf(stderr, "%s:%i vset=NULL for name='%s'\n", 
            __FILE__, __LINE__, pcp_event_info[i].name);
         continue;                                                      // .. next in loop.
      }
     
      pcp_event_info[i].numVal = vset->numval;                          // Show we have a value.
      if (vset->numval == 0) {                                          // If the value is zero, 
//       _prog_fprintf(stderr, "%s:%i Discarding, no values for event  '%s'.\n", __FILE__, __LINE__, pcp_event_info[i].name);
         continue;                                  // If no values, leave numVal = 0 for deletion. (We do see this in tests). 
      }

      pcp_event_info[i].valfmt = vset->valfmt;                          // Get the value format. (INSITU or not).
      getPMDesc(i);                                                     // Get the value descriptor.
      unsigned long long ullval= (long long) -1;   // debug stuff.
      convert_64_t convert;

      if (vset->valfmt != PM_VAL_INSITU) {                              // If not in situ, must get the type.
         pmValue *pmval = &vset->vlist[0];                              // .. Get the first value.
         pmValueBlock *pB = pmval->value.pval;                          // .. get it.
         if (pcp_event_info[i].valType != pB->vtype) {
            int strErr=snprintf(reason, rLen, "Unexpected value type fetched for %s. %i vs %i. Possible version incompatibiity.\n", 
               pcp_event_info[i].name, pcp_event_info[i].valType, pB->vtype);
            reason[rLen-1]=0;
            if( strErr > rLen) HANDLE_STRING_ERROR;
            retval = PAPI_ENOSUPP;
            goto fn_fail;
         }

//       pcp_event_info[i].valType = pB->vtype;                         // .. get the type.
         ullval = getULLValue(vset, 0);                                 // .. get the first value.

         switch(pB->vtype) {                                            // PCP's variable type; an int flag.
            case  PM_TYPE_32:                                           // 32-bit signed integer
               _prog_fprintf(stderr, "type I32, desc.sem=%i, event '%s'=", pcp_event_info[i].desc.sem, pcp_event_info[i].name); 
               break;
            case  PM_TYPE_U32:                                          // 32-bit unsigned integer
               _prog_fprintf(stderr, "type U32, desc.sem=%i, event '%s'=", pcp_event_info[i].desc.sem, pcp_event_info[i].name); 
               break;
            case  PM_TYPE_FLOAT:                                        // 32-bit floating point
               _prog_fprintf(stderr, "type F32, desc.sem=%i, event '%s'=", pcp_event_info[i].desc.sem, pcp_event_info[i].name); 
               break;                                                   // END CASE.

            case  PM_TYPE_64:                                           // 64-bit signed integer
               convert.ull = ullval;
               _prog_fprintf(stderr, "type I64, desc.sem=%i, event '%s'= (ll) %lli =", pcp_event_info[i].desc.sem, pcp_event_info[i].name, convert.ll); 
               break;
            case  PM_TYPE_U64:                                          // 64-bit unsigned integer
               _prog_fprintf(stderr, "type U64, desc.sem=%i, event '%s'= (ull) %llu =", pcp_event_info[i].desc.sem, pcp_event_info[i].name, convert.ull); 
               break;
            case  PM_TYPE_DOUBLE:                                       // 64-bit floating point
               convert.ull = ullval;
               _prog_fprintf(stderr, "type U64, desc.sem=%i, event '%s'= (double) %f =", pcp_event_info[i].desc.sem, pcp_event_info[i].name, convert.d); 
               break;                                                   // END CASE.

            // IF YOU want to return string values, this is a place
            // to change; currently all string-valued events are
            // rejected. But, pB->vbuf is the string value. I would
            // copy it into a new pcp_event_info[] field; it would
            // need to be malloc'd here and free'd at component
            // shutdown. Also PAPI would need a new way to accept a
            // char* or void*. 

            case  PM_TYPE_STRING:                                       // pB->vbuf is char* to string value.
               _prog_fprintf(stderr, "%s:%i Discarding PM_TYPE_STRING event, desc.sem=%i, event '%s'=", __FILE__, __LINE__, pcp_event_info[i].desc.sem, pcp_event_info[i].name); 
               pcp_event_info[i].numVal = 0;                            // .. .. set numVal = 0 for deletion.
               break;

            default:                                                    // If we don't recognize the type,
               _prog_fprintf(stderr, "%s:%i Dsicarding PM_UNKNOWN_TYPE event, desc.sem=%i, event '%s'=", __FILE__, __LINE__, pcp_event_info[i].desc.sem, pcp_event_info[i].name); 
               pcp_event_info[i].numVal = 0;                            // .. set numVal = 0 for deletion.
               break;
         } // end switch.
      } // If not In Situ.
      else {
         _prog_fprintf(stderr, "type IST, desc.sem=%i, event '%s'=", pcp_event_info[i].desc.sem, pcp_event_info[i].name); 
      }

      convert.ull = ullval; 
      _prog_fprintf(stderr, "%02X%02X%02X%02X %02X%02X%02X%02X\n", convert.ch[0], convert.ch[1],  convert.ch[2], convert.ch[3],  convert.ch[4], convert.ch[5],  convert.ch[6], convert.ch[7]);
      // Lookup description takes time; so we only do it for
      // multi-valued events here. For other events, we will do it
      // as needed for EventInfo filling.

      if (pcp_event_info[i].numVal > 1) {                               // If a domain qualifier is possible;
         getPMDesc(i);                                                  // .. Get the event descriptor.
         _prog_fprintf(stderr, "Event %s has %i values, indom=%i.\n", pcp_event_info[i].name, pcp_event_info[i].numVal, pcp_event_info[i].desc.indom); 
         if (pcp_event_info[i].desc.indom != PM_INDOM_NULL) {           // .. If we have a non-null domain,
            for (j=0; j<vset->numval; j++) {                            // .. for every value present,
               pmValue *pmval = &vset->vlist[j];                        // .. .. get that guy.

               char *dname = cachedGetInDom(                            // .. .. read from cached domains (and populate it when needed).
                                 pcp_event_info[i].desc.indom,
                                 pmval->inst);                          // .. .. get the name. Not malloced so don't free dName.

               makeQualifiedEvent(i, j, dname);                         // .. .. helper routine; may realloc pcp_event_info[], change sEventCount.
            } // end value list.
            
            pcp_event_info[i].numVal = 0;                               // .. let the base event be discarded.
         } // end if we have a domain.                                  
      } // end if multiple valued.                                      
   } // end for each event.

   // Trim the fat! We get rid of everything with numVal == 0.
   // We do that by compaction; moving valid entries to backfill
   // invalid ones.

   j=0;                                                                 // first destination.                                                  
   for (i=0; i<sEventCount; i++) {                                      // loop thorugh all old and new.
      if (pcp_event_info[i].numVal > 0) {                               // If we have a valid entry,
         if (i != j) pcp_event_info[j] = pcp_event_info[i];             // move if it isn't already there.
         j++;                                                           // count one moved; new count of valid ones.
      }
   } 
            
   sEventCount = j;                                                     // this is our new count.
   pcp_event_info = realloc(pcp_event_info,                             // release any extra memory. 
                        sEventCount*sizeof(_pcp_event_info_t));         // .. 
   if (pcp_event_info == NULL) {
      int strErr=snprintf(reason, rLen, 
          "Could not reallocate %lu bytes of memory for pcp_event_info.", sEventCount*sizeof(_pcp_event_info_t));
      reason[rLen-1]=0;    // force null termination.
      if (strErr > rLen) HANDLE_STRING_ERROR;    
      retval = PAPI_ENOMEM;
      goto fn_fail;
   } // end if realloc failed.

   qsort(pcp_event_info, sEventCount,                                   // sort by PMID, idx, name.
         sizeof(_pcp_event_info_t), qsPMID);                            // ..

   _time_gettimeofday(&t2, NULL);                                       // done with index explosion.
   _time_fprintf(stderr, "indexedExplosion for all took %li uS.\n", 
               (mConvertUsec(t2)-mConvertUsec(t1)) );

   for (i=0; i<HASH_SIZE; i++) {                                        // init hash table.
      sNameHash[i].idx = -1;                                            // unused entry. 
      sNameHash[i].next = NULL;                                         // ..
   }                                                                  
                                                                   
   unsigned int myHash;                              
   for (i=0; i<sEventCount; i++) {                                    
      myHash = addNameHash(pcp_event_info[i].name, i);                  // Point this hash to this entry.   
   }                                                                  

   //-----------------------------------------------------------------------------------------------------------------------
   // *************************************** DEBUG REPORT OF INFORMATION DISCOVERED ***************************************
   //-----------------------------------------------------------------------------------------------------------------------
   // We use -O2, but even in -O0, if(0){...} is completely removed from code. It costs us nothing to leave this code in.
   if (0) {                                                             // change to 1 to enable debug report. 
      unsigned int current, prev=0;                              
      printf("count, hash, name, pmid, value[idx]\n");
      for (i=0; i<sEventCount; i++) {
         myHash = stringHash(pcp_event_info[i].name, HASH_SIZE);        // Get the hash value.
         current = pcp_event_info[i].pmid;
         if (prev > 0 && current != (prev+1) && current != prev)        // print separators.
            printf("----,----,----,----\n");     
         printf("%i, %u, \"%s\", 0x%08X, %i\n", i, myHash,              // quote name, may contain \,/,comma, etc.
                pcp_event_info[i].name, 
                pcp_event_info[i].pmid, 
                pcp_event_info[i].idx);
         prev=current;                                                  // for finding pmid skips.
      } 

      // Test hashing.
      int hashErr = 0;
      _time_gettimeofday(&t1, NULL);
      for (i=0; i<sEventCount; i++) {
         int f = findNameHash(pcp_event_info[i].name);                  // Try to find this name. 
         if (f != i) hashErr++;
      }

      _time_gettimeofday(&t2, NULL);
      
      _time_fprintf(stderr, "HashLookup avg uS: %3.6f\n", ((double) (mConvertUsec(t2)-mConvertUsec(t1)) )/((double) sEventCount) );
      _time_fprintf(stderr, "Hash Errors: %i of %i.\n", hashErr, sEventCount);

   } // END DEBUG REPORT.
   //-----------------------------------------------------------------------------------------------------------------------
   // *************************************** END DEBUG REPORT INFORMATION DISCOVERED **************************************
   //-----------------------------------------------------------------------------------------------------------------------

   free(allNames);                                                      // Locals allocations not needed anymore.
   free(allPMID);                                                       // .. the pmIDs we read.
   pcp_pmFreeResult(allFetch);                                          // .. release the results we fetched.

//  For PCP, we can read any number of events at once
//  in a single event set. Set vector elements for PAPI.

   _pcp_vector.cmp_info.num_native_events = sEventCount;                // Setup our pcp vector.
   _pcp_vector.cmp_info.num_cntrs = sEventCount;                  
   _pcp_vector.cmp_info.num_mpx_cntrs = sEventCount;
   _pcp_vector.cmp_info.CmpIdx = cidx;                                  // export the component ID.

  fn_exit:
    _papi_hwd[cidx]->cmp_info.disabled = retval;
    return retval;
  fn_fail:
    goto fn_exit;
} // end routine.


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// Control of counters (Reading/Writing/Starting/Stopping/Setup) functions.
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Init vars in pcp_context.
// This is called immediately after _pcp_init_component.  
//----------------------------------------------------------------------------
static int _pcp_init_thread(hwd_context_t * ctx) 
{
   mRtnCnt(_pcp_init_thread);                                           // count this function.
   _pcp_context_t* myCtx = (_pcp_context_t*) ctx;                       // recast.
   myCtx->initDone = 1;                                                 // Nothing else to do for init.
   return PAPI_OK;
}  // end routine.


//----------------------------------------------------------------------------
// The control_state is our internal description of an event set. 
//----------------------------------------------------------------------------
static int _pcp_init_control_state( hwd_control_state_t *ctl)
{
   mRtnCnt(_pcp_init_control_state);                                    // count this function.
    _pcp_control_state_t* control = ( _pcp_control_state_t* ) ctl;
   // contents: _pcp_control_state state at this writing: 
   // contents of state: 
   // int numEvents;                                                    // The number of events we have.
   // int maxAllocated;                                                 // the current entries in pcpIndex.
   // int *pcpIndex;                                                    // array of indices into pcp_event_info[].
   // long long *values;                                                // Values read from our PCP events.

    memset(control, 0, sizeof(_pcp_control_state_t));                   // zero it.

    return PAPI_OK;
} // end routine.


//---------------------------------------------------------------------------------------------
// update_control_state: PAPI calls this when it adds or removes events
// from its EventState; so we can do our component-specific things. The
// hwd_control_state_t* is a void pointer, we recast to
// _pcp_control_state_t* to get whatever is CURRENTLY in there. We only
// need to update that, for now we just re-init it. HOWEVER, we may be
// discarding counters we set to counting. 
// This is called.  

// NOTE: This code allocates pcpIndex[] in the control and never frees
// it.  However, the PAPI code in destroying the eventSet calls this
// with a zero count; so we free() it then, without reallocating. Also,
// the values[] array

// NOTE: Also, PAPI *may* call more than once with a zero count on the
// control. If you free pcpIndex, set it to NULL, so you don't try to
// free it again.

// Content of NativeInfo_t:
// int ni_event;                // native (libpfm4) event code; always non-zero unless empty
// int ni_papi_code;            // papi event code value returned to papi applications
// int ni_position;             // counter array position where this native event lives
// int ni_owners;               // specifies how many owners share this native event
// hwd_register_t *ni_bits;     // Component defined resources used by this native event
//---------------------------------------------------------------------------------------------

static int _pcp_update_control_state( 
        hwd_control_state_t *ctl,                                       // Our internal info about events in an event set.
        NativeInfo_t *native,                                           // NativeInfoArray in papi. 
        int count,                                                      // NativeCount in papi.
        hwd_context_t *ctx)                                             // context, we don't use it.
{
   mRtnCnt(_pcp_update_control_state);                                  // count this function.
   int i, index = 0;
   ( void ) ctx;

   _pcp_control_state_t* MyCtl = ( _pcp_control_state_t* ) ctl;         // Recast ctl.

   MyCtl->numEvents = count;                                            // remember how many there are.
   if (count == 0) {                                                    // If we are deleting a set,
      if (MyCtl->pcpIndex != NULL) {                                    // If we have space allocated,
         free(MyCtl->pcpIndex);                                         // .. discard it,
         free(MyCtl->pcpValue);                                         // .. and values.
         MyCtl->pcpIndex = NULL;                                        // .. never free it again.
         MyCtl->pcpValue = NULL;                                        // .. never free it again.
      }

      MyCtl->maxAllocated = 0;                                          // .. no longer tracking max.
      return PAPI_OK;                                                   // .. get out.
   }

   // henceforth, count != 0.
   #define BlockSize 64 /* constant used multiple times. */
   int newalloc = ((count/BlockSize)*BlockSize+BlockSize);              // .. pick next up multiple of BlockSize.
   #undef  BlockSize    /*prevent bugs if same name used elsewhere.*/

   if (MyCtl->pcpIndex != NULL) {                                       // If I have a previous list of variables,
      if (count > MyCtl->maxAllocated) {                                // .. If I will need more room,
         MyCtl->pcpIndex = realloc(MyCtl->pcpIndex,                     // .. .. reallocate to make more room.
                                   newalloc*sizeof(int));               // .. .. ..
         MyCtl->pcpValue = realloc(MyCtl->pcpValue,                     // .. .. reallocate to make more room.
                                   newalloc*sizeof(unsigned long long));// .. .. ..
         MyCtl->maxAllocated = newalloc;                                // .. .. remember what we've got.
      }
   } else {                                                             // If NULL then I have no previous set,
      MyCtl->maxAllocated = newalloc;                                   // .. pick next up multiple of BlockSize.
      MyCtl->pcpIndex =                                                 // .. make room for 'count' indices,
         calloc(MyCtl->maxAllocated, sizeof(int));                      // .. 
      MyCtl->pcpValue =                                                 // .. make room for 'count' values.
         calloc(MyCtl->maxAllocated, sizeof(unsigned long long));       // .. 
   }

   if (MyCtl->pcpIndex == NULL) {                                       // If malloc failed,
      return PAPI_ENOMEM;                                               // .. out of memory.
   } // end if malloc failed.

   //------------------------------------------------------------------
   // pcpIndex alloc managed, now process all events passed in.
   // pcpIndex[i] holds the event pcp_event_info[] index for 
   // EventSet[i], and we populate the caller's ni_position for 
   // EventSet[i] with the index into pcpIndex[].
   //------------------------------------------------------------------

   for (i=0; i<count; i++) {                                            // for each event passed in,
      index = native[i].ni_event & PAPI_NATIVE_AND_MASK;                // get index.
      if (index < 0 || index >= sEventCount) {                          // if something is wrong, 
         return PAPI_ENOEVNT;                                           // no such event.
      } // end if index invalid.

      MyCtl->pcpIndex[i]=index;                                         // remember the index.
      MyCtl->pcpValue[i]=0;                                             // clear the value.   
      native[i].ni_position = i;                                        // Tell PAPI about its location (doesn't matter to us), we have no restrictions on position.
      getPMDesc(index);                                                 // Any time an event is added, ensure we have its variable descriptor.
   } // end for each event listed.

   return PAPI_OK;
} // end routine.


//---------------------------------------------------------------------
// Helper routine, for reset and read, does the work of reading all
// current raw values in an EventSet (hwd_control_state). 
//
// 1) Does not subtract zeroValue; returns raw read in ULL format.
// 2) Does not change pcp_event_info[] in any way.
// 3) stores values in _pcp_control_state.pcpValue[] for each event.
// 4) Returns with 'results' malloc'd by PCP, nest mallocs, SO: 
// 5) if (results != NULL) pcp_pmFreeResults(results);
// 6) must check on 5 EVEN IF THERE WAS AN ERROR. 
//
// RETURNS PAPI error code, or PAPI_OK.
//---------------------------------------------------------------------

static int PCP_ReadList(hwd_control_state_t *ctl,                       // the event set.
    pmResult **results)                                                 // results from pmFetch, caller must pmFreeResult(results).
{
   int i, j, ret;
    _pcp_control_state_t* myCtl = ( _pcp_control_state_t* ) ctl;
   *results = NULL;                                                     // Nothing allocated.
   if (myCtl->numEvents < 1) return PAPI_ENOEVNT;                       // No events to start.
   int nPMID = 0;                                                       // To count number of **unique** PMIDs.

   pmID *allPMID=malloc((myCtl->numEvents)*sizeof(pmID));               // Make maximum possible room.

   // We build a list of all *unique* PMIDs. Because PMID can return
   // an array of N values for a single event (e.g. one per CPU), we
   // 'explode' such events into N events for PAPI, which can only
   // return 1 value per event. Thus PAPI could add several to an
   // EventSet that all have the same PMID (PCP's ID). We only need
   // to read those once, our pcp_event_info[] contains the index
   // for each exploded event into the array returned for that PMID.

   allPMID[nPMID++] = pcp_event_info[myCtl->pcpIndex[0]].pmid;          // Move the first, increment total so far. 

   for (i=1; i<myCtl->numEvents; i++) {                                 // For every event in the event set,
      int myIdx = myCtl->pcpIndex[i];                                   // .. Get pcp_event_info[] index of the event,
      pmID myPMID = pcp_event_info[myIdx].pmid;                         // .. get the PMID for that event,
      for (j=0; j<nPMID; j++) {                                         // .. Search the unique PMID list for a match. 
         if (myPMID == allPMID[j]) break;                               // .. .. found it. break out.
      } 

      if (j == nPMID) {                                                 // full loop ==> myPMID was not found in list,
         allPMID[nPMID++] = myPMID;                                     // .. store the unique pmid in list, inc count.
      }
   }                                                                    // done building list of unique pmid.
   
   // nPMID is # of unique PMID, now ready to read all the pmid needed.
   pmResult *allFetch = NULL;                                           // result of pmFetch. 
   ret = pcp_pmFetch(nPMID, allPMID, &allFetch);                        // Fetch them all.
   *results = allFetch;                                                 // For either success or failure.
   
   if( ret != PAPI_OK) {                                                // If fetch failed .. 
      fprintf(stderr, "%s:%i:%s pcp_pmFetch failed, return=%s.\n", 
         __FILE__, __LINE__, FUNC, PAPI_strerror(ret));                 // .. report error.
         if (allPMID != NULL)  free(allPMID);                           // .. no memory leak.
         allPMID = NULL;                                                // .. prevent future use.
      return(ret);                                                      // .. exit with that error.
   }

   // For each unique PMID we just read, we must map it to the event
   // sets, which may contain multiple entries with that same PMID.
   // This is because PCP returns arrays, and PAPI does not, so each
   // of our names translates to a PMID + an index.

   for (i=0; i<nPMID; i++) {                                            // process all the fetch results.
      pmValueSet *vset = allFetch->vset[i];                             // get the result for event[i].
      pmID myPMID = allPMID[i];                                         // get the pmid from this read.
       
      // Now we must search for any events with this pmid, and get
      // the corresponding value (may be more than one, since we
      // treat each idx as its own value).         

      for (j=0; j<myCtl->numEvents; j++) {                              // for each event,
         int myPCPIdx = myCtl->pcpIndex[j];                             // .. get my pcp_event_info[] index.
         pmID thisPMID = pcp_event_info[myPCPIdx].pmid;                 // .. collect its pmid.
         if (thisPMID == myPMID) {                                      // .. If this result services that event, 
            int myArrayIdx = pcp_event_info[myPCPIdx].idx;              // .. .. get array index within result array, for result value list.
            myCtl->pcpValue[j] = getULLValue(vset, myArrayIdx);         // .. .. translate as needed, put back into pcpValue array.
         } // end if counter was found for this PMID.
      } // end loop through all events in this event set.
   } // end for each pmID read.

   if (allPMID != NULL)  free(allPMID);                                 // Done with this work area; no memory leak.
   return PAPI_OK;                                                      // All done.
} // end routine.


//----------------------------------------------------------------------------
// Reset counters.
//---------------------------------------------------------------------------
static int _pcp_reset(hwd_context_t *ctx, hwd_control_state_t *ctl) 
{
   mRtnCnt(_pcp_reset);                                                 // count this function.
   ( void ) ctx;                                                        // avoid unused var warning.
   int i, k, ret;
   unsigned long long aValue;

   pmResult *allFetch;                                                  // vars to be allocated by call.
   _pcp_control_state_t* myCtl = ( _pcp_control_state_t* ) ctl;         // make a shortcut.

   ret = PCP_ReadList(ctl, &allFetch);                                  // Read the list of events we were given.
   if (ret != PAPI_OK) {                                                // If that failed, 
      fprintf(stderr, "%s:%i:%s PCP_ReadList failed, return=%s.\n", 
         __FILE__, __LINE__, FUNC, PAPI_strerror(ret));                 // report error.
      if (allFetch != NULL) pcp_pmFreeResult(allFetch);                 // free if it was allocated.
      return(ret);                                                      // exit with that error.
   }

   // We have all the results; the values read have been extracted
   // and stored the control state.  Now set them as the zeroValue
   // in each pcp_event_info[].

   for (i=0; i<myCtl->numEvents; i++) {                                 // for each event, 
      k = myCtl->pcpIndex[i];                                           // .. get pcp_event_info[] index.
      aValue = myCtl->pcpValue[i];                                      // .. get the value for that event.
      pcp_event_info[k].zeroValue = aValue;                             // .. reset the counter. 
   } // end loop through all events in this event set.

   // That is all we do; reset the zeroValue to the current value.
   // For efficiency we do not check if it is a counter, instantaneous
   // or discrete value; that is done in subZero.
   pcp_pmFreeResult(allFetch);                                          // .. Clean up.
   return PAPI_OK;
} // end routine. 


//---------------------------------------------------------------------
// Start counters. We just reset the counters; read them and set
// zeroValue on each.
//---------------------------------------------------------------------

static int _pcp_start( hwd_context_t *ctx, hwd_control_state_t *ctl) 
{
   mRtnCnt(_pcp_start);                                                 // count this function.
   _pcp_reset(ctx, ctl);                                                // Just reset counters.
   return PAPI_OK;
} // end routine.


//---------------------------------------------------------------------
// read several pcp values. 
// This is triggered by PAPI_read( int EventSet, long long *values).
// However, the **events that we see is NOT the *values which is an
// array done by the application (or caller of PAPI_read). Instead,
// we must give *event the address of an array of our values. 
// The eventSet is represented by ctx and ctl, for us ctx is empty.
// The flags are a copy of PAPI's EventSetInfo_t.state, including
// PAPI_RUNNING, PAPI_STOPPED and other flags. We ignore that here.
//---------------------------------------------------------------------

static int _pcp_read(hwd_context_t *ctx,                                // unused.
         hwd_control_state_t *ctl,                                      // contains pmIDs in event set.
         long long **events,                                            // for our returns.
         int flags)                                                     // unused; EventSetInfo_t.state.
{
   mRtnCnt(_pcp_read);                                                  // count this function.
   (void) ctx;                                                          // avoid unused var warning.
   (void) flags;                                                        // ..

   _pcp_control_state_t* myCtl = ( _pcp_control_state_t* ) ctl;         // make a shortcut.
   int i, ret;
   pmResult *allFetch;                                                  // vars to be allocated by call.
   if (events == NULL) {
      fprintf(stderr, "%s:%i:%s 'events' is null.\n", 
         __FILE__, __LINE__, FUNC);                                     // report error.
      return(PAPI_EINVAL);                                              // invalid argument.
   }

   ret = PCP_ReadList(ctl, &allFetch);                                  // Read the list of events we were given.
   if (ret != PAPI_OK) {                                                // If that failed, 
      fprintf(stderr, "%s:%i:%s PCP_ReadList failed, return=%s.\n", 
         __FILE__, __LINE__, FUNC, PAPI_strerror(ret));                 // report error.
      return(ret);                                                      // exit with that error.
   }

   // We have all the results and values extracted from them.
   // Now subtract zero value from them.
      
   for (i=0; i<myCtl->numEvents; i++) {                                 // for each event, 
      subZero(myCtl, i);                                                // .. subtract zero value in proper type. [TONY DON"T COMMENT OUT, JUST DEBUG]
   } // end loop through all events in this event set.

   // Done, point the caller to our results list.
   *events = (long long *) myCtl->pcpValue;                             // pointer the caller needs. 
   pcp_pmFreeResult(allFetch);                                          // Clean up.
   return PAPI_OK;                                                      // exit, all okay.
} // end routine.


//---------------------------------------------------------------------
// stop counters. (does nothing). 
//---------------------------------------------------------------------
static int _pcp_stop(hwd_context_t *ctx, hwd_control_state_t *ctl) 
{
   mRtnCnt(_pcp_stop);                                                  // count this function.
    (void) ctx;                                                         // avoid var unused warning.
    (void) ctl;                                                         // avoid var unused warning.
    return PAPI_OK;
} // end routine.


//---------------------------------------------------------------------
// shutdown thread. (does nothing). 
//---------------------------------------------------------------------
static int _pcp_shutdown_thread(hwd_context_t * ctx) 
{
   mRtnCnt(_pcp_shutdown_thread);                                       // count this function.
    ( void ) ctx;                                                       // avoid var unused warning.

    return PAPI_OK;
} // end routine.


//---------------------------------------------------------------------
// shutdown PCP component. (frees allocs).
//---------------------------------------------------------------------
static int _pcp_shutdown_component(void) 
{
   int i;
   mRtnCnt(_pcp_shutdown_component);                                    // count this function.
   pcp_pmDestroyContext(ctxHandle);                                     // context handle; fails to free malloced memory, says valgrind.
   ctxHandle = -1;                                                      // reset to prevent reuse.
   free(pcp_event_info); pcp_event_info=NULL;                           // then pcp_event_info, reset.
   freeNameHash();                                                      // free sNameHash. resets itself.
   cachedGetInDom(PM_INDOM_NULL, -1);                                   // -1 for inst == free its local static mallocs.
   sEventCount = 0;                                                     // clear number of events. 

   for (i=0; i<=ctr_pcp_ntv_code_to_info; i++) 
      _prog_fprintf(stderr, "routine counter %i = %i.\n", i, cnt[i]);

   return PAPI_OK;
} // end routine.


//---------------------------------------------------------------------
// This function sets options in the component. 
// The valid codes being passed in are PAPI_DEFDOM, PAPI_DOMAIN,
// PAPI_DEFGRN, PAPI_GRANUL and PAPI_INHERIT.

// _papi_int_option_t: 
// _papi_int_overflow_t overflow;
// _papi_int_profile_t profile;
// _papi_int_domain_t domain;             // PAPI_SET_DEFDOM, PAPI_SET_DOMAIN
// _papi_int_attach_t attach;
// _papi_int_cpu_t cpu;
// _papi_int_multiplex_t multiplex;
// _papi_int_itimer_t itimer;
// _papi_int_inherit_t inherit;           // PAPI_SET_INHERIT
// _papi_int_granularity_t granularity;   // PAPI_SET_DEFGRN, PAPI_SET_GRANUL
// _papi_int_addr_range_t address_range;
//---------------------------------------------------------------------

static int _pcp_ctl (hwd_context_t *ctx, int code, _papi_int_option_t *option) 
{
   mRtnCnt(_pcp_ctl);                                                   // count this function.
   ( void ) ctx;                                                        // avoid var unused warning.
   ( void ) code;                                                       // avoid var unused warning.
   ( void ) option;                                                     // avoid var unused warning.

   switch (code) {
      case PAPI_DEFDOM:
         SUBDBG("PAPI_DEFDOM, option.domain=0x%08X\n", (unsigned int) option->domain.domain);
         break;

      case PAPI_DOMAIN:
         SUBDBG("PAPI_DOMAIN, option.domain=0x%08X\n", (unsigned int) option->domain.domain);
         break;

      case PAPI_DEFGRN:
         SUBDBG("PAPI_DEFGRN, option.granularity=0x%08X\n", (unsigned int) option->granularity.granularity);
         break;

      case PAPI_GRANUL:
         SUBDBG("PAPI_GRANUL, option.granularity=0x%08X\n", (unsigned int) option->granularity.granularity);
         break;

      case PAPI_INHERIT:
         SUBDBG("PAPI_INHERIT, option.inherit=0x%08X\n", (unsigned int) option->inherit.inherit);
         break;

      default:
         fprintf(stderr, "%s:%i:%s CODE 0x%08x IS INVALID "
            "OR UNRECOGNIZED.\n", __FILE__, __LINE__, FUNC, code);
         return PAPI_EINVAL;                                            // Invalid code.
         break;
   } // end switch by code.
 
   return PAPI_OK;                             
} // end routine.


//----------------------------------------------------------------------------
// This function has to set the bits needed to count different domains.
// PAPI_DOM_USER     : only user context is counted 
// PAPI_DOM_KERNEL   : only the Kernel/OS context is counted 
// PAPI_DOM_OTHER    : Exception/transient mode (like user TLB misses) 
// PAPI_DOM_ALL      : all of the domains, THE ONLY ONE WE ACCEPT!
// All other domains result in an invalid value.
//----------------------------------------------------------------------------
static int _pcp_set_domain(hwd_control_state_t *ctl, int domain) 
{
   mRtnCnt(_pcp_set_domain);                                            // count this function.
    (void) ctl;                                                         // avoid var unused warning.
    if ( PAPI_DOM_ALL != domain ) return PAPI_EINVAL;                   // Reject if not this one.
    return PAPI_OK;                                                     // Otherwise, OK, nothing to do.
} // end routine.


//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// NATIVE EVENT ROUTINES.
//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

//----------------------------------------------------------------------------
// Enumerate events. PAPI_NTV_ENUM_UMASKS has nothing to do; we don't have 
// qualifying masks or options on any of our events.
//----------------------------------------------------------------------------

static int _pcp_ntv_enum_events(unsigned int *EventCode, int modifier) 
{
   mRtnCnt(_pcp_ntv_enum_events);                                       // count this function.
   int idx;

   switch (modifier) {                                                  // modifier is type of enum operation desired.
       case PAPI_ENUM_FIRST:                                            // Returns event code of first event created.
           EventCode[0] = 0;                                            // Return 0 as event code after a start.
           return PAPI_OK;                                              // EXIT.
           break;                                                       // END CASE.
                                                                        
       // return EventCode of next available event.                     
       case PAPI_ENUM_EVENTS:                                           // enum base events (which is all events).
           idx = EventCode[0] & PAPI_NATIVE_AND_MASK;                   // Take off any hi order flag bits.
           if ((++idx) >= sEventCount) return PAPI_ENOEVNT;             // If we reach an invalid idx for pcp_event_info[], exit. Does nothing to EventCode.
           EventCode[0] = idx | PAPI_NATIVE_MASK;                       // If index was valid, we return it.
           return PAPI_OK;                                              // And exit.
           break;                                                       // END CASE.
                                                                        
       case PAPI_NTV_ENUM_UMASKS:                                       // Note we HAVE no qualifiers or masks.
           return PAPI_ENOEVNT;                                         // There are no qualifiers to list.
                                                                        
       default:                                                         // If we don't understand the modifier,
           return PAPI_ENOEVNT;                                         // .. Presets or other stuff, just say we have none.
           break;                                                       // END CASE.
   } // end switch(modifier).                                           
                                                                        
   return PAPI_EBUG;                                                    // Dummy return; should have exited from inside switch.
} // end routine.


//----------------------------------------------------------------------------
// Given a string, find the name in the pcp_event_info[] array.  
//---------------------------------------------------------------------------
static int _pcp_ntv_name_to_code(const char *name, unsigned int *event_code) 
{
   mRtnCnt(_pcp_ntv_name_to_code);                                      // count this function.

   if (name == NULL || strlen(name)<1) {                                // Invalid name argument.
      fprintf(stderr, "%s:%i:%s Invalid name.\n",                       // .. report it.
         __FILE__, __LINE__, FUNC);                            
         return PAPI_EINVAL;                                            // .. Invalid argument.
   }

   if (event_code == NULL) {                                            // Invalid event_code pointer.
      fprintf(stderr, "%s:%i:%s event_code is not a valid pointer.\n",  // .. report it.
         __FILE__, __LINE__, FUNC);                            
         return PAPI_EINVAL;                                            // .. Invalid argument.
   }

   int idx = findNameHash((char*) name);                                // Use our hash to find it.
   if (idx < 0) {                                                       // If we failed, 
      fprintf(stderr, "%s:%i:%s Failed to find name='%s', hash=%i.\n",  // .. report it.
         __FILE__, __LINE__, FUNC, name, 
         stringHash((char*) name, HASH_SIZE));                          // .. 
         return PAPI_EINVAL;                                            // .. Invalid argument.
   }

   *event_code = idx;                                                   // return with the index we found.
   return PAPI_OK;
} // end routine.


//----------------------------------------------------------------------------
// Collect the name of the event from the EventCode, given here as our index
// into pcp_event_info[]. We must fit it into name[len].
//----------------------------------------------------------------------------
static int _pcp_ntv_code_to_name(unsigned int pcpIdx, char *name, int len)
{
   mRtnCnt(_pcp_ntv_code_to_name);                                      // count this function.

   pcpIdx &= PAPI_NATIVE_AND_MASK;                                      // We can be called with the NATIVE bit set.
   if (pcpIdx >= (unsigned int) sEventCount) {                          // out of range?
      fprintf(stderr, "%s:%i:%s called with out-of-range pcpIdx=%u.\n", 
         __FILE__, __LINE__, FUNC, pcpIdx);
      return PAPI_EINVAL;                                               // exit with error.
   }

   if (len < 1)  {                                                      // If length is ridiculous,
      fprintf(stderr, "%s:%i:%s called with out-of-range descr len=%i.\n", 
         __FILE__, __LINE__, FUNC, len);
      return PAPI_EINVAL;                                               // exit with error.
   }

   strncpy(name, pcp_event_info[pcpIdx].name, len);                     // just copy the name.
   name[len-1]=0;                                                       // force z-termination.

   return PAPI_OK;
} // end routine.


//----------------------------------------------------------------------------
// Collect the text description of the EventCode; which is our index into our
// pcp_event_info[] array. We must fit it into descr[len].  
//---------------------------------------------------------------------------

static int _pcp_ntv_code_to_descr(unsigned int pcpIdx, char *descr, int len) 
{
   mRtnCnt(_pcp_ntv_code_to_descr);                                     // count this function.

   pcpIdx &= PAPI_NATIVE_AND_MASK;                                      // We might be called with the NATIVE bit set.
   if (pcpIdx >= (unsigned int) sEventCount) {                          // out of range?
      fprintf(stderr, "%s:%i:%s called with out-of-range pcpIdx=%u.\n", 
         __FILE__, __LINE__, FUNC, pcpIdx);
      return PAPI_EINVAL;                                               // exit with error.
   }

   if (len < 1)  {                                                      // If length is ridiculous,
      fprintf(stderr, "%s:%i:%s called with out-of-range descr len=%i.\n", 
         __FILE__, __LINE__, FUNC, len);
      return PAPI_EINVAL;                                               // exit with error.
   }

   char *helpText = NULL;                                               // pointer to receive the result.
   int ret = getHelpText(pcpIdx, &helpText);                            // get it. 
   if (ret != PAPI_OK) {                                                // If there is any error,
      if (helpText != NULL) free(helpText);                             // .. no memory leak.
      fprintf(stderr, "%s:%i:%s failed getHelpText; it returned %s.\n", 
         __FILE__, __LINE__, FUNC, PAPI_strerror(ret));
      return ret;                                                       // exit with whatever PAPI error routine had.
   }

   strncpy(descr, helpText, len);                                       // copy it over.
   descr[len-1] = 0;                                                    // force a z-terminator.
   free(helpText);                                                      // release text alloc by pm routine.
   return PAPI_OK;                                                      // EXIT, all good.
} // end routine.


//---------------------------------------------------------------------
// Fill in the PAPI_event_info_t vars. This is not a required
// function for a component, not all fields require filling in.
// Components that implement this feature generally complete the
// following fields:

// char symbol[PAPI_HUGE_STR_LEN];     // (1024 char, name of the event),
// char long_descr[PAPI_HUGE_STR_LEN]; // (1024 char, can be a paragraph);
// char units[PAPI_MIN_STR_LEN];       // (64 chars, unit of measurement);
// int  data_type;                     // data type returned by PAPI.       
// 
// data_type is PAPI_DATATYPE_INT64, PAPI_DATATYPE_UINT64,
// PAPI_DATATYPE_FP64, PAPI_DATATYPE_BIT64.
// We translate all values into INT64, UINT64, or FP64.
//
// timescope;                          // Counter or instantaneous.
// PAPI_TIMESCOPE_SINCE_START          // Data is cumulative from start.
// PAPI_TIMESCOPE_POINT                // Data is an instantaneous value.
//---------------------------------------------------------------------

static int _pcp_ntv_code_to_info(unsigned int pcpIdx, PAPI_event_info_t *info) 
{
   mRtnCnt(_pcp_ntv_code_to_info);                                      // count this function.
   int len, ret;   

   pcpIdx &= PAPI_NATIVE_AND_MASK;                                      // remove any high order bits.
   if (pcpIdx >= (unsigned int) sEventCount) {                          // out of range?
      fprintf(stderr, "%s:%i:%s called with out-of-range pcpIdx=%u.\n", 
         __FILE__, __LINE__, FUNC, pcpIdx);
      return PAPI_EINVAL;                                               // exit with error.
   }

   len=sizeof(info->symbol);                                            // get length.
   strncpy(info->symbol, pcp_event_info[pcpIdx].name, len);             // Copy. 
   info->symbol[len-1] = 0;                                             // force z-terminator.

   len=sizeof(info->long_descr);                                        // get length.
   ret = _pcp_ntv_code_to_descr(pcpIdx, info->long_descr, len);         // copy to info.
   if (ret != PAPI_OK) return(ret);                                     // return on failure. _pcp_ntv_code_to_descr already printed error.

   // Units resides in pmDesc; we need to get it if we don't already
   // have it (multi-valued events got it during init).

   getPMDesc(pcpIdx);                                                   // get the description.

   char unitStr[64];
   // This routine has been timed over 19600 trials on Saturn;
   // it requires an average of 2 uS. No daemon comm needed.

   pcp_pmUnitsStr_r(&pcp_event_info[pcpIdx].desc.units, unitStr, 64);   // Construct the unit string; needs at least 60 char.
   if ( strlen(unitStr) == 0) {
      sprintf(unitStr, "fraction");                                     // Only ever seen for 'dutycycle' events.

      // Following is for debug purposes. 
      if (0) {                                                          // Alternatively, show the details of the PCP units descriptor.
         sprintf(unitStr, "[%u, %i, %u, %u, %i, %i, %i]", 
            pcp_event_info[pcpIdx].desc.units.pad,
            pcp_event_info[pcpIdx].desc.units.scaleCount,
            pcp_event_info[pcpIdx].desc.units.scaleTime,
            pcp_event_info[pcpIdx].desc.units.scaleSpace,
            pcp_event_info[pcpIdx].desc.units.dimCount,
            pcp_event_info[pcpIdx].desc.units.dimTime,
            pcp_event_info[pcpIdx].desc.units.dimSpace
         );
      } 
   }

   len = sizeof(info->units);                                           // length of destination.
   strncpy( info->units, unitStr, len);                                 // copy it over.
   info->units[len-1] = 0;                                              // ensure a z-terminator.

   switch (pcp_event_info[pcpIdx].valType) {                            // Translate this to a papi value.

      case  PM_TYPE_32:                                                 // 32 bit was converted to long long.
      case  PM_TYPE_64:                                                 // long long.
         info->data_type = PAPI_DATATYPE_INT64;                         // What papi needs.
         break;                                                         // END CASE.

      case  PM_TYPE_U32:                                                // 32-bit was converted to 64 bit.
      case  PM_TYPE_U64:                                                // 64-bit unsigned integer
      case  PM_TYPE_STRING:                                             // array of char pointer.
         info->data_type = PAPI_DATATYPE_UINT64;                        // What papi needs.
         break;                                                         // END CASE.

      case  PM_TYPE_FLOAT:                                              // 32-bit was converted to double.
      case  PM_TYPE_DOUBLE:                                             // 64-bit floating point
         info->data_type = PAPI_DATATYPE_FP64;                          // What papi needs.
         break;                                                         // END CASE.
   };

   if (pcp_event_info[pcpIdx].desc.sem == PM_SEM_COUNTER) {             // If we have a counter,
      info->timescope = PAPI_TIMESCOPE_SINCE_START;                     // .. normal stuff.
   } else {                                                             // An instantaneous value. 
      info->timescope = PAPI_TIMESCOPE_POINT;                           // .. What PAPI calls that.
   }

   return PAPI_OK;                                                      // exit.
} // end routine.

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
// PCP Vector definition. 
//---------------------------------------------------------------------------
//----------------------------------------------------------------------------

papi_vector_t _pcp_vector = {
   .cmp_info = { /* (unspecified values are initialized to 0) */
      .name = "pcp",
      .short_name = "pcp",
      .description = "Performance Co-Pilot",
      .version = "5.6.1",
      .default_domain = PAPI_DOM_ALL,
      .default_granularity = PAPI_GRN_SYS,
      .available_granularities = PAPI_GRN_SYS,
      .hardware_intr_sig = PAPI_INT_SIGNAL,
      .available_domains = PAPI_DOM_ALL,
   },

   /* sizes of framework-opaque component-private structures */
   .size = {
      .context          = sizeof ( _pcp_context_t ),
      .control_state    = sizeof ( _pcp_control_state_t ),
      .reg_value        = sizeof ( _pcp_register_t ),
      .reg_alloc        = sizeof ( _pcp_reg_alloc_t ),
   },

   /* function pointers in this component */
   .init_thread         = _pcp_init_thread,
   .init_component      = _pcp_init_component,
   .init_control_state  = _pcp_init_control_state,
   .start               = _pcp_start,
   .stop                = _pcp_stop,
   .read                = _pcp_read,
   .shutdown_thread     = _pcp_shutdown_thread,
   .shutdown_component  = _pcp_shutdown_component,
   .ctl                 = _pcp_ctl,

   .update_control_state= _pcp_update_control_state,
   .set_domain          = _pcp_set_domain,
   .reset               = _pcp_reset,

   .ntv_enum_events     = _pcp_ntv_enum_events,
   .ntv_name_to_code    = _pcp_ntv_name_to_code,
   .ntv_code_to_name    = _pcp_ntv_code_to_name,
   .ntv_code_to_descr   = _pcp_ntv_code_to_descr,
   .ntv_code_to_info    = _pcp_ntv_code_to_info,
}; // end pcp_vector.
