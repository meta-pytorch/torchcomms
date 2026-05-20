/** 
* @file    papi_preset.h
* @author  Haihang You
*          you@cs.utk.edu
*/

#ifndef _PAPI_PRESET		 /* _PAPI_PRESET */
#define _PAPI_PRESET

/** search element for preset events defined for each platform 
 *	@internal */
typedef struct hwi_search {   
  /* eventcode should have a more specific name, like papi_preset! -pjm */
   unsigned int event_code;   /**< Preset code that keys back to sparse preset array */
   int derived;                          /**< Derived type code */
   int native[PAPI_EVENTS_IN_DERIVED_EVENT];   /**< array of native event code(s) for this preset event */
   char operation[PAPI_2MAX_STR_LEN];    /**< operation string: +,-,*,/,@(number of metrics), $(constant Mhz), %(1000000.0) */
   char *note;                          /**< optional developer notes for this event */
} hwi_search_t;

/** collected text and data info for all preset events 
 *	@internal */
typedef struct hwi_presets {  
   char *symbol;      /**< name of the preset event; i.e. PAPI_TOT_INS, etc. */
   char *short_descr; /**< short description of the event for labels, etc. */
   char *long_descr;  /**< long description (full sentence) */
   int derived_int;   /**< Derived type code */

   unsigned int count;
   unsigned int event_type;
   char *postfix;
   unsigned int code[PAPI_MAX_INFO_TERMS];
   char *name[PAPI_MAX_INFO_TERMS];
   char *note;
} hwi_presets_t;


/** This is a general description structure definition for various parameter lists 
 *	@internal */   
typedef struct hwi_describe {
   int value;                 /**< numeric value (from papi.h) */
   char *name;                /**< name of the element */
   char *descr;               /**< description of the element */
} hwi_describe_t;

extern hwi_search_t *preset_search_map;

int _papi_hwi_setup_all_presets( hwi_search_t * findem, int cidx);
int _papi_hwi_cleanup_all_presets( void );
int _xml_papi_hwi_setup_all_presets( char *arch);
int _papi_load_preset_table( char *name, int type, int cidx );

extern hwi_presets_t _papi_hwi_presets[PAPI_MAX_PRESET_EVENTS];

#endif /* _PAPI_PRESET */
