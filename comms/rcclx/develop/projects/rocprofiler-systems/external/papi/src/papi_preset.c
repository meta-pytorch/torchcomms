/* 
* File:    papi_preset.c
* Author:  Haihang You
*          you@cs.utk.edu
* Mods:    Brian Sheely
*          bsheely@eecs.utk.edu
* Author:  Vince Weaver
*          vweaver1 @ eecs.utk.edu
*          Merge of the libpfm3/libpfm4/pmapi-ppc64_events preset code
*/


#include <string.h>
#include <ctype.h>
#include <errno.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"
#include "papi_memory.h"
#include "papi_preset.h"
#include "extras.h"


// A place to put user defined events
extern hwi_presets_t user_defined_events[];
extern int user_defined_events_count;

static int papi_load_derived_events (char *pmu_str, int pmu_type, int cidx, int preset_flag);


/* This routine copies values from a dense 'findem' array of events
   into the sparse global _papi_hwi_presets array, which is assumed
   to be empty at initialization.

   Multiple dense arrays can be copied into the sparse array, allowing
   event overloading at run-time, or allowing a baseline table to be
   augmented by a model specific table at init time.

   This method supports adding new events; overriding existing events, or
   deleting deprecated events.
*/
int
_papi_hwi_setup_all_presets( hwi_search_t * findem, int cidx )
{
    int i, pnum, did_something = 0;
    unsigned int preset_index, j, k;

    /* dense array of events is terminated with a 0 preset.
       don't do anything if NULL pointer. This allows just notes to be loaded.
       It's also good defensive programming.
     */
    if ( findem != NULL ) {
       for ( pnum = 0; ( pnum < PAPI_MAX_PRESET_EVENTS ) &&
			  ( findem[pnum].event_code != 0 ); pnum++ ) {
	   /* find the index for the event to be initialized */
	   preset_index = ( findem[pnum].event_code & PAPI_PRESET_AND_MASK );
	   /* count and set the number of native terms in this event, 
              these items are contiguous.

	      PAPI_EVENTS_IN_DERIVED_EVENT is arbitrarily defined in the high 
              level to be a reasonable number of terms to use in a derived 
              event linear expression, currently 8.

	      This wastes space for components with less than 8 counters, 
              but keeps the framework independent of the components.

	      The 'native' field below is an arbitrary opaque identifier 
              that points to information on an actual native event. 
              It is not an event code itself (whatever that might mean).
	      By definition, this value can never == PAPI_NULL.
	      - dkt */

	   INTDBG( "Counting number of terms for preset index %d, "
                   "search map index %d.\n", preset_index, pnum );
	   i = 0;
	   j = 0;
	   while ( i < PAPI_EVENTS_IN_DERIVED_EVENT ) {
	      if ( findem[pnum].native[i] != PAPI_NULL ) {
		 j++;
	      }
	      else if ( j ) {
		 break;
	      }
	      i++;
	   }

	   INTDBG( "This preset has %d terms.\n", j );
	   _papi_hwi_presets[preset_index].count = j;
 
           _papi_hwi_presets[preset_index].derived_int = findem[pnum].derived;
	   for(k=0;k<j;k++) {
              _papi_hwi_presets[preset_index].code[k] =
                     findem[pnum].native[k];
	   }
	   /* preset code list must be PAPI_NULL terminated */
	   if (k<PAPI_EVENTS_IN_DERIVED_EVENT) {
              _papi_hwi_presets[preset_index].code[k] = PAPI_NULL;
	   }

	   _papi_hwi_presets[preset_index].postfix=
	                                   papi_strdup(findem[pnum].operation);

	   did_something++;
       }
    }

    _papi_hwd[cidx]->cmp_info.num_preset_events += did_something;

    return ( did_something ? PAPI_OK : PAPI_ENOEVNT );
}

int
_papi_hwi_cleanup_all_presets( void )
{
        int preset_index,cidx;
	unsigned int j;

	for ( preset_index = 0; preset_index < PAPI_MAX_PRESET_EVENTS;
		  preset_index++ ) {
	    if ( _papi_hwi_presets[preset_index].postfix != NULL ) {
	       papi_free( _papi_hwi_presets[preset_index].postfix );
	       _papi_hwi_presets[preset_index].postfix = NULL;
	    }
	    if ( _papi_hwi_presets[preset_index].note != NULL ) {
	       papi_free( _papi_hwi_presets[preset_index].note );
	       _papi_hwi_presets[preset_index].note = NULL;
	    }
	    for(j=0; j<_papi_hwi_presets[preset_index].count;j++) {
           papi_free(_papi_hwi_presets[preset_index].name[j]);
	    }
	}

	for(cidx=0;cidx<papi_num_components;cidx++) {
	   _papi_hwd[cidx]->cmp_info.num_preset_events = 0;
	}

#if defined(ITANIUM2) || defined(ITANIUM3)
	/* NOTE: This memory may need to be freed for BG/P builds as well */
	if ( preset_search_map != NULL ) {
		papi_free( preset_search_map );
		preset_search_map = NULL;
	}
#endif

	return PAPI_OK;
}



#define PAPI_EVENT_FILE "papi_events.csv"


/*  Trims blank space from both ends of a string (in place).
    Returns pointer to new start address */
static inline char *
trim_string( char *in )
{
	int len, i = 0;
	char *start = in;

	if ( in == NULL )
		return ( in );
	len = ( int ) strlen( in );
	if ( len == 0 )
		return ( in );

	/* Trim left */
	while ( i < len ) {
		if ( isblank( in[i] ) ) {
			in[i] = '\0';
			start++;
		} else
			break;
		i++;
	}

	/* Trim right */
	i = ( int ) strlen( start ) - 1;
	while ( i >= 0 ) {
		if ( isblank( start[i] ) )
			start[i] = '\0';
		else
			break;
		i--;
	}
	return ( start );
}


/*  Calls trim_string to remove blank space;
    Removes paired punctuation delimiters from
    beginning and end of string. If the same punctuation
    appears first and last (quotes, slashes) they are trimmed;
    Also checks for the following pairs: () <> {} [] */
static inline char *
trim_note( char *in )
{
	int len;
	char *note, start, end;

	note = trim_string( in );
	if ( note != NULL ) {
                len = ( int ) strlen( note );
		if ( len > 0 ) {
			if ( ispunct( *note ) ) {
				start = *note;
				end = note[len - 1];
				if ( ( start == end )
					 || ( ( start == '(' ) && ( end == ')' ) )
					 || ( ( start == '<' ) && ( end == '>' ) )
					 || ( ( start == '{' ) && ( end == '}' ) )
					 || ( ( start == '[' ) && ( end == ']' ) ) ) {
					note[len - 1] = '\0';
					*note = '\0';
					note++;
				}
			}
		}
	}
	return note;
}

static inline int
find_event_index(hwi_presets_t *array, int size, char *tmp) {
	SUBDBG("ENTER: array: %p, size: %d, tmp: %s\n", array, size, tmp);
	int i;
	for (i = 0; i < size; i++) {
		if (array[i].symbol == NULL) {
			array[i].symbol = papi_strdup(tmp);
			SUBDBG("EXIT: i: %d\n", i);
			return i;
		}
		if (strcasecmp(tmp, array[i].symbol) == 0) {
			SUBDBG("EXIT: i: %d\n", i);
			return i;
		}
	}
	SUBDBG("EXIT: PAPI_EINVAL\n");
	return PAPI_EINVAL;
}

/* Look for an event file 'name' in a couple common locations.
   Return a valid file handle if found */
static FILE *
open_event_table( char *name )
{
	FILE *table;

	SUBDBG( "Opening %s\n", name );
	table = fopen( name, "r" );
	if ( table == NULL ) {
		SUBDBG( "Open %s failed, trying ./%s.\n", 
			name, PAPI_EVENT_FILE );
		sprintf( name, "%s", PAPI_EVENT_FILE );
		table = fopen( name, "r" );
	}
	if ( table == NULL ) {
		SUBDBG( "Open ./%s failed, trying ../%s.\n", 
			name, PAPI_EVENT_FILE );
		sprintf( name, "../%s", PAPI_EVENT_FILE );
		table = fopen( name, "r" );
	}
	if ( table ) {
		SUBDBG( "Open %s succeeded.\n", name );
	}
	return table;
}

/* parse a single line from either a file or character table
   Strip trailing <cr>; return 0 if empty */
static int
get_event_line( char *line, FILE * table, char **tmp_perfmon_events_table )
{
	int i;

	if ( table ) {
	    if ( fgets( line, LINE_MAX, table ) == NULL)
		return 0;

	    i = ( int ) strlen( line );
	    if (i == 0)
		return 0;
	    if ( line[i-1] == '\n' )
		line[i-1] = '\0';
	    return 1;
	} else {
		for ( i = 0;
			  **tmp_perfmon_events_table && **tmp_perfmon_events_table != '\n';
			  i++, ( *tmp_perfmon_events_table )++ )
			line[i] = **tmp_perfmon_events_table;
		if (i == 0)
		    return 0;
		if ( **tmp_perfmon_events_table && **tmp_perfmon_events_table == '\n' ) {
		    ( *tmp_perfmon_events_table )++;
		}
		line[i] = '\0';
		return 1;
	}
}

// update tokens in formula referring to index "old_index" with tokens referring to index "new_index".
static void
update_ops_string(char **formula, int old_index, int new_index) {
	INTDBG("ENTER:   *formula: %s, old_index: %d, new_index: %d\n", *formula?*formula:"NULL", old_index, new_index);

	int cur_index;
	char *newFormula;
	char *subtoken;
	char *tok_save_ptr=NULL;

	// if formula is null just return
	if (*formula == NULL) {
		INTDBG("EXIT: Null pointer to formula passed in\n");
		return;
	}

	// get some space for the new formula we are going to create
	newFormula = papi_calloc(strlen(*formula) + 20, 1);

	// replace the specified "replace" tokens in the new original formula with the new insertion formula
	newFormula[0] = '\0';
	subtoken = strtok_r(*formula, "|", &tok_save_ptr);
	while ( subtoken != NULL) {
//		INTDBG("subtoken: %s, newFormula: %s\n", subtoken, newFormula);
		char work[16];
		// if this is the token we want to replace with the new token index, do it now
		if ((subtoken[0] == 'N')  &&  (isdigit(subtoken[1]))) {
			cur_index = atoi(&subtoken[1]);
			// if matches old index, use the new one
			if (cur_index == old_index) {
				sprintf (work, "N%d", new_index);
				strcat (newFormula, work);
			} else if (cur_index > old_index) {
				// current token greater than old index, make it one less than what it was
				sprintf (work, "N%d", cur_index-1);
				strcat (newFormula, work);
			} else {
				// current token less than old index, copy this part of the original formula into the new formula
				strcat(newFormula, subtoken);
			}
		} else {
			// copy this part of the original formula into the new formula
			strcat(newFormula, subtoken);
		}
		strcat (newFormula, "|");
		subtoken = strtok_r(NULL, "|", &tok_save_ptr);
	}
	papi_free (*formula);
	*formula = newFormula;

	INTDBG("EXIT: newFormula: %s\n", newFormula);
	return;
}

//
// Handle creating a new derived event of type DERIVED_ADD.  This may create a new formula
// which can be used to compute the results of the new event from the events it depends on.
// This code is also responsible for making sure that all the needed native events are in the
// new events native event list and that the formula's referenced to this array are correct.
//
static void
ops_string_append(hwi_presets_t *results, hwi_presets_t *depends_on, int addition) {
	INTDBG("ENTER: results: %p, depends_on: %p, addition %d\n", results, depends_on, addition);

	int i;
	int second_event = 0;
	char newFormula[PAPI_MIN_STR_LEN] = "";
	char work[20];

	// if our results already have a formula, start with what was collected so far
	// this should only happens when processing the second event of a new derived add
	if (results->postfix != NULL) {
		INTDBG("Event %s has existing formula %s\n", results->symbol, results->postfix);
		// get the existing formula
		strncat(newFormula, results->postfix, sizeof(newFormula)-1);
		newFormula[sizeof(newFormula)-1] = '\0';
		second_event = 1;
	}

	// process based on what kind of event the one we depend on is
	switch (depends_on->derived_int) {
		case DERIVED_POSTFIX: {
			// the event we depend on has a formula, append it our new events formula

			// if event we depend on does not have a formula, report error
			if (depends_on->postfix == NULL) {
				INTDBG("Event %s is of type DERIVED_POSTFIX but is missing operation string\n", depends_on->symbol);
				return;
			}

			// may need to renumber the native event index values in the depends on event formula before putting it into new derived event
			char *temp = papi_strdup(depends_on->postfix);

			// If this is not the first event of the new derived add, need to adjust native event index values in formula.
			// At this time we assume that all the native events in the second events formula are unique for the new event
			// and just bump the indexes by the number of events already known to the new event.  Later when we add the events
			// to the native event list for this new derived event, we will check to see if the native events are already known
			// to the new derived event and if so adjust the indexes again.
			if (second_event) {
				for ( i=depends_on->count-1 ; i>=0 ; i--) {
					update_ops_string(&temp, i, results->count + i);
				}
			}

			// append the existing formula from the event we depend on (but get rid of last '|' character)
			strncat(newFormula, temp, sizeof(newFormula)-1);
			newFormula[sizeof(newFormula)-1] = '\0';
			papi_free (temp);
			break;
		}
		case DERIVED_ADD: {
			// the event we depend on has no formula, create a formula for our new event to add together the depends_on native event values

			// build a formula for this add event
			sprintf(work, "N%d|N%d|+|", results->count, results->count + 1);
			strcat(newFormula, work);
			break;
		}
		case DERIVED_SUB: {
			// the event we depend on has no formula, create a formula for our new event to subtract the depends_on native event values

			// build a formula for this subtract event
			sprintf(work, "N%d|N%d|-|", results->count, results->count + 1);
			strcat(newFormula, work);
			break;
		}
		case NOT_DERIVED: {
			// the event we depend on has no formula and is itself only based on one native event, create a formula for our new event to include this native event

			// build a formula for this subtract event
			sprintf(work, "N%d|", results->count);
			strcat(newFormula, work);
			break;
		}
		default: {
			// the event we depend on has unsupported derived type, put out some debug and give up
			INTDBG("Event %s depends on event %s which has an unsupported derived type of %d\n", results->symbol, depends_on->symbol, depends_on->derived_int);
			return;
		}
	}

	// if this was the second event, append to the formula an operation to add or subtract the results of the two events
	if (second_event) {
		if (addition != 0) {
			strcat(newFormula, "+|");
		} else {
			strcat(newFormula, "-|");
		}
		// also change the new derived events type to show it has a formula now
		results->derived_int = DERIVED_POSTFIX;
	}

	// we need to free the existing space (created by malloc and we need to create a new one)
	papi_free (results->postfix);
	results->postfix = papi_strdup(newFormula);
	INTDBG("EXIT: newFormula: %s\n", newFormula);
	return;
}

// merge the 'insertion' formula into the 'original' formula replacing the
// 'replaces' token in the 'original' formula.
static void
ops_string_merge(char **original, char *insertion, int replaces, int start_index) {
	INTDBG("ENTER: original: %p, *original: %s, insertion: %s, replaces: %d, start_index: %d\n", original, *original, insertion, replaces, start_index);

	int orig_len=0;
	int ins_len=0;
	char *subtoken;
	char *workBuf;
	char *workPtr;
	char *tok_save_ptr=NULL;
	char *newOriginal;
	char *newInsertion;
	char *newFormula;
	int insert_events;

	if (*original != NULL) {
		orig_len = strlen(*original);
	}
	if (insertion != NULL) {
		ins_len = strlen(insertion);
	}
	newFormula = papi_calloc (orig_len + ins_len + 40, 1);

	// if insertion formula is not provided, then the original formula remains basically unchanged.
	if (insertion == NULL) {
		// if the original formula has a leading '|' then get rid of it
		workPtr = *original;
		if (workPtr[0] == '|') {
			strcpy(newFormula, &workPtr[1]);
		} else {
			strcpy(newFormula, workPtr);
		}
		// formula fields are always malloced space so free the previous one
		papi_free (*original);
		*original = newFormula;
		INTDBG("EXIT: newFormula: %s\n", *original);
		return;
	}

	// renumber the token numbers in the insertion formula
	// also count how many native events are used in this formula
	insert_events = 0;
	newInsertion = papi_calloc(ins_len+20, 1);
	workBuf = papi_calloc(ins_len+10, 1);
	workPtr = papi_strdup(insertion);
	subtoken = strtok_r(workPtr, "|", &tok_save_ptr);
	while ( subtoken != NULL) {
//		INTDBG("subtoken: %s, newInsertion: %s\n", subtoken, newInsertion);
		if ((subtoken[0] == 'N')  &&  (isdigit(subtoken[1]))) {
			insert_events++;
			int val = atoi(&subtoken[1]);
			val += start_index;
			subtoken[1] = '\0';
			sprintf (workBuf, "N%d", val);
		} else {
			strcpy(workBuf, subtoken);
		}
		strcat (newInsertion, workBuf);
		strcat (newInsertion, "|");
		subtoken = strtok_r(NULL, "|", &tok_save_ptr);
	}
	papi_free (workBuf);
	papi_free (workPtr);
	INTDBG("newInsertion: %s\n", newInsertion);

	// if original formula is not provided, then the updated insertion formula becomes the new formula
	// but we still had to renumber the native event tokens in case another native event was put into the list first
	if (*original == NULL) {
		*original = papi_strdup(newInsertion);
		INTDBG("EXIT: newFormula: %s\n", newInsertion);
		papi_free (newInsertion);
		papi_free (newFormula);
		return;
	}

	// if token to replace not valid, return null  (do we also need to check an upper bound ???)
	if ((replaces < 0)) {
		papi_free (newInsertion);
		papi_free (newFormula);
		INTDBG("EXIT: Invalid value for token in original formula to be replaced\n");
		return;
	}

	// renumber the token numbers in the original formula
	// tokens with an index greater than the replaces token need to be incremented by number of events in insertion formula-1
	newOriginal = papi_calloc (orig_len+20, 1);
	workBuf = papi_calloc(orig_len+10, 1);
	workPtr = papi_strdup(*original);

	subtoken = strtok_r(workPtr, "|", &tok_save_ptr);
	while ( subtoken != NULL) {
//		INTDBG("subtoken: %s, newOriginal: %s\n", subtoken, newOriginal);
		// prime the work area with the next token, then see if we need to change it
		strcpy(workBuf, subtoken);
		if ((subtoken[0] == 'N')  &&  (isdigit(subtoken[1]))) {
			int val = atoi(&subtoken[1]);
			if (val > replaces) {
				val += insert_events-1;
				subtoken[1] = '\0';
				sprintf (workBuf, "N%d", val);
			}
		}
		// put the work buffer into the new original formula
		strcat (newOriginal, workBuf);
		strcat (newOriginal, "|");
		subtoken = strtok_r(NULL, "|", &tok_save_ptr);
	}
	papi_free (workBuf);
	papi_free (workPtr);
	INTDBG("newOriginal: %s\n", newOriginal);

	// replace the specified "replace" tokens in the new original formula with the new insertion formula
	newFormula[0] = '\0';
	workPtr = newOriginal;
	subtoken = strtok_r(workPtr, "|", &tok_save_ptr);
	while ( subtoken != NULL) {
//		INTDBG("subtoken: %s, newFormula: %s\n", subtoken, newFormula);
		// if this is the token we want to replace with the insertion string, do it now
		if ((subtoken[0] == 'N')  &&  (isdigit(subtoken[1]))  &&  (replaces == atoi(&subtoken[1]))) {
			// copy updated insertion string into the original string (replacing this token)
			strcat(newFormula, newInsertion);
		} else {
			// copy this part of the original formula into the new formula
			strcat(newFormula, subtoken);
			strcat(newFormula, "|");
		}
		subtoken = strtok_r(NULL, "|", &tok_save_ptr);
	}
	papi_free (newInsertion);
	papi_free (workPtr);

	// formula fields are always malloced space so free the previous one
	papi_free (*original);
	*original = newFormula;
	INTDBG("EXIT: newFormula: %s\n", newFormula);
	return;
}

//
//  Check to see if an event the new derived event being created depends on is known.  We check both preset and user defined derived events here.
//  If it is a known derived event then we set the new event being defined to include the necessary native events and formula to compute its
//  derived value and use it in the correct context of the new derived event being created.  Depending on the inputs, the operations strings (formulas)
//  to be used by the new derived event may need to be created and/or adjusted to reference the correct native event indexes for the new derived event.
//  The formulas processed by this code must be reverse polish notation (RPN) or postfix format and they must contain place holders (like N0, N1) which
//  identify indexes into the native event array used to compute the new derived events final value.
//
//  Arguments:
//    target:  event we are looking for
//    derived_type:  type of derived event being created (add, subtract, postfix)
//    results:  where to build the new preset event being defined.
//    search: table of known existing preset or user events the new derived event is allowed to use (points to a table of either preset or user events).
//    search_size:  number of entries in the search table.
//
static int
check_derived_events(char *target, int derived_type, hwi_presets_t* results, hwi_presets_t * search, int search_size, int token_index)
{
	INTDBG("ENTER: target: %p (%s), results: %p, search: %p, search_size: %d, token_index: %d\n", target, target, results, search, search_size, token_index);
	unsigned int i;
	int j;
	int k;
	int found = 0;

	for (j=0; j < search_size; j++) {
		//	INTDBG("search[%d].symbol: %s, looking for: %s\n", j, search[j].symbol, target);
		if (search[j].symbol == NULL) {
			INTDBG("EXIT: returned: 0\n");
			return 0;
		}

		// if not the event we depend on, just look at next
		if ( strcasecmp( target, search[j].symbol) != 0 ) {
			continue;
		}

		INTDBG("Found a match\n");

		// derived formulas need to be adjusted based on what kind of derived event we are processing
		// the derived type passed to this function is the type of the new event being defined (not the events it is based on)
		// when we get here the formula must be in reverse polish notation (RPN) format
		switch (derived_type) {
			case DERIVED_POSTFIX: {
				// go create a formula to merge the second formula into a spot identified by one of the tokens in
				// the first formula.
				ops_string_merge(&(results->postfix), search[j].postfix, token_index, results->count);
				break;
			}
			case DERIVED_ADD: {
				// the new derived event adds two things together, go handle this target events role in the add
				ops_string_append(results, &search[j], 1);
				break;
			}
			case DERIVED_SUB: {
				// go create a formula to subtract the value generated by the second formula from the value generated by the first formula.
				ops_string_append(results, &search[j], 0);
				break;
			}
				default: {
				INTDBG("Derived type: %d, not currently handled\n", derived_type);
				break;
			}
		}

		// copy event name and code used by the derived event into the results table (place where new derived event is getting created)
		for ( k = 0; k < (int)search[j].count; k++ ) {
//			INTDBG("search[%d]: %p, name[%d]: %s, code[%d]: %#x\n", j, &search[j], k, search[j].name[k], k, search[j].code[k]);
			// if this event is already in the list, just update the formula so that references to this event point to the existing one
			for (i=0 ; i < results->count ; i++) {
				if (results->code[i] == search[j].code[k]) {
					INTDBG("event: %s, code: %#x, already in results at index: %d\n", search[j].name[k], search[j].code[k], i);
					// replace all tokens in the formula that refer to index "results->count + found" with a token that refers to index "i".
					// the index "results->count + found" identifies the index used in the formula for the event we just determined is a duplicate
					update_ops_string(&(results->postfix), results->count + found, i);
					found++;
					break;
				}
			}

			// if we did not find a match, copy native event info into results array
			if (found == 0) {
				// not a duplicate, go ahead and copy into results and bump number of native events in results
				if (search[j].name[k]) {
					results->name[results->count] = papi_strdup(search[j].name[k]);
				} else {
					results->name[results->count] = papi_strdup(target);
				}
				results->code[results->count] = search[j].code[k];
				INTDBG("results: %p, name[%d]: %s, code[%d]: %#x\n", results, results->count, results->name[results->count], results->count, results->code[results->count]);

				results->count++;
			}
		}

		INTDBG("EXIT: returned: 1\n");
		return 1;
	}

	INTDBG("EXIT: returned: 0\n");
	return 0;
}

static int
check_native_events(char *target, hwi_presets_t* results)
{
	INTDBG("ENTER: target: %p (%s), results: %p\n", target, target, results);
	int ret;

	// find this native events code
	if ( ( ret = _papi_hwi_native_name_to_code( target, (int *)(&results->code[results->count])) ) != PAPI_OK ) {
		INTDBG("EXIT: returned: 0, call to convert name to event code failed with ret: %d\n", ret);
		return 0;
	}

	// if the code returned was 0, return to show it is not a valid native event
	if ( results->code[results->count] == 0 ) {
		INTDBG( "EXIT: returned: 0, event code not found\n");
		return 0;
	}

	// if this native event is not for component 0, return to show it can not be used in derived events
	// it should be possible to create derived events for other components as long as all events in the derived event are associated with the same component
	if ( _papi_hwi_component_index(results->code[results->count]) != 0 ) {
		INTDBG( "EXIT: returned: 0, new event not associated with component 0 (current limitation with derived events)\n");
		return 0;
	}

	//	  found = 1;
	INTDBG("\tFound a native event %s\n", target);
	results->name[results->count++] = papi_strdup(target);

	INTDBG( "EXIT: returned: 1\n");
	return 1;
}

// see if the event_name string passed in matches a known event name
// if it does these calls also updates information in event definition tables to remember the event
static int
is_event(char *event_name, int derived_type, hwi_presets_t* results, int token_index) {
	INTDBG("ENTER: event_name: %p (%s), derived_type: %d, results: %p, token_index: %d\n", event_name, event_name, derived_type, results, token_index);

	/* check if its a preset event */
	if ( check_derived_events(event_name, derived_type, results, &_papi_hwi_presets[0], PAPI_MAX_PRESET_EVENTS, token_index) ) {
		INTDBG("EXIT: found preset event\n");
		return 1;
	}

	/* check if its a user defined event */
	if ( check_derived_events(event_name, derived_type, results, user_defined_events, user_defined_events_count, token_index) ) {
		INTDBG("EXIT: found user event\n");
		return 1;
	}

	/* check if its a native event */
	if ( check_native_events(event_name, results) ) {
		INTDBG("EXIT: found native event\n");
		return 1;
	}

	INTDBG("EXIT: event not found\n");
	return 0;
}

/* Static version of the events file. */
#if defined(STATIC_PAPI_EVENTS_TABLE)
#include "papi_events_table.h"
#else
static char *papi_events_table = NULL;
#endif

int _papi_load_preset_table(char *pmu_str, int pmu_type, int cidx) {
	SUBDBG("ENTER: pmu_str: %s, pmu_type: %d, cidx: %d\n", pmu_str, pmu_type, cidx);

	int retval;

	// go load papi preset events (last argument tells function if we are loading presets or user events)
	retval = papi_load_derived_events(pmu_str, pmu_type, cidx, 1);
	if (retval != PAPI_OK) {
		SUBDBG("EXIT: retval: %d\n", retval);
		return retval;
	}

	// go load the user defined event definitions if any are defined
	retval = papi_load_derived_events(pmu_str, pmu_type, cidx, 0);

	SUBDBG("EXIT: retval: %d\n", retval);
	return retval;
}

// global variables
static char stack[2*PAPI_HUGE_STR_LEN]; // stack
static int stacktop = -1; // stack length

// priority: This function returns the priority of the operator
static
int priority( char symbol ) {
        switch( symbol ) {
        case '@':
                return -1;
        case '(':
                return 0;
	case '+':
	case '-':
                return 1;
	case '*':
	case '/':
	case '%':
		return 2;
	default :
		return 0;
	} // end switch symbol
} // end priority

static
int push( char symbol ) {
  if (stacktop >= 2*PAPI_HUGE_STR_LEN - 1) {
    INTDBG("stack overflow converting algebraic expression (%d,%c)\n", stacktop,symbol );
    return -1;  //***TODO: Figure out how to exit gracefully
  } // end if stacktop>MAX
  stack[++stacktop] = symbol;
  return 0;
} // end push

// pop from stack
static
char pop() {
  if( stacktop < 0 ) {
    INTDBG("stack underflow converting algebraic expression\n" );
    return '\0';  //***TODO: Figure out how to exit gracefully
  } // end if empty
  return( stack[stacktop--] );
} // end pop

/* infix_to_postfix:
   routine that will be called with parameter:
   char *in characters of infix notation (algebraic formula)
   returns: char * pointer to string of returned postfix */
static char *
infix_to_postfix( char *infix ) {
	INTDBG("ENTER: in: %s, size: %zu\n", infix, strlen(infix));
	static char postfix[2*PAPI_HUGE_STR_LEN];	// output
        unsigned int index;
        int postfixlen;
        char token;
        if ( strlen(infix) > PAPI_HUGE_STR_LEN ) 
            PAPIERROR("A infix string (probably in user-defined presets) is too big (max allowed %d): %s", PAPI_HUGE_STR_LEN, infix );

        // initialize stack
	memset(stack, 0, 2*PAPI_HUGE_STR_LEN);
	stacktop = -1; 
	push('#'); 
        stacktop = 0; // after initialization of stack to #
        /* initialize output string */
	memset(postfix,0,2*PAPI_HUGE_STR_LEN);
        postfixlen = 0;

	for( index=0; index<strlen(infix); index++ ) {
                token = infix[index];
                INTDBG("INTDBG: in: %s, length: %zu, index: %d token %c\n", infix, strlen( infix ), index, token);
		switch( token ) {
		case '(':
			push( token );
			break;
		case ')':
                        if (postfix[postfixlen-1]!='|') postfix[postfixlen++] = '|';
                        while ( stack[stacktop] != '(' ) {
                                postfix[postfixlen++] = pop();
                                postfix[postfixlen++] = '|';
                        }
                        token = pop();  /* pop the '(' character */
			break;
		case '+':
		case '-':
		case '*':
		case '/':
		case '%':
		case '^':       /* if an operator */
                        if (postfix[postfixlen-1]!='|') postfix[postfixlen++] = '|';
                        while ( priority(stack[stacktop]) > priority(token) ) {
                                postfix[postfixlen++] = pop();
                                postfix[postfixlen++] = '|';
                        }
                        push( token ); /* save current operator */
                        break;
		default: // if alphanumeric character which is not parenthesis or an operator
                        postfix[postfixlen++] = token;
			break;
		} // end switch symbol
	} // end while

        /* Write any remaining operators */
        if (postfix[postfixlen-1]!='|') postfix[postfixlen++] = '|';
        while ( stacktop>0 ) {
                postfix[postfixlen++] = pop();
                postfix[postfixlen++] = '|';
        }
        postfix[postfixlen++] = '\0';
	stacktop = -1; 

	INTDBG("EXIT: postfix: %s, size: %zu\n", postfix, strlen(postfix));
	return (postfix);
} // end infix_to_postfix

/*
 * This function will load event definitions from either a file or an in memory table.  It is used to load both preset events
 * which are defined by the PAPI development team and delivered with the product and user defined events which can be defined
 * by papi users and provided to papi to be processed at library initialization.  Both the preset events and user defined events
 * support the same event definition syntax.
 *
 * Event definition file syntax:
 * see PAPI_derived_event_files(1) man page.
 *
 * Blank lines are ignored
 * Lines that begin with '#' are comments.
 * Lines that begin with 'CPU' identify a pmu name and have the following effect.
 *      If this pmu name does not match the pmu_str passed in, it is ignored and we get the next input line.
 *      If this pmu name matches the pmu_str passed in, we set a 'process events' flag.
 *      Multiple consecutive 'CPU' lines may be provided and if any of them match the pmu_str passed in, we set a 'process events' flag.
 *      When a 'CPU' line is found following event definition lines, it turns off the 'process events' flag and then does the above checks.
 * Lines that begin with 'PRESET' or 'EVENT' specify an event definition and are processed as follows.
 *      If the 'process events' flag is not set, the line is ignored and we get the next input line.
 *      If the 'process events' flag is set, the event is processed and the event information is put into the next slot in the results array.
 *
 * There are three possible sources of input for preset event definitions.  The code will first look for the environment variable
 * "PAPI_CSV_EVENT_FILE".  If found its value will be used as the pathname of where to get the preset information.  If not found,
 * the code will look for a built in table containing preset events.  If the built in table was not created during the build of
 * PAPI then the code will build a pathname of the form "PAPI_DATADIR/PAPI_EVENT_FILE".  Each of these are build variables, the
 * PAPI_DATADIR variable can be given a value during the configure of PAPI at build time, and the PAPI_EVENT_FILE variable has a
 * hard coded value of "papi_events.csv".
 *
 * There is only one way to define user events.  The code will look for an environment variable "PAPI_USER_EVENTS_FILE".  If found
 * its value will be used as the pathname of a file which contains user event definitions.  The events defined in this file will be
 * added to the ones known by PAPI when the call to PAPI_library_init is done.
 *
 * TODO:
 * Look into restoring the ability to specify a user defined event file with a call to PAPI_set_opt(PAPI_USER_EVENTS_FILE).
 * This needs to figure out how to pass a pmu name (could use default pmu from component 0) to this function.
 *
 * Currently code elsewhere in PAPI limits the events which preset and user events can depend on to those events which are known to component 0.  This possibly could
 * be relaxed to allow events from different components.  But since all the events used by any derived event must be added to the same eventset, it will always be a
 * requirement that all events used by a given derived event must be from the same component.
 *
 */


static int
papi_load_derived_events (char *pmu_str, int pmu_type, int cidx, int preset_flag) {
	SUBDBG( "ENTER: pmu_str: %s, pmu_type: %d, cidx: %d, preset_flag: %d\n", pmu_str, pmu_type, cidx, preset_flag);

	char pmu_name[PAPI_MIN_STR_LEN];
	char line[LINE_MAX];
	char name[PATH_MAX] = "builtin papi_events_table";
	char *event_file_path=NULL;
	char *event_table_ptr=NULL;
	int event_type_bits = 0;
	char *tmpn;
	char *tok_save_ptr=NULL;
	FILE *event_file = NULL;
	hwi_presets_t *results=NULL;
	int result_size = 0;
	int *event_count = NULL;
	int invalid_event;
	int line_no = 0;  /* count of lines read from event definition input */
	int derived = 0;
	int res_idx = 0;  /* index into results array for where to store next event */
	int preset = 0;
	int get_events = 0; /* only process derived events after CPU type they apply to is identified      */
	int found_events = 0; /* flag to track if event definitions (PRESETS) are found since last CPU declaration */
#ifdef PAPI_DATADIR
		char path[PATH_MAX];
#endif


	if (preset_flag) {
		/* try the environment variable first */
		if ((tmpn = getenv("PAPI_CSV_EVENT_FILE")) && (strlen(tmpn) > 0)) {
			event_file_path = tmpn;
		}
		/* if no valid environment variable, look for built-in table */
		else if (papi_events_table) {
			event_table_ptr = papi_events_table;
		}
		/* if no env var and no built-in, search for default file */
		else {
#ifdef PAPI_DATADIR
			sprintf( path, "%s/%s", PAPI_DATADIR, PAPI_EVENT_FILE );
			event_file_path = path;
#else
			event_file_path = PAPI_EVENT_FILE;
#endif
		}
		event_type_bits = PAPI_PRESET_MASK;
		results = &_papi_hwi_presets[0];
		result_size = PAPI_MAX_PRESET_EVENTS;
		event_count = &_papi_hwd[cidx]->cmp_info.num_preset_events;
	} else {
		if ((event_file_path = getenv( "PAPI_USER_EVENTS_FILE" )) == NULL ) {
			SUBDBG("EXIT: User event definition file not provided.\n");
			return PAPI_OK;
		}

		event_type_bits = PAPI_UE_MASK;
		results = &user_defined_events[0];
		result_size = PAPI_MAX_USER_EVENTS;
		event_count = &user_defined_events_count;
	}

	// if we have an event file pathname, open it and read event definitions from the file
	if (event_file_path != NULL) {
		if ((event_file = open_event_table(event_file_path)) == NULL) {
			// if file open fails, return an error
			SUBDBG("EXIT: Event file open failed.\n");
			return PAPI_ESYS;
		}
		strncpy(name, event_file_path, sizeof(name)-1);
		name[sizeof(name)-1] = '\0';
	} else if (event_table_ptr == NULL) {
		// if we do not have a path name or table pointer, return an error
		SUBDBG("EXIT: Both event_file_path and event_table_ptr are NULL.\n");
		return PAPI_ESYS;
	}

	/* copy the pmu identifier, stripping commas if found */
	tmpn = pmu_name;
	while (*pmu_str) {
		if (*pmu_str != ',')
			*tmpn++ = *pmu_str;
		pmu_str++;
	}
	*tmpn = '\0';

	/* at this point we have either a valid file pointer or built-in table pointer */
	while (get_event_line(line, event_file, &event_table_ptr)) {
		char *t;
		int i;

		// increment number of lines we have read
		line_no++;

		t = trim_string(strtok_r(line, ",", &tok_save_ptr));

		/* Skip blank lines */
		if ((t == NULL) || (strlen(t) == 0))
			continue;

		/* Skip comments */
		if (t[0] == '#') {
			continue;
		}

		if (strcasecmp(t, "CPU") == 0) {
			if (get_events != 0 && found_events != 0) {
				SUBDBG( "Ending event scanning at line %d of %s.\n", line_no, name);
				get_events = 0;
				found_events = 0;
			}

			t = trim_string(strtok_r(NULL, ",", &tok_save_ptr));
			if ((t == NULL) || (strlen(t) == 0)) {
				PAPIERROR("Expected name after CPU token at line %d of %s -- ignoring", line_no, name);
				continue;
			}

			if (strcasecmp(t, pmu_name) == 0) {
				int type;

				SUBDBG( "Process events for PMU %s found at line %d of %s.\n", t, line_no, name);

				t = trim_string(strtok_r(NULL, ",", &tok_save_ptr));
				if ((t == NULL) || (strlen(t) == 0)) {
					SUBDBG("No additional qualifier found, matching on string.\n");
					get_events = 1;
				} else if ((sscanf(t, "%d", &type) == 1) && (type == pmu_type)) {
					SUBDBG( "Found CPU %s type %d at line %d of %s.\n", pmu_name, type, line_no, name);
					get_events = 1;
				} else {
					SUBDBG( "Additional qualifier match failed %d vs %d.\n", pmu_type, type);
				}
			}
			continue;
		}

		if ((strcasecmp(t, "PRESET") == 0)  || (strcasecmp(t, "EVENT") == 0)) {

			if (get_events == 0)
				continue;

			found_events = 1;
			t = trim_string(strtok_r(NULL, ",", &tok_save_ptr));

			if ((t == NULL) || (strlen(t) == 0)) {
				PAPIERROR("Expected name after PRESET token at line %d of %s -- ignoring", line_no, name);
				continue;
			}

			SUBDBG( "Examining event %s\n", t);

			// see if this event already exists in the results array, if not already known it sets up event in unused entry
			if ((res_idx = find_event_index (results, result_size, t)) < 0) {
				PAPIERROR("No room left for event %s -- ignoring", t);
				continue;
			}

			// add the proper event bits (preset or user defined bits)
			preset = res_idx | event_type_bits;
			(void) preset;

			SUBDBG( "Use event code: %#x for %s\n", preset, t);

			t = trim_string(strtok_r(NULL, ",", &tok_save_ptr));
			if ((t == NULL) || (strlen(t) == 0)) {
				// got an error, make this entry unused
                if (results[res_idx].symbol != NULL){
                    papi_free (results[res_idx].symbol);
                    results[res_idx].symbol = NULL;
                }
				PAPIERROR("Expected derived type after PRESET token at line %d of %s -- ignoring", line_no, name);
				continue;
			}

			if (_papi_hwi_derived_type(t, &derived) != PAPI_OK) {
				// got an error, make this entry unused
                if (results[res_idx].symbol != NULL){
                    papi_free (results[res_idx].symbol);
                    results[res_idx].symbol = NULL;
                }
				PAPIERROR("Invalid derived name %s after PRESET token at line %d of %s -- ignoring", t, line_no, name);
				continue;
			}

			/****************************************/
			/* Have an event, let's start assigning */
			/****************************************/

			SUBDBG( "Adding event: %s, code: %#x, derived: %d results[%d]: %p.\n", t, preset, derived, res_idx, &results[res_idx]);

			/* results[res_idx].event_code = preset; */
			results[res_idx].derived_int = derived;

			/* Derived support starts here */
			/* Special handling for postfix and infix */
			if ((derived == DERIVED_POSTFIX)  || (derived == DERIVED_INFIX)) {
				t = trim_string(strtok_r(NULL, ",", &tok_save_ptr));
				if ((t == NULL) || (strlen(t) == 0)) {
					// got an error, make this entry unused
                    if (results[res_idx].symbol != NULL){
                        papi_free (results[res_idx].symbol);
                        results[res_idx].symbol = NULL;
                    }
					PAPIERROR("Expected Operation string after derived type DERIVED_POSTFIX or DERIVED_INFIX at line %d of %s -- ignoring", line_no, name);
					continue;
				}

				// if it is an algebraic formula, we need to convert it to postfix
				if (derived == DERIVED_INFIX) {
					SUBDBG( "Converting InFix operations %s\n", t);
					t = infix_to_postfix( t );
					results[res_idx].derived_int = DERIVED_POSTFIX;
				}

				SUBDBG( "Saving PostFix operations %s\n", t);
				results[res_idx].postfix = papi_strdup(t);
			}

			/* All derived terms collected here */
			i = 0;
			invalid_event = 0;
			results[res_idx].count = 0;
			do {
				t = trim_string(strtok_r(NULL, ",", &tok_save_ptr));
				if ((t == NULL) || (strlen(t) == 0))
					break;
				if (strcasecmp(t, "NOTE") == 0)
					break;
				if (strcasecmp(t, "LDESC") == 0)
					break;
				if (strcasecmp(t, "SDESC") == 0)
					break;

				SUBDBG( "Adding term (%d) %s to derived event %#x, current native event count: %d.\n", i, t, preset, results[res_idx].count);

				// show that we do not have an event code yet (the component may create one and update this info)
				// this also clears any values left over from a previous call
				_papi_hwi_set_papi_event_code(-1, -1);

				// make sure that this term in the derived event is a valid event name
				// this call replaces preset and user event names with the equivalent native events in our results table
				// it also updates formulas for derived events so that they refer to the correct native event index
				if (is_event(t, results[res_idx].derived_int, &results[res_idx], i) == 0) {
					invalid_event = 1;
					PAPIERROR("Missing event %s, used in derived event %s", t, results[res_idx].symbol);
					break;
				}

				i++;
			} while (results[res_idx].count < PAPI_EVENTS_IN_DERIVED_EVENT);

			/* preset code list must be PAPI_NULL terminated */
			if (i < PAPI_EVENTS_IN_DERIVED_EVENT) {
				results[res_idx].code[results[res_idx].count] = PAPI_NULL;
			}

			if (invalid_event) {
				// got an error, make this entry unused
			        // preset table is statically allocated, user defined is dynamic
                unsigned int j;
                for (j = 0; j < results[res_idx].count; j++){
                    if (results[res_idx].name[j] != NULL){
                        papi_free( results[res_idx].name[j] );
                        results[res_idx].name[j] = NULL;
                    }
                }

                if (!preset_flag){
                    if(results[res_idx].symbol != NULL){
                        papi_free (results[res_idx].symbol);
                        results[res_idx].symbol = NULL;
                    }
                }

				continue;
			}

			/* End of derived support */

			// if we did not find any terms to base this derived event on, report error
			if (i == 0) {
				// got an error, make this entry unused
                if (!preset_flag){
                    if(results[res_idx].symbol != NULL){
                        papi_free (results[res_idx].symbol);
                        results[res_idx].symbol = NULL;
                    }
                }
                PAPIERROR("Expected PFM event after DERIVED token at line %d of %s -- ignoring", line_no, name);
				continue;
			}

			if (i == PAPI_EVENTS_IN_DERIVED_EVENT) {
				t = trim_string(strtok_r(NULL, ",", &tok_save_ptr));
			}

			// if something was provided following the list of events to be used by the operation, process it
			if ( t!= NULL  && strlen(t) > 0 ) {
				do {
					// save the field name
					char *fptr = papi_strdup(t);

					// get the value to be used with this field
					t = trim_note(strtok_r(NULL, ",", &tok_save_ptr));
					if ( t== NULL  || strlen(t) == 0 ) {
						papi_free(fptr);
						break;
					}

					// Handle optional short descriptions, long descriptions and notes
					if (strcasecmp(fptr, "SDESC") == 0) {
						results[res_idx].short_descr = papi_strdup(t);
					}
					if (strcasecmp(fptr, "LDESC") == 0) {
						results[res_idx].long_descr = papi_strdup(t);
					}
					if (strcasecmp(fptr, "NOTE") == 0) {
						results[res_idx].note = papi_strdup(t);
					}

					SUBDBG( "Found %s (%s) on line %d\n", fptr, t, line_no);
					papi_free (fptr);

					// look for another field name
					t = trim_string(strtok_r(NULL, ",", &tok_save_ptr));
					if ( t== NULL  || strlen(t) == 0 ) {
						break;
					}
				} while (t != NULL);
			}
			(*event_count)++;
			continue;
		}

		PAPIERROR("Unrecognized token %s at line %d of %s -- ignoring", t, line_no, name);
	}

	if (event_file) {
		fclose(event_file);
	}

	SUBDBG("EXIT: Done processing derived event file.\n");
	return PAPI_OK;
}




/* The following code is proof of principle for reading preset events from an
   xml file. It has been tested and works for pentium3. It relys on the expat
   library and is invoked by adding
   XMLFLAG		= -DXML
   to the Makefile. It is presently hardcoded to look for "./papi_events.xml"
*/
#ifdef XML

#define BUFFSIZE 8192
#define SPARSE_BEGIN 0
#define SPARSE_EVENT_SEARCH 1
#define SPARSE_EVENT 2
#define SPARSE_DESC 3
#define ARCH_SEARCH 4
#define DENSE_EVENT_SEARCH 5
#define DENSE_NATIVE_SEARCH 6
#define DENSE_NATIVE_DESC 7
#define FINISHED 8

char buffer[BUFFSIZE], *xml_arch;
int location = SPARSE_BEGIN, sparse_index = 0, native_index, error = 0;

/* The function below, _xml_start(), is a hook into expat's XML
 * parser.  _xml_start() defines how the parser handles the
 * opening tags in PAPI's XML file.  This function can be understood
 * more easily if you follow along with its logic while looking at
 * papi_events.xml.  The location variable is a global telling us
 * where we are in the XML file.  Have we found our architecture's
 * events yet?  Are we looking at an event definition?...etc.
 */
static void
_xml_start( void *data, const char *el, const char **attr )
{
	int native_encoding;

	if ( location == SPARSE_BEGIN && !strcmp( "papistdevents", el ) ) {
		location = SPARSE_EVENT_SEARCH;
	} else if ( location == SPARSE_EVENT_SEARCH && !strcmp( "papievent", el ) ) {
		_papi_hwi_presets[sparse_index].info.symbol = papi_strdup( attr[1] );
//      strcpy(_papi_hwi_presets.info[sparse_index].symbol, attr[1]);
		location = SPARSE_EVENT;
	} else if ( location == SPARSE_EVENT && !strcmp( "desc", el ) ) {
		location = SPARSE_DESC;
	} else if ( location == ARCH_SEARCH && !strcmp( "availevents", el ) &&
				!strcmp( xml_arch, attr[1] ) ) {
		location = DENSE_EVENT_SEARCH;
	} else if ( location == DENSE_EVENT_SEARCH && !strcmp( "papievent", el ) ) {
		if ( !strcmp( "PAPI_NULL", attr[1] ) ) {
			location = FINISHED;
			return;
		} else if ( PAPI_event_name_to_code( ( char * ) attr[1], &sparse_index )
					!= PAPI_OK ) {
			PAPIERROR( "Improper Preset name given in XML file for %s.",
					   attr[1] );
			error = 1;
		}
		sparse_index &= PAPI_PRESET_AND_MASK;

		/* allocate and initialize data space for this event */
		papi_valid_free( _papi_hwi_presets[sparse_index].data );
		_papi_hwi_presets[sparse_index].data =
			papi_malloc( sizeof ( hwi_preset_data_t ) );
		native_index = 0;
		_papi_hwi_presets[sparse_index].data->native[native_index] = PAPI_NULL;
		_papi_hwi_presets[sparse_index].data->operation[0] = '\0';


		if ( attr[2] ) {	 /* derived event */
			_papi_hwi_presets[sparse_index].data->derived =
				_papi_hwi_derived_type( ( char * ) attr[3] );
			/* where does DERIVED POSTSCRIPT get encoded?? */
			if ( _papi_hwi_presets[sparse_index].data->derived == -1 ) {
				PAPIERROR( "No derived type match for %s in Preset XML file.",
						   attr[3] );
				error = 1;
			}

			if ( attr[5] ) {
				_papi_hwi_presets[sparse_index].count = atoi( attr[5] );
			} else {
				PAPIERROR( "No count given for %s in Preset XML file.",
						   attr[1] );
				error = 1;
			}
		} else {
			_papi_hwi_presets[sparse_index].data->derived = NOT_DERIVED;
			_papi_hwi_presets[sparse_index].count = 1;
		}
		location = DENSE_NATIVE_SEARCH;
	} else if ( location == DENSE_NATIVE_SEARCH && !strcmp( "native", el ) ) {
		location = DENSE_NATIVE_DESC;
	} else if ( location == DENSE_NATIVE_DESC && !strcmp( "event", el ) ) {
		if ( _papi_hwi_native_name_to_code( attr[1], &native_encoding ) !=
			 PAPI_OK ) {
			printf( "Improper Native name given in XML file for %s\n",
					attr[1] );
			PAPIERROR( "Improper Native name given in XML file for %s",
					   attr[1] );
			error = 1;
		}
		_papi_hwi_presets[sparse_index].data->native[native_index] =
			native_encoding;
		native_index++;
		_papi_hwi_presets[sparse_index].data->native[native_index] = PAPI_NULL;
	} else if ( location && location != ARCH_SEARCH && location != FINISHED ) {
		PAPIERROR( "Poorly-formed Preset XML document." );
		error = 1;
	}
}

/* The function below, _xml_end(), is a hook into expat's XML
 * parser.  _xml_end() defines how the parser handles the
 * end tags in PAPI's XML file.
 */
static void
_xml_end( void *data, const char *el )
{
	int i;

	if ( location == SPARSE_EVENT_SEARCH && !strcmp( "papistdevents", el ) ) {
		for ( i = sparse_index; i < PAPI_MAX_PRESET_EVENTS; i++ ) {
			_papi_hwi_presets[i].info.symbol = NULL;
			_papi_hwi_presets[i].info.long_descr = NULL;
			_papi_hwi_presets[i].info.short_descr = NULL;
		}
		location = ARCH_SEARCH;
	} else if ( location == DENSE_NATIVE_DESC && !strcmp( "native", el ) ) {
		location = DENSE_EVENT_SEARCH;
	} else if ( location == DENSE_EVENT_SEARCH && !strcmp( "availevents", el ) ) {
		location = FINISHED;
	}
}

/* The function below, _xml_content(), is a hook into expat's XML
 * parser.  _xml_content() defines how the parser handles the
 * text between tags in PAPI's XML file.  The information between
 * tags is usally text for event descriptions.
 */
static void
_xml_content( void *data, const char *el, const int len )
{
	int i;
	if ( location == SPARSE_DESC ) {
		_papi_hwi_presets[sparse_index].info.long_descr =
			papi_malloc( len + 1 );
		for ( i = 0; i < len; i++ )
			_papi_hwi_presets[sparse_index].info.long_descr[i] = el[i];
		_papi_hwi_presets[sparse_index].info.long_descr[len] = '\0';
		/* the XML data currently doesn't contain a short description */
		_papi_hwi_presets[sparse_index].info.short_descr = NULL;
		sparse_index++;
		_papi_hwi_presets[sparse_index].data = NULL;
		location = SPARSE_EVENT_SEARCH;
	}
}

int
_xml_papi_hwi_setup_all_presets( char *arch, hwi_dev_notes_t * notes )
{
	int done = 0;
	FILE *fp = fopen( "./papi_events.xml", "r" );
	XML_Parser p = XML_ParserCreate( NULL );

	if ( !p ) {
		PAPIERROR( "Couldn't allocate memory for XML parser." );
		fclose(fp);
		return ( PAPI_ESYS );
	}
	XML_SetElementHandler( p, _xml_start, _xml_end );
	XML_SetCharacterDataHandler( p, _xml_content );
	if ( fp == NULL ) {
		PAPIERROR( "Error opening Preset XML file." );
		fclose(fp);
		return ( PAPI_ESYS );
	}

	xml_arch = arch;

	do {
		int len;
		void *buffer = XML_GetBuffer( p, BUFFSIZE );

		if ( buffer == NULL ) {
			PAPIERROR( "Couldn't allocate memory for XML buffer." );
			fclose(fp);
			return ( PAPI_ESYS );
		}
		len = fread( buffer, 1, BUFFSIZE, fp );
		if ( ferror( fp ) ) {
			PAPIERROR( "XML read error." );
			fclose(fp);
			return ( PAPI_ESYS );
		}
		done = feof( fp );
		if ( !XML_ParseBuffer( p, len, len == 0 ) ) {
			PAPIERROR( "Parse error at line %d:\n%s",
					   XML_GetCurrentLineNumber( p ),
					   XML_ErrorString( XML_GetErrorCode( p ) ) );
			fclose(fp);
			return ( PAPI_ESYS );
		}
		if ( error ) {
			fclose(fp);
			return ( PAPI_ESYS );
		}
	} while ( !done );
	XML_ParserFree( p );
	fclose( fp );
	return ( PAPI_OK );
}
#endif
