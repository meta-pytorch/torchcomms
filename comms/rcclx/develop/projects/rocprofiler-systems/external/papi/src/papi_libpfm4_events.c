/*
* File:    papi_libpfm4_events.c
* Author:  Vince Weaver vincent.weaver @ maine.edu
*          based heavily on existing papi_libpfm3_events.c
*/

#include <string.h>

#include "papi.h"
#include "papi_internal.h"
#include "papi_vector.h"

#include "papi_libpfm4_events.h"

#include "perfmon/pfmlib.h"
#include "perfmon/pfmlib_perf_event.h"

/**********************************************************/
/* Local scope globals                                    */
/**********************************************************/

static int libpfm4_users=0;

/***********************************************************/
/* Exported functions                                      */
/***********************************************************/

/** @class  _papi_libpfm4_error
 *  @brief  convert libpfm error codes to PAPI error codes
 *
 *  @param[in] pfm_error
 *             -- a libpfm4 error code
 *
 *  @returns returns a PAPI error code
 *
 */

int
_papi_libpfm4_error( int pfm_error ) {

  switch ( pfm_error ) {
  case PFM_SUCCESS:      return PAPI_OK;       /* success */
  case PFM_ERR_NOTSUPP:  return PAPI_ENOSUPP;  /* function not supported */
  case PFM_ERR_INVAL:    return PAPI_EINVAL;   /* invalid parameters */
  case PFM_ERR_NOINIT:   return PAPI_ENOINIT;  /* library not initialized */
  case PFM_ERR_NOTFOUND: return PAPI_ENOEVNT;  /* event not found */
  case PFM_ERR_FEATCOMB: return PAPI_ECOMBO;   /* invalid combination of features */
  case PFM_ERR_UMASK:    return PAPI_EATTR;    /* invalid or missing unit mask */
  case PFM_ERR_NOMEM:    return PAPI_ENOMEM;   /* out of memory */
  case PFM_ERR_ATTR:     return PAPI_EATTR;    /* invalid event attribute */
  case PFM_ERR_ATTR_VAL: return PAPI_EATTR;    /* invalid event attribute value */
  case PFM_ERR_ATTR_SET: return PAPI_EATTR;    /* attribute value already set */
  case PFM_ERR_TOOMANY:  return PAPI_ECOUNT;   /* too many parameters */
  case PFM_ERR_TOOSMALL: return PAPI_ECOUNT;   /* parameter is too small */
  default:
	PAPIWARN("Unknown libpfm error code %d, returning PAPI_EINVAL",pfm_error);
	return PAPI_EINVAL;
  }
}

/** @class  _papi_libpfm4_shutdown
 *  @brief  Shutdown any initialization done by the libpfm4 code
 *
 *  @param[in] component
 *        -- component doing the shutdown
 *
 *  @retval PAPI_OK       Success
 *
 */

int
_papi_libpfm4_shutdown(papi_vector_t *my_vector) {

	/* clean out and free the native events structure */
	_papi_hwi_lock( NAMELIB_LOCK );

	libpfm4_users--;

	/* Only free if we're the last user */

	if (!libpfm4_users) {
		pfm_terminate();
	}

	_papi_hwi_unlock( NAMELIB_LOCK );

	strcpy(my_vector->cmp_info.support_version,"");

	return PAPI_OK;
}

/** @class  _papi_libpfm4_init
 *  @brief  Initialize the libpfm4 code
 *
 *  @param[in] my_vector
 *        -- vector of the component doing the initialization
 *
 *  @retval PAPI_OK       Success
 *  @retval PAPI_ECMP     There was an error initializing
 *
 */

int
_papi_libpfm4_init(papi_vector_t *my_vector) {

	int version;
	pfm_err_t retval = PFM_SUCCESS;

	_papi_hwi_lock( NAMELIB_LOCK );

	if (!libpfm4_users) {
		retval = pfm_initialize();
		if ( retval == PFM_SUCCESS ) {
			libpfm4_users++;
		}
		else {
			strncpy(my_vector->cmp_info.disabled_reason,
				pfm_strerror(retval),PAPI_MAX_STR_LEN-1);
			_papi_hwi_unlock( NAMELIB_LOCK );
			return PAPI_ESBSTR;
		}
	}
	else {
		libpfm4_users++;
	}

	_papi_hwi_unlock( NAMELIB_LOCK );

	/* get the libpfm4 version */

	version=pfm_get_version( );
	if (version >= 0) {

		/* Complain if the compiled-against version */
		/* doesn't match current version            */

		if ( PFM_MAJ_VERSION( version ) !=
			PFM_MAJ_VERSION( LIBPFM_VERSION ) ) {

			PAPIWARN( "Version mismatch of libpfm: "
				"compiled %#x vs. installed %#x\n",
				PFM_MAJ_VERSION( LIBPFM_VERSION ),
				PFM_MAJ_VERSION( version ) );

		}

		/* Set the version */
		sprintf( my_vector->cmp_info.support_version, "%d.%d",
			PFM_MAJ_VERSION( version ),
			PFM_MIN_VERSION( version ) );

	} else {
		PAPIWARN( "pfm_get_version(): %s", pfm_strerror( retval ) );
	}

	return PAPI_OK;
}
