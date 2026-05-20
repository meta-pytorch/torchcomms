/****************************/
/* THIS IS OPEN SOURCE CODE */
/****************************/

#ifndef __PFMLIB_POWER_PRIV_H__
#define __PFMLIB_POWER_PRIV_H__

/*
* File:    pfmlib_power_priv.h
* CVS:
* Author:  Corey Ashford
*          cjashfor@us.ibm.com
* Mods:    <your name here>
*          <your email address>
*
* (C) Copyright IBM Corporation, 2007.  All Rights Reserved.
* Contributed by Corey Ashford <cjashfor.ibm.com>
*
* Note: This code was automatically generated and should not be modified by
* hand.
*
*/
typedef struct {
   char *pme_name;
   unsigned pme_code;
   char *pme_short_desc;
   char *pme_long_desc;
   const int *pme_event_ids;
   const unsigned long long *pme_group_vector;
} pme_power_entry_t;

typedef struct {
   char *pmg_name;
   char *pmg_desc;
   const int *pmg_event_ids;
   unsigned long long pmg_mmcr0;
   unsigned long long pmg_mmcr1;
   unsigned long long pmg_mmcra;
} pmg_power_group_t;


#endif

