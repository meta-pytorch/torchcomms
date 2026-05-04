#
# Copyright (c) 2002-2006 Hewlett-Packard Development Company, L.P.
# Contributed by Stephane Eranian <eranian@hpl.hp.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies 
# of the Software, and to permit persons to whom the Software is furnished to do so, 
# subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.  
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A 
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT 
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF 
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE 
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#

TOPDIR  := $(shell if [ "$$PWD" != "" ]; then echo $$PWD; else pwd; fi)/..

include $(TOPDIR)/config.mk
include $(TOPDIR)/rules.mk

ifeq ($(CONFIG_PFMLIB_ARCH_IA64),y)
ARCH_MAN=libpfm_itanium.3 libpfm_itanium2.3 libpfm_montecito.3
endif

ifeq ($(CONFIG_PFMLIB_ARCH_I386),y)
ARCH_MAN=libpfm_p6.3 libpfm_core.3 libpfm_amd64.3 libpfm_atom.3 libpfm_nehalem.3
endif

ifeq ($(CONFIG_PFMLIB_ARCH_X86_64),y)
ARCH_MAN=libpfm_amd64.3 libpfm_core.3 libpfm_atom.3 libpfm_nehalem.3
endif

ifeq ($(CONFIG_PFMLIB_ARCH_MIPS64),y)
endif

ifeq ($(CONFIG_PFMLIB_ARCH_POWERPC),y)
ARCH_MAN=libpfm_powerpc.3
endif

ifeq ($(CONFIG_PFMLIB_ARCH_CRAYXT),y)
endif

ifeq ($(CONFIG_PFMLIB_CELL),y)
endif

GEN_MAN= libpfm.3 pfm_dispatch_events.3 pfm_find_event.3 pfm_find_event_bycode.3 \
         pfm_find_event_bycode_next.3 pfm_find_event_mask.3 pfm_find_full_event.3 \
         pfm_force_pmu.3 pfm_get_cycle_event.3 pfm_get_event_code.3 pfm_get_event_code_counter.3 \
         pfm_get_event_counters.3 pfm_get_event_description.3 pfm_get_event_mask_code.3 \
         pfm_get_event_mask_description.3 pfm_get_event_mask_name.3 pfm_get_event_name.3 \
         pfm_get_full_event_name.3 pfm_get_hw_counter_width.3 pfm_get_impl_counters.3 \
         pfm_get_impl_pmcs.3 pfm_get_impl_pmds.3 pfm_get_inst_retired.3 pfm_get_max_event_name_len.3 \
         pfm_get_num_counters.3 pfm_get_num_events.3 pfm_get_num_pmcs.3 \
         pfm_get_num_pmds.3 pfm_get_pmu_name.3 pfm_get_pmu_name_bytype.3 \
         pfm_get_pmu_type.3 pfm_get_version.3 pfm_initialize.3 \
         pfm_list_supported_pmus.3 pfm_pmu_is_supported.3 pfm_regmask_and.3 \
         pfm_regmask_clr.3 pfm_regmask_copy.3 pfm_regmask_eq.3 pfm_regmask_isset.3 \
         pfm_regmask_or.3 pfm_regmask_set.3 pfm_regmask_weight.3 pfm_set_options.3 \
         pfm_strerror.3

MAN=$(GEN_MAN) $(ARCH_MAN)
install:

	-mkdir -p $(DESTDIR)$(MANDIR)/man3
	( cd man3; $(INSTALL) -m 644 $(MAN) $(DESTDIR)$(MANDIR)/man3 )
