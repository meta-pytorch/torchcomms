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

# perfmon/perfmon.h is installed below
HEADERS=perfmon/perfmon_dfl_smpl.h \
	perfmon/pfmlib.h           \
	perfmon/perfmon_v2.h	   \
	perfmon/pfmlib_comp.h      \
	perfmon/pfmlib_os.h

ifeq ($(CONFIG_PFMLIB_ARCH_IA64),y)
HEADERS += perfmon/pfmlib_os_ia64.h       \
	   perfmon/pfmlib_comp_ia64.h     \
	   perfmon/perfmon_ia64.h         \
	   perfmon/perfmon_compat.h       \
	   perfmon/perfmon_default_smpl.h \
	   perfmon/pfmlib_itanium.h	  \
	   perfmon/pfmlib_itanium2.h	  \
	   perfmon/pfmlib_montecito.h	  \
	   perfmon/pfmlib_gen_ia64.h
endif

ifeq ($(CONFIG_PFMLIB_ARCH_X86_64),y)
HEADERS += perfmon/pfmlib_os_x86_64.h   \
	   perfmon/pfmlib_os_i386.h   	\
	   perfmon/pfmlib_comp_x86_64.h \
	   perfmon/pfmlib_comp_i386.h 	\
	   perfmon/perfmon_x86_64.h     \
	   perfmon/perfmon_i386.h       \
	   perfmon/pfmlib_i386_p6.h	\
	   perfmon/perfmon_pebs_p4_smpl.h  \
	   perfmon/perfmon_pebs_core_smpl.h  \
	   perfmon/perfmon_pebs_smpl.h  \
	   perfmon/pfmlib_amd64.h	\
	   perfmon/pfmlib_pentium4.h	\
	   perfmon/pfmlib_core.h	\
	   perfmon/pfmlib_coreduo.h	\
	   perfmon/pfmlib_intel_atom.h	\
	   perfmon/pfmlib_intel_nhm.h	\
	   perfmon/pfmlib_gen_ia32.h

endif

ifeq ($(CONFIG_PFMLIB_ARCH_I386),y)
HEADERS += perfmon/pfmlib_os_i386.h   	\
	   perfmon/pfmlib_comp_i386.h 	\
	   perfmon/perfmon_i386.h     	\
	   perfmon/perfmon_pebs_p4_smpl.h \
	   perfmon/perfmon_pebs_core_smpl.h \
	   perfmon/perfmon_pebs_smpl.h  \
	   perfmon/pfmlib_amd64.h       \
	   perfmon/pfmlib_pentium4.h    \
	   perfmon/pfmlib_core.h        \
	   perfmon/pfmlib_coreduo.h	\
	   perfmon/pfmlib_intel_atom.h	\
	   perfmon/pfmlib_intel_nhm.h	\
	   perfmon/pfmlib_i386_p6.h	\
	   perfmon/pfmlib_gen_ia32.h

endif

ifeq ($(CONFIG_PFMLIB_ARCH_POWERPC),y)
HEADERS += perfmon/pfmlib_cell.h	 \
	   perfmon/pfmlib_os_powerpc.h   \
	   perfmon/pfmlib_comp_powerpc.h \
	   perfmon/perfmon_powerpc.h     \
	   perfmon/pfmlib_powerpc.h
endif

ifeq ($(CONFIG_PFMLIB_ARCH_SPARC),y)
HEADERS += perfmon/pfmlib_os_sparc.h   \
	   perfmon/pfmlib_comp_sparc.h \
	   perfmon/perfmon_sparc.h     \
	   perfmon/pfmlib_sparc.h
endif

ifeq ($(CONFIG_PFMLIB_ARCH_MIPS64),y)
HEADERS += perfmon/pfmlib_os_mips64.h   \
	   perfmon/pfmlib_comp_mips64.h \
	   perfmon/perfmon_mips64.h     \
	   perfmon/pfmlib_gen_mips64.h  \
	   perfmon/pfmlib_sicortex.h
endif

ifeq ($(CONFIG_PFMLIB_ARCH_CRAYX2),y)
HEADERS += perfmon/pfmlib_os_crayx2.h   \
	   perfmon/pfmlib_comp_crayx2.h \
	   perfmon/perfmon_crayx2.h     \
	   perfmon/pfmlib_crayx2.h
endif

.PHONY:	perfmon.h dir

perfmon.h: dir

perfmon.h:
ifeq ($(CONFIG_PFMLIB_OLD_PFMV2),y)
	echo "#ifndef PFMLIB_OLD_PFMV2" > $(DESTDIR)$(INCDIR)/perfmon/perfmon.h
	echo "#define PFMLIB_OLD_PFMV2" >> $(DESTDIR)$(INCDIR)/perfmon/perfmon.h
	echo "#endif" >> $(DESTDIR)$(INCDIR)/perfmon/perfmon.h
	cat perfmon/perfmon.h >> $(DESTDIR)$(INCDIR)/perfmon/perfmon.h
	chmod 644 $(DESTDIR)$(INCDIR)/perfmon/perfmon.h
else
	$(INSTALL) -m 644 perfmon/perfmon.h $(DESTDIR)$(INCDIR)/perfmon
endif

dir:
	mkdir -p $(DESTDIR)$(INCDIR)/perfmon

install: dir perfmon.h $(HEADERS)

install:
	$(INSTALL) -m 644 $(HEADERS) $(DESTDIR)$(INCDIR)/perfmon
