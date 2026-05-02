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

#
# Common files
#
SRCS=pfmlib_common.c pfmlib_priv.c

ifeq ($(SYS),Linux)
SRCS += pfmlib_os_linux.c pfmlib_os_linux_v2.c

ifneq ($(CONFIG_PFMLIB_OLD_PFMV2),y)
SRCS += pfmlib_os_linux_v3.c
endif

endif

CFLAGS+=-D_REENTRANT

#
# list all library support modules
#
ifeq ($(CONFIG_PFMLIB_ARCH_X86_64),y)
INCARCH = $(INC_X86_64)
SRCS   += pfmlib_pentium4.c pfmlib_amd64.c pfmlib_core.c pfmlib_gen_ia32.c pfmlib_intel_atom.c \
          pfmlib_intel_nhm.c
CFLAGS += -DCONFIG_PFMLIB_ARCH_X86_64
endif


ifeq ($(SYS),Linux)
SLDFLAGS=-shared -Wl,-soname -Wl,$(VLIBPFM)
SLIBPFM=libpfm.so.$(VERSION).$(REVISION).$(AGE)
VLIBPFM=libpfm.so.$(VERSION)
SOLIBEXT=so
endif

CFLAGS+=-I.
ALIBPFM=libpfm.a

TARGETS=$(ALIBPFM)

ifeq ($(CONFIG_PFMLIB_SHARED),y)
TARGETS += $(SLIBPFM)
endif

OBJS=$(SRCS:.c=.o)
SOBJS=$(OBJS:.o=.lo)

INC_COMMON= $(PFMINCDIR)/perfmon/pfmlib.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp.h \
  $(PFMINCDIR)/perfmon/pfmlib_os.h \
  $(PFMINCDIR)/perfmon/perfmon.h \
  $(PFMINCDIR)/perfmon/perfmon_dfl_smpl.h \
  pfmlib_priv.h pfmlib_priv_comp.h \

INC_IA64= $(PFMINCDIR)/perfmon/pfmlib_itanium.h \
  $(PFMINCDIR)/perfmon/pfmlib_itanium2.h \
  $(PFMINCDIR)/perfmon/pfmlib_montecito.h \
  $(PFMINCDIR)/perfmon/perfmon_compat.h \
  $(PFMINCDIR)/perfmon/perfmon_default_smpl.h \
  $(PFMINCDIR)/perfmon/perfmon_ia64.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp_ia64.h \
  $(PFMINCDIR)/perfmon/pfmlib_gen_ia64.h \
  $(PFMINCDIR)/perfmon/pfmlib_os_ia64.h \
  itanium_events.h  itanium2_events.h montecito_events.h

INC_IA32=$(PFMINCDIR)/perfmon/perfmon_pebs_core_smpl.h \
  $(PFMINCDIR)/perfmon/perfmon_pebs_p4_smpl.h \
  $(PFMINCDIR)/perfmon/pfmlib_pentium4.h \
  $(PFMINCDIR)/perfmon/pfmlib_amd64.h \
  $(PFMINCDIR)/perfmon/pfmlib_core.h \
  $(PFMINCDIR)/perfmon/pfmlib_intel_atom.h \
  $(PFMINCDIR)/perfmon/pfmlib_intel_nhm.h \
  $(PFMINCDIR)/perfmon/pfmlib_i386_p6.h \
  $(PFMINCDIR)/perfmon/pfmlib_gen_ia32.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp_i386.h \
  $(PFMINCDIR)/perfmon/pfmlib_os_i386.h \
  amd64_events.h  i386_p6_events.h \
  pentium4_events.h gen_ia32_events.h coreduo_events.h core_events.h \
  intel_atom_events.h intel_corei7_events.h intel_corei7_unc_events.h

INC_X86_64= $(PFMINCDIR)/perfmon/perfmon_pebs_core_smpl.h \
  $(PFMINCDIR)/perfmon/perfmon_pebs_p4_smpl.h \
  $(PFMINCDIR)/perfmon/pfmlib_amd64.h \
  $(PFMINCDIR)/perfmon/pfmlib_core.h \
  $(PFMINCDIR)/perfmon/pfmlib_intel_atom.h \
  $(PFMINCDIR)/perfmon/pfmlib_intel_nhm.h \
  $(PFMINCDIR)/perfmon/pfmlib_gen_ia32.h \
  $(PFMINCDIR)/perfmon/pfmlib_pentium4.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp_x86_64.h \
  $(PFMINCDIR)/perfmon/pfmlib_os_x86_64.h \
  amd64_events.h pentium4_events.h gen_ia32_events.h core_events.h \
  intel_atom_events.h intel_corei7_events.h intel_corei7_unc_events.h

INC_MIPS64= $(PFMINCDIR)/perfmon/pfmlib_gen_mips64.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp_mips64.h \
  $(PFMINCDIR)/perfmon/pfmlib_os_mips64.h \
  gen_mips64_events.h

INC_SICORTEX= $(INC_MIPS64) $(PFMINCDIR)/perfmon/pfmlib_sicortex.h 

INC_POWERPC= $(PFMINCDIR)/perfmon/pfmlib_powerpc.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp_powerpc.h \
  $(PFMINCDIR)/perfmon/pfmlib_os_powerpc.h \
  ppc970_events.h ppc970mp_events.h power4_events.h \
  power5_events.h power5+_events.h power6_events.h \
  power7_events.h powerpc_reg.h

INC_SPARC= $(PFMINCDIR)/perfmon/pfmlib_sparc.h \
  $(PFMINCDIR)/perfmon/pfmlib_comp_sparc.h \
  $(PFMINCDIR)/perfmon/pfmlib_os_sparc.h \
  ultra12_events.h ultra3_events.h ultra3plus_events.h ultra3i_events.h \
  ultra4plus_events.h niagara1_events.h niagara2_events.h

INC_CRAYX2= $(PFMINCDIR)/perfmon/pfmlib_crayx2.h \
  crayx2_events.h pfmlib_crayx2_priv.h

INC_CELL= $(PFMINCDIR)/perfmon/pfmlib_cell.h \
  cell_events.h

INCDEP=$(INC_COMMON) $(INCARCH)

all: $(TARGETS)

$(OBJS) $(SOBJS): $(TOPDIR)/config.mk $(TOPDIR)/rules.mk Makefile $(INCDEP)

libpfm.a:  $(OBJS)
	$(RM) $@
	$(AR) cru $@ $(OBJS)

$(SLIBPFM):  $(SOBJS)
	$(CC) $(CFLAGS) $(SLDFLAGS)  -o $@ $(SOBJS)
	$(LN) $@ $(VLIBPFM)
	$(LN) $@ libpfm.$(SOLIBEXT)

clean:
	$(RM) -f *.o *.lo *.a *.so* *~ *.$(SOLIBEXT)

distclean: clean

depend:
	$(MKDEP) $(CFLAGS) $(SRCS)

install: $(TARGETS)

install:
	@echo building: $(TARGETS)
	-mkdir -p $(DESTDIR)$(LIBDIR)
	$(INSTALL) -m 644 $(ALIBPFM) $(DESTDIR)$(LIBDIR)
ifeq ($(CONFIG_PFMLIB_SHARED),y)
	$(INSTALL) $(SLIBPFM) $(DESTDIR)$(LIBDIR)
	cd $(DESTDIR)$(LIBDIR); $(LN) $(SLIBPFM) $(VLIBPFM)
	cd $(DESTDIR)$(LIBDIR); $(LN) $(SLIBPFM) libpfm.$(SOLIBEXT)
endif
