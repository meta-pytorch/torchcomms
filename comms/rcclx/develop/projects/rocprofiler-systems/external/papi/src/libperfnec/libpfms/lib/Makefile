#
# Copyright (c) 2006 Hewlett-Packard Development Company, L.P.
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

TOPDIR  := $(shell if [ "$$PWD" != "" ]; then echo $$PWD; else pwd; fi)/../..

include $(TOPDIR)/config.mk
include $(TOPDIR)/rules.mk

CFLAGS+= -pthread -D_GNU_SOURCE
LDFLAGS+=-static 

PFMSINCDIR=../include

#
# Library version
#
VERSION=0
REVISION=1
AGE=0

SRCS=libpfms.c
HEADERS=../include/libpfms.h

ALIBPFM=libpfms.a
TARGETS=$(ALIBPFM)
ifneq ($(CONFIG_PFMLIB_ARCH_CRAYX2),y)
SLIBPFM=libpfms.so.$(VERSION).$(REVISION).$(AGE)
VLIBPFM=libpfms.so.$(VERSION)
endif

OBJS=$(SRCS:.c=.o)
SOBJS=$(OBJS:.o=.lo)



#
# assume that if llibpfm built static, libpfms should
# also be static, i.e., likely platform does not support
# shared libraries.
#
ifeq ($(CONFIG_PFMLIB_SHARED),y)
TARGETS += $(SLIBPFM)
endif

ifeq ($(SYS),Linux)
SLDFLAGS=-shared -Wl,-soname -Wl,libpfms.so.$(VERSION)
endif

CFLAGS+=-I$(PFMSINCDIR)

all: $(TARGETS)

$(OBJS) $(SOBJS): $(HEADERS) $(TOPDIR)/config.mk $(TOPDIR)/rules.mk Makefile

libpfms.a: $(OBJS)
	$(RM) $@
	$(AR) cru $@ $(OBJS)

$(SLIBPFM): $(SOBJS)
	$(CC) $(CFLAGS) $(SLDFLAGS) -o $@ $(SOBJS)
	$(LN) -sf $@ libpfms.so.$(VERSION)

clean:
	$(RM) -f *.o *.lo *.a *.so* *~

distclean: clean

install: $(TARGETS)

install:
	-mkdir -p $(DESTDIR)$(LIBDIR)
	$(INSTALL) -m 644 $(ALIBPFM) $(DESTDIR)$(LIBDIR)
	$(INSTALL) $(SLIBPFM) $(DESTDIR)$(LIBDIR)
	cd $(DESTDIR)$(LIBDIR); $(LN) $(SLIBPFM) $(VLIBPFM)
	cd $(DESTDIR)$(LIBDIR); $(LN) $(SLIBPFM) libpfms.so
	-mkdir -p $(DESTDIR)$(INCDIR)/perfmon
	$(INSTALL) -m 644 $(HEADERS) $(DESTDIR)$(INCDIR)/perfmon
