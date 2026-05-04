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

TOPDIR  := $(shell if [ "$$PWD" != "" ]; then echo $$PWD; else pwd; fi)/..

include $(TOPDIR)/config.mk
include $(TOPDIR)/rules.mk

DIRS=lib
CFLAGS+= -pthread -D_GNU_SOURCE -I./include
LIBS += -L$(TOPDIR)/libpfms/lib -lpfms $(PFMLIB) -lm

TARGETS=syst_smp

all: $(TARGETS)

syst_smp: ./lib/libpfms.a syst_smp.o
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ syst_smp.o $(LIBS) -lpthread

clean:
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done
	$(RM) -f *.o $(TARGETS) *~

distclean: clean

.FORCE: lib/libpfms.a
lib/libpfms.a: 
	@set -e ; $(MAKE) -C lib all

install depend: $(TARGETS)

install depend:
ifeq ($(CONFIG_PFMLIB_ARCH_SICORTEX),y)
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done
endif
