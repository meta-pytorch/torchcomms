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

#
# Look in config.mk for options
#
TOPDIR  := $(shell if [ "$$PWD" != "" ]; then echo $$PWD; else pwd; fi)

include config.mk

DIRS=lib include docs

ifeq ($(SYS),Linux)
DIRS +=libpfms
endif

DIRS += $(EXAMPLES_DIRS)

all: 
	@echo Compiling for \'$(ARCH)\' target
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done

lib:
	$(MAKE) -C lib

clean: 
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done

distclean:  clean

depend: 
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done

tar: clean
	a=`basename $$PWD`; cd ..; tar zcf $$a.tar.gz $$a; echo generated ../$$a.tar.gz;

tarcvs: clean
	a=`basename $$PWD`; cd ..; tar --exclude=CVS -zcf $$a.tar.gz $$a; echo generated ../$$a.tar.gz;
install: 
	@set -e ; for d in $(DIRS) ; do $(MAKE) -C $$d $@ ; done

install_examples:
	@set -e ; for d in $(EXAMPLES_DIRS) ; do $(MAKE) -C $$d $@ ; done

.PHONY: tar tarcvs lib

# DO NOT DELETE
