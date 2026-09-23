# File: testlib/Makefile

include Makefile.target

INCLUDE = -I. -I..

TESTLIBOBJS:= test_utils.o

UTILOBJS:= do_loops.o test_utils.o clockcore.o
ifeq ($(ENABLE_FORTRAN),yes)
UTILOBJS+= ftests_util.o
TESTLIBOBJS+= ftests_util.o
endif

all:	libtestlib.a $(UTILOBJS)

libtestlib.a: $(TESTLIBOBJS)
	$(AR) $(ARG64) rv $@ $(TESTLIBOBJS)

do_loops.o: do_loops.c papi_test.h do_loops.h
	$(CC) $(INCLUDE) $(CFLAGS) -O0 -c do_loops.c
#	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -c do_loops.c

clockcore.o: clockcore.c
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -c clockcore.c

test_utils.o: test_utils.c
	$(CC) $(INCLUDE) $(CFLAGS) $(TOPTFLAGS) -c test_utils.c

ftests_util.o: ftests_util.F fpapi_test.h 
	$(F77) $(INCLUDE) $(FFLAGS) $(FTOPTFLAGS) -c ftests_util.F

clean:
	rm -f *.o *genmod.f90 *genmod.mod *.stderr *.stdout core *~ $(ALL) libtestlib.a libtestlib.so

distclean: clean
	rm -f Makefile.target

install: 
	@echo "Papi testlib (DATADIR) being installed in: \"$(DATADIR)\""; 
	-mkdir -p $(DATADIR)/testlib
	-chmod go+rx $(DATADIR)
	-chmod go+rx $(DATADIR)/testlib
	-find . -name "*.[chaF]" -type f -exec cp {} $(DATADIR)/testlib \;
	-cp Makefile.target $(DATADIR)/testlib/Makefile
