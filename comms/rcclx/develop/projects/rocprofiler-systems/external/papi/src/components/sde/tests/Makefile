NAME=sde
include ../../Makefile_comp_tests.target
INCLUDE += -I$(datadir)/sde_lib -I..

intel_compilers := ifort ifx
cray_compilers := ftn crayftn

ifeq ($(notdir $(F77)),gfortran)
	FFLAGS +=-ffree-form -ffree-line-length-none
else ifeq ($(notdir $(F77)),flang)
	FFLAGS +=-ffree-form
else ifeq ($(findstring $(notdir $(F77)), $(intel_compilers)),)
    FFLAGS +=-free
else ifeq ($(findstring $(notdir $(F77)), $(cray_compilers)),)
    FFLAGS +=-ffree
endif
FFLAGS +=-g
CFLAGS +=-g
CXXFLAGS +=-g -std=c++11

ifeq ($(BUILD_LIBSDE_SHARED),yes)
	LDFLAGS += -Llib -L$(datadir) -L$(datadir)/libpfm4/lib -lpapi -lpfm -lsde
	LIBSDE=yes
else ifeq ($(BUILD_LIBSDE_STATIC),yes)
	LDFLAGS += -Llib $(datadir)/libsde.a -lpapi
	LIBSDE=yes
endif

SDE_F08_API=../sde_F.F90

ifeq ($(LIBSDE),yes)
	TESTS = Minimal_Test Minimal_Test++ Simple_Test Simple2_Test Simple2_NoPAPI_Test Simple2_Test++ Recorder_Test Recorder_Test++ Created_Counter_Test Created_Counter_Test++ Overflow_Test Counting_Set_Simple_Test Counting_Set_MemLeak_Test Counting_Set_Simple_Test++ Counting_Set_MemLeak_Test++
endif
ifeq ($(BUILD_LIBSDE_STATIC),yes)
	TESTS += Overflow_Static_Test
endif

ifeq ($(ENABLE_FORTRAN_TESTS),yes)
    TESTS += sde_test_f08
endif


sde_tests: $(TESTS)

################################################################################
## Minimal test
prfx=Minimal

Minimal_Test: $(prfx)/Minimal_Test.c
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) $(LDFLAGS)

Minimal_Test++: $(prfx)/Minimal_Test++.cpp
	$(CXX) $< -o $@ $(INCLUDE) $(CXXFLAGS) $(UTILOBJS) $(LDFLAGS)

################################################################################
## Simple test
prfx=Simple

libSimple.so: $(prfx)/Simple_Lib.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $^

Simple_Test: $(prfx)/Simple_Driver.c libSimple.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) -lSimple $(LDFLAGS) -lm

################################################################################
## Simple2 test
prfx=Simple2

libSimple2.so: $(prfx)/Simple2_Lib.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $^

Simple2_Test: $(prfx)/Simple2_Driver.c libSimple2.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) -lSimple2 $(LDFLAGS) -lm

Simple2_NoPAPI_Test: $(prfx)/Simple2_NoPAPI_Driver.c libSimple2.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) -Llib -lSimple2 -L$(datadir) -lsde -lm -ldl

libSimple2++.so: $(prfx)/Simple2_Lib++.cpp
	$(CXX) -shared -Wall -fPIC $(CXXFLAGS) $(INCLUDE) -o lib/$@ $^

Simple2_Test++: $(prfx)/Simple2_Driver++.cpp libSimple2++.so
	$(CXX) $< -o $@ $(INCLUDE) $(CXXFLAGS) $(UTILOBJS) -lSimple2++ $(LDFLAGS) -lm

################################################################################
## Recorder test
prfx=Recorder

libRecorder.so: $(prfx)/Lib_With_Recorder.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $^

Recorder_Test: $(prfx)/Recorder_Driver.c libRecorder.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) -lRecorder $(LDFLAGS) -lm

libRecorder++.so: $(prfx)/Lib_With_Recorder++.cpp
	$(CXX) -shared -Wall -fPIC $(CXXFLAGS) $(INCLUDE) -o lib/$@ $^

Recorder_Test++: $(prfx)/Recorder_Driver++.cpp libRecorder++.so
	$(CXX) $< -o $@ $(INCLUDE) $(CXXFLAGS) $(UTILOBJS) -lRecorder++ $(LDFLAGS) -lm


################################################################################
## Created Counter test
prfx=Created_Counter

libCreated_Counter.so: $(prfx)/Lib_With_Created_Counter.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $^

libCreated_Counter_static.a: $(prfx)/Lib_With_Created_Counter.c
	$(CC) -Bstatic -static -Wall $(CFLAGS) $(INCLUDE) -c -o lib/Lib_With_Created_Counter.o $^
	ar rs lib/$@ lib/Lib_With_Created_Counter.o
	rm lib/Lib_With_Created_Counter.o

Created_Counter_Test: $(prfx)/Created_Counter_Driver.c libCreated_Counter.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) -lCreated_Counter $(LDFLAGS) -lm
Overflow_Test: $(prfx)/Overflow_Driver.c libCreated_Counter.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) -lCreated_Counter $(LDFLAGS) -lm
Overflow_Static_Test: $(prfx)/Overflow_Driver.c libCreated_Counter_static.a
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) lib/libCreated_Counter_static.a $(datadir)/libpapi.a $(datadir)/libsde.a -ldl -lrt

libCreated_Counter++.so: $(prfx)/Lib_With_Created_Counter++.cpp
	$(CXX) -shared -Wall -fPIC $(CXXFLAGS) $(INCLUDE) -o lib/$@ $^

Created_Counter_Test++: $(prfx)/Created_Counter_Driver++.cpp libCreated_Counter++.so
	$(CXX) $< -o $@ $(INCLUDE) $(CXXFLAGS) $(UTILOBJS) -lCreated_Counter++ $(LDFLAGS) -lm

################################################################################
## Counting Set test
prfx=Counting_Set

libCounting_Set++.so: $(prfx)/CountingSet_Lib++.cpp $(prfx)/cset_lib.hpp
	$(CXX) -shared -Wall -fPIC $(CXXFLAGS) $(INCLUDE) -o lib/$@ $<

Counting_Set_MemLeak_Test++: $(prfx)/MemoryLeak_CountingSet_Driver++.cpp libCounting_Set++.so
	$(CXX) $< -o $@ $(INCLUDE) $(CXXFLAGS) $(UTILOBJS) -lCounting_Set++ $(LDFLAGS)

Counting_Set_Simple_Test++: $(prfx)/Simple_CountingSet_Driver++.cpp libCounting_Set++.so
	$(CXX) $< -o $@ $(INCLUDE) $(CXXFLAGS) $(UTILOBJS) -lCounting_Set++ $(LDFLAGS)

libCounting_Set.so: $(prfx)/CountingSet_Lib.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $^

Counting_Set_Simple_Test: $(prfx)/Simple_CountingSet_Driver.c libCounting_Set.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) -lCounting_Set $(LDFLAGS)

Counting_Set_MemLeak_Test: $(prfx)/MemoryLeak_CountingSet_Driver.c libCounting_Set.so
	$(CC) $< -o $@ $(INCLUDE) $(CFLAGS) $(UTILOBJS) -lCounting_Set $(LDFLAGS)

################################################################################
## Advanced test
prfx=Advanced_C+FORTRAN
rcrd_prfx=Recorder

libXandria.so: $(prfx)/Xandria.F90
	$(F77) -shared -Wall -fPIC $(FFLAGS) $(INCLUDE) -o lib/$@ $(SDE_F08_API) $<

libGamum.so: $(prfx)/Gamum.c
	$(CC) -shared -Wall -fPIC $(CFLAGS) $(INCLUDE) -o lib/$@ $^

sde_test_f08: $(prfx)/sde_test_f08.F90 $(UTILOBJS) $(PAPILIB) libXandria.so libGamum.so libRecorder.so
	$(F77) $< -o $@ $(INCLUDE) $(FFLAGS) $(UTILOBJS) -lXandria -lGamum -lRecorder $(LDFLAGS)

################################################################################
## Cleaning

clean:
	rm -f *.o *.mod lib/*.so lib/*.a $(TESTS)
