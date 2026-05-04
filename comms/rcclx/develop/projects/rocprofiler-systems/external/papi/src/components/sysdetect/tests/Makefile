NAME=sysdetect

include ../../Makefile_comp_tests.target

%.o:%.c
	$(CC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

ifneq ($(MPICC),)
    ifeq ($(NO_MPI_TESTS),)
        MPITESTS = query_device_mpi
    else
        MPITESTS =
    endif
endif

FTESTS =
ifeq ($(ENABLE_FORTRAN_TESTS),yes)
    FTESTS = query_device_simple_f
endif

intel_compilers := ifort ifx
cray_compilers := ftn crayftn

ifeq ($(notdir $(F77)),gfortran)
    FFLAGS +=-ffree-form -ffree-line-length-none
else ifeq ($(patsubst %flang,,$(notdir $(F77))),)  # compiler name ends with flang
    FFLAGS +=-ffree-form
else ifneq ($(findstring $(notdir $(F77)), $(intel_compilers)),)
    FFLAGS +=-free
else ifneq ($(findstring $(notdir $(F77)), $(cray_compilers)),)
    FFLAGS +=-ffree
endif

TESTS = query_device_simple \
        $(FTESTS)           \
        $(MPITESTS)

sysdetect_tests: $(TESTS)

query_device_simple: query_device_simple.o $(UTILOBJS) $(PAPILIB)
	$(CC) $(CFLAGS) $(INCLUDE) -o query_device_simple query_device_simple.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

query_device_mpi: query_device_mpi.o $(UTILOBJS) $(PAPILIB)
	$(MPICC) $(CFLAGS) $(INCLUDE) -o query_device_mpi query_device_mpi.o $(UTILOBJS) $(PAPILIB) $(LDFLAGS)

query_device_mpi.o: query_device_mpi.c
	$(MPICC) $(CFLAGS) $(OPTFLAGS) $(INCLUDE) -c -o $@ $<

query_device_simple_f: query_device_simple_f.F
	$(F77) $(FFLAGS) -I../../.. -o $@ $< $(PAPILIB) $(LDFLAGS)

clean:
	rm -f $(TESTS) *.o
