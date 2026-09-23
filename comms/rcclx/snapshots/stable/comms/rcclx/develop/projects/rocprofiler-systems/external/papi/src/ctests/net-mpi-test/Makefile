CC              = gcc
CC_R            = gcc -pthread
CC_SHR          = gcc -shared
#MXMPIPATH       = /usr/local/mpich/mpich-gcc
#MXMPIPATH       = /usr/local/mpich-mx
#MPICC           = $(MXMPIPATH)/bin/mpicc
#MPICC           = /usr/bin/mpicc
MPICC           = mpicc
MPICC_SHR       = $(MPICC) -shared
MPICCLD_SHR     = $(MPICC_SHR)
F77             = g77
FLAGS           = -g -Wall
CFLAGS          = $(FLAGS) -O3 # -DPROFILE_TIMER -DDEBUG -DVERBOSE

BLASLIBS        = -lblas
#BLASLIBS       = -L/usr/local/lib -lf77blas -latlas
LAPACKLIBS      = -llapack

UTILOBJS= ../do_loops.o ../test_utils.o ../dummy.o
INCLUDE = -I.. -I../.. -I/usr/include
PAPILIB = -L../.. -lpapi 
MPILIBS = 
MPIINC = 

XTRALIBS        =
PTHRLIBS        =
MPILIBS         =
LIBS            =$(PAPILIB) -lm

TESTS   	= cpi

tests: $(TESTS)

# Applications

# Test programs
../test_utils.o: ../test_utils.c ../papi_test.h ../test_utils.h
	$(CC) $(CFLAGS) $(INCLUDE) -c ../test_utils.c -o ../test_utils.o

../do_loops.o: ../do_loops.c ../papi_test.h ../test_utils.h
	$(CC) $(CFLAGS) $(INCLUDE) -c ../do_loops.c -o ../do_loops.o

../dummy.o: ../dummy.c
	$(CC) $(CFLAGS) $(INCLUDE) -c ../dummy.c -o ../dummy.o

cpi: cpi.c $(UTILOBJS)
	$(MPICC) $(MPFLAGS) $(CFLAGS) $(INCLUDE) $(MPIINC) $(TOPTFLAGS) cpi.c $(UTILOBJS) $(PAPILIB) $(MPILIBS) -o cpi

#cpi: cpi.c 
#	$(MPICC)  $(FLAGS) cpi.c -o $@ $(MPIPERFLIBS) $(XTRALIBS) $(MPILIBS) -lm

clean:
	rm -f core $(TESTS)  *~ *.o 
