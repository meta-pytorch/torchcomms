CC = gcc
CFLAGS = -O2 -Wall -fopenmp
LFLAGS = 
PAPI_INCLUDE = ../../.. 
PAPI_LIBRARY = ../../../libpapi.a

#all:	powercap_plot

powercap_plot:	powercap_plot.o
	$(CC) $(LFLAGS) -o powercap_plot -fopenmp powercap_plot.o  $(PAPI_LIBRARY)

powercap_plot.o:	powercap_plot.c
	$(CC) $(CFLAGS) -I$(PAPI_INCLUDE)  -c powercap_plot.c

clean:	
	rm -f *~ *.o powercap_plot results*

