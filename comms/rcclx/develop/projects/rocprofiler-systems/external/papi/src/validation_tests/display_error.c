#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "display_error.h"

double display_error(long long average,
		     long long high,
		     long long low,
		     long long expected,
                     int quiet) {

   double error;

   error=(((double)average-expected)/expected)*100.0;

   if (!quiet) {
      printf("   Expected: %lld\n", expected);
      printf("   High: %lld   Low:  %lld   Average:  %lld\n",
          high,low,average);

      printf("   ( note, a small value above %lld may be expected due\n",
	  expected);
      printf("     to overhead and interrupt noise, among other reasons)\n");

      printf("   Average Error = %.2f%%\n",error);
   }

   return error;

}
