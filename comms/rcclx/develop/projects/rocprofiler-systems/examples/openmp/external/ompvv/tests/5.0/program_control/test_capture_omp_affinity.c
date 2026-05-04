//===--- test_capture_omp_affinity.c ----------------------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// This test checks that the omp_capture_affinity() can be used within a 
// parallel region to determine affinity of each thread.
//
// Adopted from OpenMP 5.0 Example affinity_display.3.c
//===------------------------------------------------------------------------===//


#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <omp.h>
#include "ompvv.h"

#define BUFFER_STORE 80

int main() {

   int i, threads, errors = 0, thrd_num, num_threads = 0;
   int  format_size = 13; //size of my_format
   char format[] = "thrd_num=%0.4n";

   char **buffer;

   threads = OMPVV_NUM_THREADS_HOST;

   buffer = (char **) malloc( sizeof(char *) * threads );

   for (i = 0; i < threads; i++) { 
      buffer[i] = (char *) malloc( sizeof(char) * BUFFER_STORE); 
   }

   #pragma omp parallel private(thrd_num) num_threads(threads) 
   {
      thrd_num = omp_get_thread_num();
      if(thrd_num > threads) { //safety: dont exceed # of buffers
        OMPVV_ERROR("Number of thread greater than requested");
        exit(1); 
      }

      if(thrd_num == 0)
        num_threads = omp_get_num_threads();

      omp_capture_affinity(buffer[thrd_num], (size_t) BUFFER_STORE, format);
   }

   // Checks if the affinity string is corrrectly captured
   for(i = 0; i < num_threads; i++) {
      char str[30];
      snprintf(str, sizeof(str), "thrd_num=%0.4d", i);

      for(int j = 0; j < format_size; ++j) {
        if(str[j] != buffer[i][j]) {
          errors++;
          break;
        }
      }
      free(buffer[i]);
   }

   free (buffer);

   OMPVV_REPORT_AND_RETURN(errors);
}
