/*
 *  This code is a simple filter for the helgrind_out.txt file
 *  produced by:
 *  "valgrind --tool=helgrind --log-file=helgrind_out.txt someProgram"
 *
 * This is useful because the tool does not recognize PAPI locks,
 * thus reports as possible race conditions reads/writes by
 * different threads that are actually fine (surrounded by locks).
 *
 * This was written particularly for krentel_pthreads_race.c 
 * when processed by the above valgrind. We produce a line per
 * condition, in the form:
 * OP@file:line OP@file:line
 * where OP is R or W. The first file:line code occurred
 * after the second file:line code, and on a different thread.
 * 
 * We print the results to stdout. It is useful to filter this
 * through the standard utility 'uniq', each occurrence only 
 * needs to be investigated once. Just insure there are
 * MATCHING locks around each operation within the code.
 *
 * An example run (using uniq): The options -uc will print 
 * only unique lines, preceeded by a count of how many times
 * it occurs.
 *
 * ./filter_helgrind | uniq -uc
 *
 * An example output line (piped through uniq as above):
 *       1 R@threads.c:190                    W@threads.c:206
 * An investigation shows threads.c:190 is protected by 
 * _papi_hwi_lock(THREADS_LOCK); and threads.c:206 is
 * protected by the same lock. Thus no data race can 
 * occur for this instance.
 *
 * Compilation within the papi/src/ctests directory:
 * make filter_helgrind
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** args) {
   (void) argc;
   (void) args;

   char myLine[16384];
   int state, size;
   char type1, type2;
   char fname1[256], fname2[256];
   char *paren1, *paren2;

   FILE *HELOUT = fopen("helgrind_out.txt", "r");  // Read the file.
   if (HELOUT == NULL) {
      fprintf(stderr, "Could not open helgrind_out.txt.\n");
      exit(-1);
   }

   char PDRR[]="Possible data race during read";
   char PDRW[]="Possible data race during write";
   char TCWW[]="This conflicts with a previous write";
   char TCWR[]="This conflicts with a previous read";
   char atSTR[]="   at ";

   // State machine:
   // State 0: We are looking for a line with PDRR or PDRW.
   //          We don't exit until we find it, or run out of lines.
   //          if we find it, we remember which and go to state 1.
   // State 1: Looking for "   at " in column 11. 
   //          When found, we extract the string betweeen '(' and ')'
   //          which is program name:line. go to state 2.
   // State 2: We are searching for TCWW, TCWR, PDRW, PDRR.
   //          If we find the first two:
   //             Remember which, and go to state 3.
   //          If we find either of the second two, go back to State 1.
   // State 3: Looking for "   at " in column 11.
   //          When found, extract the string betweeen '(' and ')',
   //          which is program name:line.
   //          OUTPUT LINE for an investigation.
   //          Go to State 0.

   state = 0;        // looking for PDRR, PDRW. 
   while (fgets(myLine, 16384, HELOUT) != NULL) {
      if (strlen(myLine) < 20) continue;
      switch (state) {
         case 0:  // Looking for PDRR or PRDW.
            if (strstr(myLine, PDRR) != NULL) { 
               type1='R';
               state=1;
               continue;
            }

            if (strstr(myLine, PDRW) != NULL) {
               type1='W';
               state=1;
               continue;
            }
      
            continue;
            break;

         case 1: // Looking for atSTR in column 11.
            if (strncmp(myLine+10, atSTR, 6) != 0) continue;
            paren1=strchr(myLine, '(');
            paren2=strchr(myLine, ')');
            if (paren1 == NULL || paren2 == NULL ||
                paren1 > paren2) {
               state=0;             // Abort, found something I don't understand.
               continue;
            }

            size = paren2-paren1-1;          // compute length of name.
            strncpy(fname1, paren1+1, size); // Copy the name.
            fname1[size]=0;                  // install z-terminator.
            state=2;
            continue;
            break;

         case 2: // Looking for TCWW, TCWR, PDRR, PDRW.
            if (strstr(myLine, TCWR) != NULL) {
               type2='R';
               state=3;
               continue;
            }

            if (strstr(myLine, TCWW) != NULL) { 
               type2='W';
               state=3;
               continue;
            }

            if (strstr(myLine, PDRR) != NULL) { 
               type1='R';
               state=1;
               continue;
            }

            if (strstr(myLine, PDRW) != NULL) {
               type1='W';
               state=1;
               continue;
            }

            continue;
            break;

         case 3: // Looking for atSTR in column 11.
            if (strncmp(myLine+10, atSTR, 6) != 0) continue;
            paren1=strchr(myLine, '(');
            paren2=strchr(myLine, ')');
            if (paren1 == NULL || paren2 == NULL ||
                paren1 > paren2) {
               state=0;             // Abort, found something I don't understand.
               continue;
            }

            size = paren2-paren1-1;          // compute length of name.
            strncpy(fname2, paren1+1, size); // Copy the name.
            fname2[size]=0;                  // install z-terminator.
            fprintf(stdout, "%c@%-32s %c@%-32s\n", type1, fname1, type2, fname2);
            state=0;
            continue;
            break;
      } // end switch.
   } // end while.
   
   fclose(HELOUT);
   exit(0);
}  
