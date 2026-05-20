//-----------------------------------------------------------------------------
// benchStats: Reads a CSV file of three columns; and produces various stats
// for each column. 
//-----------------------------------------------------------------------------


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <papi.h>
#include "papi_test.h" 
#include <sys/time.h>

typedef struct {
   double init;
   double pcpVal;
} BenchData_t;
   
//-----------------------------------------------------------------------------
// qsort comparison to sort benchData array into ascending order of Init.
//-----------------------------------------------------------------------------
int sortAscBDInit(const void *vA, const void *vB) {
   BenchData_t *sA=(BenchData_t*) vA;        // recast to cmp type.
   BenchData_t *sB=(BenchData_t*) vB;        // ..

   if (sA->init < sB->init) return(-1);      // move cA toward front of list.
   if (sA->init > sB->init) return( 1);      // move cA toward end of list.
   return(0);                                // equality.
} // end routine.
               

//-----------------------------------------------------------------------------
// qsort comparison to sort benchData array into ascending order of pcpVal.
//-----------------------------------------------------------------------------
int sortAscBDPCP(const void *vA, const void *vB) {
   BenchData_t *sA=(BenchData_t*) vA;       // recast to cmp type.
   BenchData_t *sB=(BenchData_t*) vB;       // ..

   if (sA->pcpVal < sB->pcpVal) return(-1); // move cA toward front of list.
   if (sA->pcpVal > sB->pcpVal) return( 1); // move cA toward end of list.
   return(0);                               // equality.
} // end routine.
               

//-----------------------------------------------------------------------------
// median of sorted fRec init value array, for count elements.
// if count is odd, median is at (count/2).  e.g. count=7, 7/2=3, a[3] is it.
// if count is even, median is average of (count/2)-1 and (count/2). e.g. 
// count=8, a[3]+a[4] are two center values.
//-----------------------------------------------------------------------------
double median_BDInit(BenchData_t* fRec, int count) {
   if ((count&1)) return fRec[(count>>1)].init;          // median if count is odd.
   return ( (fRec[(count>>1)-1].init +
             fRec[(count>>1)].init)/2.0 );               // median if count is even.
} // end routine.   


//-----------------------------------------------------------------------------
// median of sorted fRec pcp value array, for count elements.
// if count is odd, median is at (count/2).  e.g. count=7, 7/2=3, a[3] is it.
// if count is even, median is average of (count/2)-1 and (count/2). e.g. 
// count=8, a[3]+a[4] are two center values.
//-----------------------------------------------------------------------------
double median_BDPCP(BenchData_t* fRec, int count) {
   if ((count&1)) return fRec[(count>>1)].pcpVal;        // median if count is odd.
   return ( (fRec[(count>>1)-1].pcpVal +
             fRec[(count>>1)].pcpVal)/2.0 );             // median if count is even.
} // end routine.   


//-----------------------------------------------------------------------------
// Write a rough ASCII histogram.
//-----------------------------------------------------------------------------

void histogram(int *BinCount, int bins, int total, double minVal, double width) {
   int i;
   double ctr, pcnt;
   for (i=0; i<bins; i++) {
      ctr = minVal + ( (i+.5)*width);                    // compute bin center.
      pcnt = (100.*BinCount[i]) / (total+0.0);           // compute percentage of values.
      printf("%8.1f, %8i, =%5.2f%%\n", ctr, BinCount[i], pcnt);
   }
} // end routine.


//-----------------------------------------------------------------------------
// MAIN.
//-----------------------------------------------------------------------------

int main(int argc, char **argv) {                                       // args to set filename.
   int i,j,ret;
   FILE *Inp;
   int Block = 64;
   BenchData_t *fRec = calloc(Block, sizeof(BenchData_t));              // Make some initial space.
   char title1[128], title2[128];
   int count=0, size=Block;                                             // records we have, size of malloc.

   if (argc != 2) {                                                     // progname inpfile 
      fprintf(stderr, "You must specify a CSV file on the command line.\n");
      exit(-1);
   }

   Inp =fopen(argv[1], "r");                                            // Open arg1 as input file.
   if (Inp == NULL) {                                                   // In case of failure...
      fprintf(stderr, "Error reading file %s.\n", argv[1]);             // .. report it.
      exit(-1);                                                         // .. exit.
   }

   ret = fscanf(Inp, "%128[^,],%128[^\n]", title1, title2);  // Read the three titles.
   if (ret !=2) {                                                             // If we did not get 3, 
      fprintf(stderr, "File Format Error, read %i of 2 expected column titles in file '%s'.\n", ret, argv[1]);
      fclose(Inp);
      exit(-1);
   }

   while (1) {                                                          // We break out of this.
      ret = fscanf(Inp, "%lf,%*[ ]%lf\n", &fRec[count].init, &fRec[count].pcpVal);  // read two values.
      if (ret == EOF) break;                                            // quiet break at EOF.
      if (ret != 2) {                                                   // If we did not get 3 values,
         fprintf(stderr, "File Format Error, read %i of 2 expected column values in file '%s', line %i.\n", ret, argv[1], count+2);
         fclose(Inp);
         free(fRec);
         exit(-1);
      }

      count++;                                                          // increment count of lines.
      if (count == size) {                                              // If that was last in size,
         size += Block;                                                 // Need to add another chunk.
         fRec = realloc(fRec, size*sizeof(BenchData_t));                // make more space.
         if (fRec == NULL) {
            fprintf(stderr, "Memory failure, failed to realloc(fRec, %i)\n", size);
            fclose(Inp);
            exit(-1);
         }
      } // end if realloc needed.
   } // end of all reading.
  
   fclose(Inp);                                                         // Done with this file.
   fprintf(stderr, "Read a total of %i data lines.\n", count);

   // We can go ahead and find the min, max, max-but-1 and average for both Init and PCP.
   double minInit=fRec[0].init;
   double maxInit=minInit;
   double firstInit=minInit;
   double avgInit=minInit; 
   double maxBut1Init = fRec[1].init;                                  // start this at second value.

   double minPCP=fRec[0].pcpVal;
   double maxPCP=minPCP;
   double firstPCP=minPCP;
   double avgPCP=minPCP; 
   double maxBut1PCP = fRec[1].pcpVal;                                  // start this at second value.

   for (i=1; i<count; i++) {                                            // check the rest.
      double v;
      v=fRec[i].init;                                                   // get it.
      if (v > maxInit) maxInit=v;
      if (v > maxBut1Init) maxBut1Init=v;
      if (v < minInit) minInit=v;
      avgInit += v;                                                     // build for average.

      v=fRec[i].pcpVal;                                                 // get it.
      if (v > maxPCP) maxPCP=v;
      if (v > maxBut1PCP) maxBut1PCP=v;
      if (v < minPCP) minPCP=v;
      avgPCP += v;                                                      // build for average.
   }

   avgInit /= ((double) count);                                         // compute average.
   avgPCP /= ((double) count);                                          // compute average.


   // The mode: This is the highest value in a histogram of values;
   // but that can be quite arbitary in picking the window size. We
   // use as our # of bins the square root of the count; and make 
   // the bin size the (max-min)/#bins.

   int maxBin, *InitBins=NULL, *ReadBins=NULL;                          // counters for compute.
   int bins = ceil(sqrtf((double) count));
   int expCount = round(((count+0.0) / (bins+0.0)));                    // expected count per bin.

   // Mode for Init. 
   qsort(fRec, count, sizeof(BenchData_t), sortAscBDInit);              // put in ascending order of Init time.
   
   double medInit = median_BDInit(fRec, count);                         // get the overall median. 
   double Initrange=(maxInit-minInit);                                  // compute the range.
   double Initbin=ceil(Initrange/bins);                                 // bin size.
   if (Initbin == 0.0) Initbin=1.0;                                     // If all the same, just one bin.
   InitBins = calloc(bins, sizeof(int));                                // make counters for them.

   for (i=0; i<count; i++) {                                            // for every record,
      double v = fRec[i].init - minInit;                                // ..value to bin.
      for (j=0; j<bins; j++) {                                          // ..search for bin.
         if (v < (j+1)*Initbin) break;                                  // ..if we found the bin, break.
      }

      InitBins[j]++;                                                    // Add to number in that bin.
   }

   maxBin = 0;                                                          // first bin is beginning of max.
   for (i=1; i<bins; i++) {
      if (InitBins[i] > InitBins[maxBin]) maxBin=i;                     // remember index of max bin.
   }

   double modeInit = minInit + ((maxBin+0.5) * Initbin);                // find the mode as centerpoint.
   int actInitCount = InitBins[maxBin];                                 // actual init count.


   // Mode for PCP. 
   qsort(fRec, count, sizeof(BenchData_t), sortAscBDPCP);               // put in ascending order of pcpVal.
   double medPCP = median_BDPCP(fRec, count);                           // get the overall median. 
   double PCPrange=(maxPCP-minPCP);                                     // compute the range.
   double PCPbin=ceil(PCPrange/bins);                                 // bin size.
   if (PCPbin == 0.0) PCPbin=1.0;                                       // If all the same, just one bin.
   ReadBins = calloc(bins, sizeof(int));                                // make counters for them.

   for (i=0; i<count; i++) {                                            // for every record,
      double v = fRec[i].pcpVal - minPCP;                               // ..value to bin.
      for (j=0; j<bins; j++) {                                          // ..search for bin.
         if (v < (j+1)*PCPbin) break;                                   // ..if we found the bin, break.
      }

      ReadBins[j]++;                                                    // Add to number in that bin.
   }

   maxBin = 0;                                                          // first bin is beginning of max.
   for (i=1; i<bins; i++) {
      if (ReadBins[i] > ReadBins[maxBin]) maxBin=i;                     // remember index of max bin.
   }

   double modePCP = minPCP + ((maxBin+0.5) * PCPbin);                   // find the mode as centerpoint.
   int actPCPCount = ReadBins[maxBin];                                  // actual init count.


   // Stats for init.

   printf("Stats for Initialization time in file '%s'.\n" , argv[1]   );
   printf("Sample Values                  ,%8i\n"  , count            );
   printf("Minimum uS                     ,%8.1f\n", minInit          );
   printf("Maximum uS                     ,%8.1f\n", maxInit          );
   printf("Average uS                     ,%8.1f\n", avgInit          );
   printf("Median  uS                     ,%8.1f\n", medInit          );
   printf("First   uS                     ,%8.1f\n", firstInit        );
   printf("Max w/o First                  ,%8.1f\n", maxBut1Init      );
   printf("Range   uS                     ,%8.1f\n", Initrange        );
   printf("Histogram Bins chosen          ,%8i\n"  , bins             );
   printf("Bin width uS                   ,%8.1f\n", Initbin          );
   printf("Mode (center highest Bin Count),%8.1f\n", modeInit         );
   printf("Mode Bin Count                 ,%8i\n"  , actInitCount     );
   printf("Bin Expected Count             ,%8i\n"  , expCount         );
   printf("\n");
   printf("Initialization Histogram:\n"
          "binCenter, Count, %% of Count\n");

   histogram(InitBins, bins, count, minInit, Initbin);

   printf("\n");
   printf("Stats for PCP event read time in file '%s'.\n" , argv[1]   );
   printf("Sample Values                  ,%8i\n"  , count            );
   printf("Minimum uS                     ,%8.1f\n", minPCP           );
   printf("Maximum uS                     ,%8.1f\n", maxPCP           );
   printf("Average uS                     ,%8.1f\n", avgPCP           );
   printf("Median  uS                     ,%8.1f\n", medPCP           );
   printf("First   uS                     ,%8.1f\n", firstPCP         );
   printf("Max w/o First                  ,%8.1f\n", maxBut1PCP       );
   printf("Range   uS                     ,%8.1f\n", PCPrange         );
   printf("Histogram Bins chosen          ,%8i\n"  , bins             );
   printf("Bin width uS                   ,%8.1f\n", PCPbin           );
   printf("Mode (center highest Bin Count),%8.1f\n", modePCP          );
   printf("Mode Bin Count                 ,%8i\n"  , actPCPCount      );
   printf("Bin Expected Count             ,%8i\n"  , expCount         );
   printf("\n");
   printf("Read Event Histogram:\n"
          "binCenter, Count, %% of Count\n");

   histogram(ReadBins, bins, count, minPCP, PCPbin);
   printf("\n");
   free(InitBins); InitBins=NULL;                                       // Lose this version of InitBins array.
   free(ReadBins); ReadBins=NULL;                                       // Lose this version of ReadBins array.
   free(fRec);
} // end main.
