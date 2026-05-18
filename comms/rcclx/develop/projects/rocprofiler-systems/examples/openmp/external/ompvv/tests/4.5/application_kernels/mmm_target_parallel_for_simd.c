//===--- mmm_target_parallel_for_simd.c--- MM on a target parallel for simd--===//
//
// OpenMP API Version 4.5 Nov 2015
//
//  DESCRIPTION:  
//    OpenMp Example - Matrix Multiply - C Version
//    Demonstrates a matrix multiply using OpenMP. Threads share row iterations
//  
//  Last modified by Jose M Monsalve Diaz, December 25, 2019
//
////===----------------------------------------------------------------------===//
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "ompvv.h"

#define rowA 500        
#define colA 500        
#define colB 500        


int main (int argc, char *argv[]) 
{
  OMPVV_TEST_OFFLOADING;
  int tid, nthreads, i, j, k;
  int	*a = (int*) malloc(sizeof(int) * rowA * colA);           // matrix A to be multiplied
  int	*b = (int*) malloc(sizeof(int) * colA * colB);           // matrix B to be multiplied 
  int	*c = (int*) malloc(sizeof(int) * rowA * colB);           // result matrix C 

  // Initialize matrices
  for (i = 0; i < rowA; i++)
    for (j = 0; j < colA; j++)
      a[i*rowA+j] = 10; // i+j;
  for (i = 0; i < colA; i++)
    for (j = 0; j < colB; j++)
      b[i*colA+j] = 50; //i*j;
  for (i = 0; i < rowA; i++)
    for (j = 0; j < colB; j++)
      c[i*rowA+j] = 0;

  int DimA = rowA*colA;
  int DimB = colB*colA;
  int DimC = rowA*colA;

#pragma omp target map(to: a[0:DimA], b[0:DimB]) map(from: c[0:DimC])
  {
#pragma omp teams distribute parallel for simd collapse(2) private(k)
    for (i = 0; i < rowA; i++)
      for(j = 0; j < colB; j++)
        for(k = 0; k < colA; k++)
          c[i*rowA+j] = a[i*rowA+j] * b[k*colA+j];
  }//end-target

  // Check results
  int error = 0;
  for (i = 0; i < rowA; i++)
  {
    for (j = 0; j < colB; j++) {
      OMPVV_TEST_AND_SET(error, 500 != c[i*rowA+j]);
      OMPVV_ERROR_IF(500 != c[i*rowA+j], "Error: [%d][%d] should be 500 is %d",i,j,c[i*rowA+j]);
    }
  }
  free(a);
  free(b);
  free(c);

  OMPVV_REPORT_AND_RETURN(error);
}

