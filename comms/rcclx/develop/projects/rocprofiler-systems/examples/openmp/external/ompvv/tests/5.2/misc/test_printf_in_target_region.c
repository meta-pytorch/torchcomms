//===------ test_printf_in_target_region.c ----------------------------------===//
//
// OpenMP API Version 5.2 Nov 2021
//
// This test checks that printf inside of a target region is supported
// for all common data types.
//
//
//===------------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"


int main () {

   int isOffloading = 0;
   OMPVV_TEST_AND_SET_OFFLOADING(isOffloading);
   OMPVV_WARNING_IF(isOffloading == 0, "Support for printf in target region cannot be evaluated since there are no available devices");

   int integer = 71;
   float floater = 81.1000;
   double doubler = 643.2390000;
   char single_char = 'S';
   short int shortie = 10000; 
   unsigned short int unsigned_shortie = 56555;
   long int long_one = 194834345;
   unsigned long int unsigned_long = 1777444;
   long long int long_long = 282828293898;

   #pragma omp target map(tofrom: integer, floater, doubler, single_char, shortie, unsigned_shortie, long_one, unsigned_long, long_long) 
   {
      printf("The value of the integer mapped into target region is %d\n", integer);
      printf("The value of the float mapped into target region is %f\n", floater);
      printf("The value of the double mapped into target region is %lf\n", doubler);
      printf("The value of the single char mapped into target region is %c\n", single_char);
      printf("The value of the short int mapped into target region is %hd\n", shortie);
      printf("The value of the unsigned short int mapped into target region is %hu\n", unsigned_shortie);
      printf("The value of the long int mapped into target region is %ld\n", long_one);
      printf("The value of the unsigned long int mapped into target region is %lu\n", unsigned_long);
      printf("The value of the long long int mapped into target region is %lld\n", long_long);
      printf("Values of everything mapped in target region are %d, %f, %lf, %c, %hd, %hu, %ld, %lu, %lld\n", integer, floater, doubler, single_char,
shortie, unsigned_shortie, long_one, unsigned_long, long_long);
   }
   
   return 0;
}
