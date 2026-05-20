//===--- test_tile.c --------------------------------------------------------===//
//
// OpenMP API Version 5.1 Nov 2020
//
// This test checks the tile directive with required sizes clause. The tile
// directive reorganizes `n` of the nested loops into `2n` tiled loops
// according to the sizes provided in the sizes clause, where `n` is the
// number of sizes provided in the sizes clause. The nested loops under a
// tile directive must be perfectly nested and rectangular, and there must be
// at least `n` of them. The loop tiles are iterated over in lexographic
// order, and two iterations may be reordered by an implementation if at
// least one of them is from a partial tile and if thier logical iteration
// vectors do not have a product order relation. Thus, this test checks that
// a completely-tiled loop nest is tiled in the correct order, but only
// checks results correctness for a tiling with some partial tiles.
//
////===----------------------------------------------------------------------===//
#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include "ompvv.h"

#define N 8

int test_tile_complete() {
  OMPVV_INFOMSG("test_tile_complete");
  int errors = 0;
  int result[N][N];
  int expected[N][N];
  int time = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      result[i][j] = 0;
      expected[i][j] = 0;
    }
  }

#pragma omp tile sizes(4, 4)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      result[i][j] = time++;
    }
  }

  time = 0;

  for (int i1 = 0; i1 < N; i1 += 4) {
    for (int j1 = 0; j1 < N; j1 += 4) {
      for (int i2 = i1; i2 < i1 + 4; i2 += 1) {
        for (int j2 = j1; j2 < j1 + 4; j2 += 1) {
          expected[i2][j2] = time++;
        }
      }
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      OMPVV_TEST_AND_SET(errors, result[i][j] != expected[i][j]);
    }
  }

  return errors;
}

int test_tile_partial() {
  OMPVV_INFOMSG("test_tile_partial");
  int errors = 0;
  int result[N][N];

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      result[i][j] = 1;
    }
  }

#pragma omp tile sizes(3, 3)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      result[i][j] += i*j;
    }
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      OMPVV_TEST_AND_SET_VERBOSE(errors, result[i][j] != 1 + i*j);
    }
  }

  return errors;
}

int main() {
  int errors = 0;

  OMPVV_TEST_AND_SET_VERBOSE(errors, test_tile_complete());
  OMPVV_TEST_AND_SET_VERBOSE(errors, test_tile_partial());

  OMPVV_REPORT_AND_RETURN(errors);
}
