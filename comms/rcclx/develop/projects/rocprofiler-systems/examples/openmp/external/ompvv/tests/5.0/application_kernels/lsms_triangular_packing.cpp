/*
 * Copyright (C) 2014, The University of Texas at Austin
 * Copyright (C) 2014-2015, Michael Lehn
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *  - Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  - Neither the name of The University of Texas at Austin nor the names
 *    of its contributors may be used to endorse or promote products
 *    derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <complex>
#include <cstdio>
#include <omp.h>
#include "ompvv.h"

const int MR = 16;
const int M = 256;
const int N = 256;

static_assert(!(M % MR), "Block size must be divisible by the matrix size");

namespace ulmBLAS {

/* Perform a triangular packing algorithm on an input matrix. Converting it to
 * a lower unit triangular matrix and packing column blocks of size MR x MR into
 * an auxilliary buffer.
 *
 *  _|--- MR ----|
 *  | 1   2  3  4  5  6  7  8       1   0  0  0  0  0  0  0
 * MR 9  10 11 12 13 14 15 16       9   1  0  0  0  0  0  0
 *  | 17 18 19 20 21 22 23 24       17 18  1  0  0  0  0  0
 *  - 25 26 27 28 29 30 31 32  -->  25 26 27  1  0  0  0  0
 *    33 34 35 36 37 38 39 40       33 34 35 36  1  0  0  0
 *    41 42 43 44 45 46 47 48       41 42 43 44 45  1  0  0
 *    49 50 51 52 53 54 55 56       49 50 51 52 53 54  1  0
 *    57 58 59 60 61 62 63 64       57 58 59 60 61 62 63  1
 *      
 *  Packed into a collection of MR by MR blocks stored column-major in memory
 *
 *  1  9 17 25      33 41 49 57      1 45 53 61  
 *  0  1 18 26  ... 34 42 50 58 ...  0  1 54 61  
 *  0  0  1 27      35 43 51 59      0  0  1 63  
 *  0  0  0  1      36 44 52 60      0  0  0  1
 *
 */

template <typename IndexType, typename TL, typename Buffer>
void
trlspack(IndexType   mc,
         bool        unit,
         const TL    *L,
         IndexType   incRowL,
         IndexType   incColL,
         Buffer      *p)
{
    OMPVV_INFOMSG("app_kernel_lsms_triangular_packing");
    IndexType mp = (mc+MR-1) / MR;

// Collapsing the whole loop nest here causes errors
#pragma omp target teams distribute parallel for collapse(4)
    for (IndexType j=0; j<mp; ++j) {
// Collapsing here is fine
//#pragma omp target teams distribute parallel for collapse(3)
        for (IndexType j0=0; j0<MR; ++j0) {
            for (IndexType i=j; i<mp; ++i) {
                for (IndexType i0=0; i0<MR; ++i0) {
                    IndexType I  = i*MR+i0;
                    IndexType J  = j*MR+j0;
                    IndexType nu = (i+1)*i/2*MR*MR + j*MR*MR + j0*MR +i0;
                    p[nu] = (I==J && unit)
                            ? Buffer(1)
                          : (I==J && !unit)
                            ? Buffer(1) / L[I*(incRowL+incColL)]
                          : (I>=mc || J>=mc)
                            ? Buffer(0)
                          : (I>J)
                            ? L[I*incRowL+J*incColL]
                          : Buffer(0);
                }
            }
        }
    }
}

} // namespace ulmBLAS

int main() {
    const bool unit = true;

    double A[M*N];
    double buffer[M*M + MR];

    OMPVV_TEST_OFFLOADING;

    for (int i = 0; i < M; i++) 
        for (int j = 0; j < N; j++) 
            A[i*N + j] = (i == j) ? 1.0 
                       : (j >  i) ? 0.0
                       : drand48()*2.0 - 1.0;

#pragma omp target data           \
    map(to:A[0 : M*N])            \
    map(from:buffer[0 : M*M + MR])
    {
    ulmBLAS::trlspack(M, unit, A, N, 1, buffer);
    }

    double error_sum = 0.0;
    int mp = (M + MR - 1) / MR;
    for (int j = 0; j < mp; j++) {
        for (int j0 = 0; j0 < MR; j0++) {
            for (int i = j; i < mp; i++) {
                for (int i0 = 0; i0 < MR; i0++) {
                    int I  = i*MR+i0;
                    int J  = j*MR+j0;
                    int nu = (i+1)*i/2*MR*MR + j*MR*MR + j0*MR +i0;
                    OMPVV_TEST_AND_SET(error_sum,buffer[nu] != A[I*N + J]);
                }
            }
        }
    }

    OMPVV_REPORT_AND_RETURN(error_sum != 0.0);
}

