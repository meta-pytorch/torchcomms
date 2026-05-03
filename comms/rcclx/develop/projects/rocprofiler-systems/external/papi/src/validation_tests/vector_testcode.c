#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#define NUMBER 100

inline void
inline_packed_sse_add( float *aa, float *bb, float *cc )
{
	__asm__ __volatile__( "movaps (%0), %%xmm0;"
						  "movaps (%1), %%xmm1;"
						  "addps %%xmm0, %%xmm1;"
						  "movaps %%xmm1, (%2);"::"r"( aa ),
						  "r"( bb ), "r"( cc )
						  :"%xmm0", "%xmm1" );
}
inline void
inline_packed_sse_mul( float *aa, float *bb, float *cc )
{
	__asm__ __volatile__( "movaps (%0), %%xmm0;"
						  "movaps (%1), %%xmm1;"
						  "mulps %%xmm0, %%xmm1;"
						  "movaps %%xmm1, (%2);"::"r"( aa ),
						  "r"( bb ), "r"( cc )
						  :"%xmm0", "%xmm1" );
}
inline void
inline_packed_sse2_add( double *aa, double *bb, double *cc )
{
	__asm__ __volatile__( "movapd (%0), %%xmm0;"
						  "movapd (%1), %%xmm1;"
						  "addpd %%xmm0, %%xmm1;"
						  "movapd %%xmm1, (%2);"::"r"( aa ),
						  "r"( bb ), "r"( cc )
						  :"%xmm0", "%xmm1" );
}
inline void
inline_packed_sse2_mul( double *aa, double *bb, double *cc )
{
	__asm__ __volatile__( "movapd (%0), %%xmm0;"
						  "movapd (%1), %%xmm1;"
						  "mulpd %%xmm0, %%xmm1;"
						  "movapd %%xmm1, (%2);"::"r"( aa ),
						  "r"( bb ), "r"( cc )
						  :"%xmm0", "%xmm1" );
}
inline void
inline_unpacked_sse_add( float *aa, float *bb, float *cc )
{
	__asm__ __volatile__( "movss (%0), %%xmm0;"
						  "movss (%1), %%xmm1;"
						  "addss %%xmm0, %%xmm1;"
						  "movss %%xmm1, (%2);"::"r"( aa ), "r"( bb ), "r"( cc )
						  :"%xmm0", "%xmm1" );
}
inline void
inline_unpacked_sse_mul( float *aa, float *bb, float *cc )
{
	__asm__ __volatile__( "movss (%0), %%xmm0;"
						  "movss (%1), %%xmm1;"
						  "mulss %%xmm0, %%xmm1;"
						  "movss %%xmm1, (%2);"::"r"( aa ), "r"( bb ), "r"( cc )
						  :"%xmm0", "%xmm1" );
}
inline void
inline_unpacked_sse2_add( double *aa, double *bb, double *cc )
{
	__asm__ __volatile__( "movsd (%0), %%xmm0;"
						  "movsd (%1), %%xmm1;"
						  "addsd %%xmm0, %%xmm1;"
						  "movsd %%xmm1, (%2);"::"r"( aa ), "r"( bb ), "r"( cc )
						  :"%xmm0", "%xmm1" );
}
inline void
inline_unpacked_sse2_mul( double *aa, double *bb, double *cc )
{
	__asm__ __volatile__( "movsd (%0), %%xmm0;"
						  "movsd (%1), %%xmm1;"
						  "mulsd %%xmm0, %%xmm1;"
						  "movsd %%xmm1, (%2);"::"r"( aa ), "r"( bb ), "r"( cc )
						  :"%xmm0", "%xmm1" );
}

int
main( int argc, char **argv )
{
	int i, packed = 0, sse = 0;
	float a[4] = { 1.0, 2.0, 3.0, 4.0 };
	float b[4] = { 2.0, 3.0, 4.0, 5.0 };
	float c[4] = { 0.0, 0.0, 0.0, 0.0 };
	double d[4] = { 1.0, 2.0, 3.0, 4.0 };
	double e[4] = { 2.0, 3.0, 4.0, 5.0 };
	double f[4] = { 0.0, 0.0, 0.0, 0.0 };

	if ( argc != 3 ) {
	  bail:
		printf( "Usage %s: <packed|unpacked> <sse|sse2>\n", argv[0] );
		exit( 1 );
	}
	if ( strcasecmp( argv[1], "packed" ) == 0 )
		packed = 1;
	else if ( strcasecmp( argv[1], "unpacked" ) == 0 )
		packed = 0;
	else
		goto bail;
	if ( strcasecmp( argv[2], "sse" ) == 0 )
		sse = 1;
	else if ( strcasecmp( argv[2], "sse2" ) == 0 )
		sse = 0;
	else
		goto bail;

#if 0
	if ( ( sse ) &&
		 ( system( "cat /proc/cpuinfo | grep sse > /dev/null" ) != 0 ) ) {
		printf( "This processor does not have SSE.\n" );
		exit( 1 );
	}
	if ( ( sse == 0 ) &&
		 ( system( "cat /proc/cpuinfo | grep sse2 > /dev/null" ) != 0 ) ) {
		printf( "This processor does not have SSE2.\n" );
		exit( 1 );
	}
#endif

	printf( "Vector 1: %f %f %f %f\n", a[0], a[1], a[2], a[3] );
	printf( "Vector 2: %f %f %f %f\n\n", b[0], b[1], b[2], b[3] );

	if ( ( packed == 0 ) && ( sse == 1 ) ) {
		for ( i = 0; i < NUMBER; i++ ) {
			inline_unpacked_sse_add( &a[0], &b[0], &c[0] );
		}
		printf( "%d SSE Unpacked Adds: Result %f\n", NUMBER, c[0] );

		for ( i = 0; i < NUMBER; i++ ) {
			inline_unpacked_sse_mul( &a[0], &b[0], &c[0] );
		}
		printf( "%d SSE Unpacked Muls: Result %f\n", NUMBER, c[0] );
	}
	if ( ( packed == 1 ) && ( sse == 1 ) ) {
		for ( i = 0; i < NUMBER; i++ ) {
			inline_packed_sse_add( a, b, c );
		}
		printf( "%d SSE Packed Adds: Result %f %f %f %f\n", NUMBER, c[0], c[1],
				c[2], c[3] );
		for ( i = 0; i < NUMBER; i++ ) {
			inline_packed_sse_mul( a, b, c );
		}
		printf( "%d SSE Packed Muls: Result %f %f %f %f\n", NUMBER, c[0], c[1],
				c[2], c[3] );
	}

	if ( ( packed == 0 ) && ( sse == 0 ) ) {
		for ( i = 0; i < NUMBER; i++ ) {
			inline_unpacked_sse2_add( &d[0], &e[0], &f[0] );
		}
		printf( "%d SSE2 Unpacked Adds: Result %f\n", NUMBER, c[0] );

		for ( i = 0; i < NUMBER; i++ ) {
			inline_unpacked_sse2_mul( &d[0], &e[0], &f[0] );
		}
		printf( "%d SSE2 Unpacked Muls: Result %f\n", NUMBER, c[0] );
	}
	if ( ( packed == 1 ) && ( sse == 0 ) ) {
		for ( i = 0; i < NUMBER; i++ ) {
			inline_packed_sse2_add( &d[0], &e[0], &f[0] );
		}
		printf( "%d SSE2 Packed Adds: Result %f\n", NUMBER, c[0] );

		for ( i = 0; i < NUMBER; i++ ) {
			inline_packed_sse2_mul( &d[0], &e[0], &f[0] );
		}
		printf( "%d SSE2 Packed Muls: Result %f\n", NUMBER, c[0] );
	}


	exit( 0 );
}
