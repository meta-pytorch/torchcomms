/*
 * This software contains source code provided by NVIDIA Corporation
 * 
 * According to the Nvidia EULA (compute 5.5 version)
 * http://developer.download.nvidia.com/compute/cuda/5_5/rel/docs/EULA.pdf
 * 
 * Chapter 2. NVIDIA CORPORATION CUDA SAMPLES END USER LICENSE AGREEMENT
 * 2.1.1. Source Code
 * Developer shall have the right to modify and create derivative works with the Source
 * Code. Developer shall own any derivative works ("Derivatives") it creates to the Source
 * Code, provided that Developer uses the Materials in accordance with the terms and
 * conditions of this Agreement. Developer may distribute the Derivatives, provided that
 * all NVIDIA copyright notices and trademarks are propagated and used properly and
 * the Derivatives include the following statement: “This software contains source code
 * provided by NVIDIA Corporation.”
 */

#ifndef TIMER_H
#define TIMER_H

#include <stdlib.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#endif

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
double PCFreq = 0.0;
__int64 timerStart = 0;
#else
struct timeval timerStart;
#endif

void StartTimer()
{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    LARGE_INTEGER li;

    if (!QueryPerformanceFrequency(&li))
    {
        printf("QueryPerformanceFrequency failed!\n");
    }

    PCFreq = (double)li.QuadPart/1000.0;
    QueryPerformanceCounter(&li);
    timerStart = li.QuadPart;
#else
    gettimeofday(&timerStart, NULL);
#endif
}

// time elapsed in ms
double GetTimer()
{
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
    LARGE_INTEGER li;
    QueryPerformanceCounter(&li);
    return (double)(li.QuadPart-timerStart)/PCFreq;
#else
    struct timeval timerStop, timerElapsed;
    gettimeofday(&timerStop, NULL);
    timersub(&timerStop, &timerStart, &timerElapsed);
    return timerElapsed.tv_sec*1000.0+timerElapsed.tv_usec/1000.0;
#endif
}
#endif // TIMER_H

