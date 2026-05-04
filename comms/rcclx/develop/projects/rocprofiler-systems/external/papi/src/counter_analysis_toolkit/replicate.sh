#!/bin/bash

for ((i=1; i<12; i++)); do
    cp icache_seq_kernel_0.so icache_seq_kernel_${i}.so;
done

