#!/bin/bash
rm -f externals.h.tmp
for y in $(seq 0 $1); do for x in $(seq 0 $2); do echo extern template class Mult\<${y}, ${x}\>\;; done; done > externals.h.tmp
for y in $(seq 0 $1); do for x in $(seq 0 $2); do echo extern template class Add\<${y}, ${x}\>\;; done; done >> externals.h.tmp
if ! [ -f externals.h ] || ! diff -q externals.h externals.h.tmp; then mv -f externals.h.tmp externals.h; fi
rm -f externals.h.tmp

