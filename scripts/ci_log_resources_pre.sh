#!/bin/bash
# CI instrumentation only. Must never fail the build.
free -g || true
df -h . || true
ulimit -a || true
exit 0
