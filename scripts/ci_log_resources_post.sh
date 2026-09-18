#!/bin/bash
# CI instrumentation only. Must never fail the build.
du -sh build/ncclx/obj build/ncclx/lib 2>/dev/null || true
free -g || true
df -h . || true
exit 0
