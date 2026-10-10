#!/bin/bash
# CI instrumentation only. Must never fail the build.
du -sh build/ncclx/obj build/ncclx/lib 2>/dev/null || true
free -g || true
df -h . || true
echo "=== memory high-water mark ===" || true
cat /sys/fs/cgroup/memory.peak 2>/dev/null \
  || cat /sys/fs/cgroup/memory/memory.max_usage_in_bytes 2>/dev/null || true
echo "=== oom_kill counter ===" || true
cat /sys/fs/cgroup/memory.events 2>/dev/null | grep -E "max|oom" \
  || cat /sys/fs/cgroup/memory/memory.oom_control 2>/dev/null || true
exit 0
