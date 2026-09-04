#!/bin/bash
# CI instrumentation only. Must never fail the build.
free -g || true
df -h . || true
ulimit -a || true
echo "=== cgroup placement ===" || true
cat /proc/self/cgroup 2>/dev/null | head -5 || true
echo "=== memory limit (v2 then v1) ===" || true
cat /sys/fs/cgroup/memory.max 2>/dev/null \
  || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true
echo "=== mounts ===" || true
df -h / /tmp /dev/shm . 2>/dev/null || true
nproc || true
exit 0
