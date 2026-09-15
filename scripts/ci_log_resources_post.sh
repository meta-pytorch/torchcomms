#!/bin/bash
# CI instrumentation only. Must never fail the build.
du -sh build/ncclx/obj build/ncclx/lib 2>/dev/null || true
free -g || true
df -h . || true
# Cumulative sccache stats for the build (one server, SCCACHE_IDLE_TIMEOUT=0).
# These are how a silent all-miss regression gets caught: without them a cache
# that hits nothing looks the same as no cache at all. Stopping the server is
# hygiene on reused self-hosted runners.
#
# Gated on the marker _build_wheel.sh writes when it actually enabled the cache:
# this shell does not inherit its SCCACHE_BUCKET/SCCACHE_DIR, and a bare
# `command -v sccache` would also match a binary installed for something else —
# stopping that server would be someone else's build going cold, or a
# default-configured server started here just to be torn down.
if [ -f "${GITHUB_WORKSPACE:-$PWD}/.sccache-enabled" ] && command -v sccache > /dev/null 2>&1; then
  sccache --show-stats || true
  tail -50 "${GITHUB_WORKSPACE:-$PWD}/sccache-error.log" 2>/dev/null || true
  sccache --stop-server || true
fi
exit 0
