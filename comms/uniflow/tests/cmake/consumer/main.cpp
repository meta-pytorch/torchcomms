// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <comms/uniflow/Uniflow.h>

int main() {
  uniflow::UniflowAgentConfig config;
  uniflow::UniflowAgent agent(config);
  agent.shutdown();
  return 0;
}
