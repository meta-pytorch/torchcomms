# Copyright (c) Meta Platforms, Inc. and affiliates.

uniflow_add_unit_test(uniflow_logger_test LoggerTest.cpp)

# Separate binary: this one shuts spdlog's global registry down, which no other
# test in the same process could survive logging after.
uniflow_add_unit_test(uniflow_logger_shutdown_test LoggerShutdownTest.cpp)

