// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <string>
#include "comms/ctran/hints/Hints.h"

std::vector<std::string> t = {
    "true",
    "True",
    "tRue",
    "TRUE",
    "t",
    "T",
    "yes",
    "Yes",
    "yEs",
    "YES",
    "y",
    "Y",
    "1"};
std::vector<std::string> f = {
    "false",
    "False",
    "fAlse",
    "FALSE",
    "f",
    "F",
    "no",
    "No",
    "nO",
    "NO",
    "n",
    "N",
    "0"};
std::vector<std::string> garbage = {"garbage", "true garbage"};

TEST(HintsTests, GarbageKey) {
  meta::comms::Hints hints;
  commResult_t res = hints.set("garbage", "true");
  EXPECT_EQ(res, commInvalidArgument);
}
