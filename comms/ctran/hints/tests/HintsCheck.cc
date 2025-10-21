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

TEST(HintsTests, AlltoallvDynamicBool) {
  commResult_t res;

  /* map keys to expected defaults */
  std::unordered_map<std::string, std::string> bool_keys;
  bool_keys["ncclx_alltoallv_dynamic_sendbuffs_contig"] = "false";
  bool_keys["ncclx_alltoallv_dynamic_recvbuffs_contig"] = "false";

  /* check default values */
  for (auto& [key, val] : bool_keys) {
    std::string s;
    meta::comms::Hints hints;

    res = hints.get(key, s);
    EXPECT_EQ(res, commSuccess);
    EXPECT_TRUE(s == val);
  }

  /* set to true and check */
  for (auto& [key, _] : bool_keys) {
    for (const auto& b : t) {
      std::string s;
      meta::comms::Hints hints;

      res = hints.set(key, b);
      EXPECT_EQ(res, commSuccess);
      res = hints.get(key, s);
      EXPECT_EQ(res, commSuccess);

      EXPECT_TRUE(s == "true");
    }
  }

  /* set to false and check */
  for (auto& [key, _] : bool_keys) {
    for (const auto& b : f) {
      std::string s;
      meta::comms::Hints hints;

      res = hints.set(key, b);
      EXPECT_EQ(res, commSuccess);
      res = hints.get(key, s);
      EXPECT_EQ(res, commSuccess);

      EXPECT_TRUE(s == "false");
    }
  }

  /* set to garbage and check */
  for (auto& [key, _] : bool_keys) {
    for (const auto& b : garbage) {
      meta::comms::Hints hints;

      res = hints.set(key, b);
      EXPECT_EQ(res, commInvalidArgument);
    }
  }
}

TEST(HintsTests, AlltoallvDynamicLocation) {
  commResult_t res;

  /* map keys to expected defaults */
  std::unordered_map<std::string, std::string> location_keys;
  location_keys["ncclx_alltoallv_dynamic_sendbuffs_location"] = "auto";
  location_keys["ncclx_alltoallv_dynamic_sendcounts_location"] = "auto";
  location_keys["ncclx_alltoallv_dynamic_recvbuffs_location"] = "auto";
  location_keys["ncclx_alltoallv_dynamic_max_sendcounts_location"] = "auto";
  location_keys["ncclx_alltoallv_dynamic_max_recvcounts_location"] = "auto";
  location_keys["ncclx_alltoallv_dynamic_actual_recvcounts_location"] = "auto";

  /* check default values */
  for (auto& [key, val] : location_keys) {
    std::string s;
    meta::comms::Hints hints;

    res = hints.get(key, s);
    EXPECT_EQ(res, commSuccess);
    EXPECT_TRUE(s == val);
  }

  /* set to different valid options and check */
  for (auto& [key, _] : location_keys) {
    for (auto x : {"cpu", "gpu", "auto"}) {
      std::string s;
      meta::comms::Hints hints;

      res = hints.set(key, x);
      EXPECT_EQ(res, commSuccess);
      res = hints.get(key, s);
      EXPECT_EQ(res, commSuccess);

      EXPECT_TRUE(s == x);
    }
  }

  /* set to an invalid option and check */
  for (auto& [key, _] : location_keys) {
    meta::comms::Hints hints;

    res = hints.set(key, "garbage");
    EXPECT_EQ(res, commInvalidArgument);
  }

  /* set garbage key and check */
  meta::comms::Hints hints;
  res = hints.set("garbage", "true");
  EXPECT_EQ(res, commInvalidArgument);
}
