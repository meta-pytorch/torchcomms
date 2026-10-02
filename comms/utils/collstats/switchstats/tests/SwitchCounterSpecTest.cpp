// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <map>
#include <string>

// boost::regex on purpose, and only here: fb303 compiles the pattern with it
// (fb303/ServiceData.cpp), so matching through any other engine would assert
// against a grammar the switch will not use. The performance objection the
// linter raises is about serving traffic; this is a fixed pattern over a
// handful of literals in a unit test.
// @lint-ignore CLANGTIDY facebook-hte-BadInclude-boost/regex.hpp
#include <boost/regex.hpp>
#include <gtest/gtest.h>

#include "comms/utils/collstats/switchstats/SwitchCounterSource.h"

namespace meta::comms::switchstats {

namespace {

// A real port name from a switch this was built against.
constexpr std::string_view kPort = "eth1/5/1";

std::string counterName(
    const SwitchCounterSpec& spec,
    std::string_view port,
    SwitchMeasurement measurement) {
  return std::string(port) + "." + counterSuffix(spec, measurement) + ".sum";
}

} // namespace

TEST(SwitchCounterSpecTest, TheRegexMatchesEveryCounterItAsksFor) {
  const SwitchCounterSpec spec;
  // @lint-ignore CLANGTIDY facebook-hte-BoostRegexRisky
  const boost::regex re(countersRegex(spec));

  for (const SwitchMeasurementEntry& entry : kMeasurements) {
    const std::string name = counterName(spec, kPort, entry.measurement);
    EXPECT_TRUE(boost::regex_match(name, re)) << name;
  }
}

TEST(SwitchCounterSpecTest, EveryCounterTheRegexAsksForParsesBack) {
  const SwitchCounterSpec spec;
  for (const SwitchMeasurementEntry& entry : kMeasurements) {
    const std::string name = counterName(spec, kPort, entry.measurement);
    const auto parsed = parseCounterName(spec, name);
    ASSERT_TRUE(parsed.has_value()) << name;
    EXPECT_EQ(parsed->port, kPort);
    EXPECT_EQ(parsed->measurement, entry.measurement);
  }
}

// The whole reason the regex and the parser are generated from one table.
TEST(SwitchCounterSpecTest, ARespecifiedQueueMovesBothTheRegexAndTheParse) {
  SwitchCounterSpec spec;
  spec.losslessQueueSegment = "queue5.other";

  // @lint-ignore CLANGTIDY facebook-hte-BoostRegexRisky
  const boost::regex re(countersRegex(spec));

  const std::string moved =
      counterName(spec, kPort, SwitchMeasurement::OutBytes);
  EXPECT_EQ(moved, "eth1/5/1.queue5.other.out_bytes.sum");
  EXPECT_TRUE(boost::regex_match(moved, re));
  EXPECT_TRUE(parseCounterName(spec, moved).has_value());

  // And the default queue's counter is now neither asked for nor accepted.
  const std::string oldName = "eth1/5/1.queue2.rdma.out_bytes.sum";
  EXPECT_FALSE(boost::regex_match(oldName, re));
  EXPECT_FALSE(parseCounterName(spec, oldName).has_value());
}

// fb303 full-matches the pattern, so an ungrouped alternation would turn the
// first alternative into "the key is exactly this port name" and silently drop
// every counter on that port.
TEST(SwitchCounterSpecTest, AnAlternationInThePortPatternStaysInsideIt) {
  SwitchCounterSpec spec;
  spec.portPattern = "eth1/5/1|eth1/6/1";
  // @lint-ignore CLANGTIDY facebook-hte-BoostRegexRisky
  const boost::regex re(countersRegex(spec));

  for (const std::string_view port : {"eth1/5/1", "eth1/6/1"}) {
    const std::string name =
        counterName(spec, port, SwitchMeasurement::InErrors);
    EXPECT_TRUE(boost::regex_match(name, re)) << name;
  }
  EXPECT_FALSE(
      boost::regex_match(
          counterName(spec, "eth1/7/1", SwitchMeasurement::InErrors), re));
}

TEST(SwitchCounterSpecTest, ThePriorityFollowsTheSpec) {
  SwitchCounterSpec spec;
  spec.losslessPriority = 3;
  EXPECT_EQ(
      counterName(spec, kPort, SwitchMeasurement::OutPfcFrames),
      "eth1/5/1.out_pfc_frames.priority3.sum");
}

// `out_congestion_discards` is a prefix of `out_congestion_discards_bytes`, so
// a parser matching on the shorter one first would charge the longer counter's
// value to the shorter measurement.
TEST(SwitchCounterSpecTest, ALongerLeafIsNotTakenForTheShorterOneInsideIt) {
  const SwitchCounterSpec spec;
  const auto parsed = parseCounterName(
      spec, "eth1/5/1.queue2.rdma.out_congestion_discards_bytes.sum");
  ASSERT_TRUE(parsed.has_value());
  EXPECT_EQ(parsed->measurement, SwitchMeasurement::OutCongestionDiscardBytes);
}

// The windowed and rate forms are what this feature exists to avoid: taking one
// would pin an interval's resolution to the agent's window instead of to the
// gap between two reads.
TEST(
    SwitchCounterSpecTest,
    TheWindowedAndRateFormsAreNeitherAskedForNorParsed) {
  const SwitchCounterSpec spec;
  // @lint-ignore CLANGTIDY facebook-hte-BoostRegexRisky
  const boost::regex re(countersRegex(spec));

  for (const std::string& name :
       {std::string("eth1/5/1.queue2.rdma.out_bytes.sum.60"),
        std::string("eth1/5/1.queue2.rdma.out_bytes.sum.600"),
        std::string("eth1/5/1.queue2.rdma.out_bytes.sum.3600"),
        std::string("eth1/5/1.queue2.rdma.out_bytes.rate"),
        std::string("eth1/5/1.queue2.rdma.out_bytes.rate.60")}) {
    EXPECT_FALSE(boost::regex_match(name, re)) << name;
    EXPECT_FALSE(parseCounterName(spec, name).has_value()) << name;
  }
}

TEST(SwitchCounterSpecTest, AKeyThatWasNotAskedForParsesToNothing) {
  const SwitchCounterSpec spec;
  EXPECT_FALSE(parseCounterName(spec, "").has_value());
  EXPECT_FALSE(parseCounterName(spec, "in_errors.sum").has_value());
  EXPECT_FALSE(
      parseCounterName(spec, "eth1/5/1.in_unicast_pkts.sum").has_value());
  // A queue this spec does not name.
  EXPECT_FALSE(parseCounterName(spec, "eth1/5/1.queue0.default.out_bytes.sum")
                   .has_value());
}

// The distinction the whole status enum exists for.
TEST(SwitchCounterSpecTest, AnIdleCounterIsZeroAndAMissingOneIsUnsupported) {
  const SwitchCounterSpec spec;
  const std::map<std::string, int64_t> counters{
      {counterName(spec, kPort, SwitchMeasurement::OutBytes), 0},
      {counterName(spec, kPort, SwitchMeasurement::InErrors), 7},
  };

  const SwitchReading reading = readingFromCounters(
      spec, counters, SwitchReadMechanism::Fb303GetRegexCounters);
  ASSERT_TRUE(reading.ok);
  ASSERT_EQ(reading.ports.size(), 1u);
  const SwitchPortReading& port = reading.ports.at(std::string(kPort));

  // Exported and zero: an idle link.
  EXPECT_EQ(
      port.status[measurementIndex(SwitchMeasurement::OutBytes)],
      SwitchCounterStatus::Ok);
  EXPECT_EQ(port.values[measurementIndex(SwitchMeasurement::OutBytes)], 0);

  EXPECT_EQ(
      port.status[measurementIndex(SwitchMeasurement::InErrors)],
      SwitchCounterStatus::Ok);
  EXPECT_EQ(port.values[measurementIndex(SwitchMeasurement::InErrors)], 7);

  // Asked for and not exported. Not zero.
  EXPECT_EQ(
      port.status[measurementIndex(SwitchMeasurement::OutEcnCounter)],
      SwitchCounterStatus::Unsupported);
  EXPECT_EQ(
      port.values[measurementIndex(SwitchMeasurement::OutEcnCounter)],
      kSwitchCounterNoValue);
}

TEST(SwitchCounterSpecTest, EachPortGetsItsOwnValues) {
  const SwitchCounterSpec spec;
  const std::map<std::string, int64_t> counters{
      {counterName(spec, "eth1/5/1", SwitchMeasurement::OutBytes), 100},
      {counterName(spec, "eth2/6/1", SwitchMeasurement::OutBytes), 200},
  };
  const SwitchReading reading = readingFromCounters(
      spec, counters, SwitchReadMechanism::Fb303GetRegexCounters);

  ASSERT_EQ(reading.ports.size(), 2u);
  const uint32_t i = measurementIndex(SwitchMeasurement::OutBytes);
  EXPECT_EQ(reading.ports.at("eth1/5/1").values[i], 100);
  EXPECT_EQ(reading.ports.at("eth2/6/1").values[i], 200);
}

} // namespace meta::comms::switchstats
