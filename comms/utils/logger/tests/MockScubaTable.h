// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "comms/utils/logger/ScubaLogger.h"

class MockScubaTable : public DataTable {
 public:
  MockScubaTable(
      const std::string& tableType,
      const std::string& tableName,
      bool callParent)
      : DataTable(tableType, tableName), callParent_(callParent) {}

  size_t getWriteCount() const;
  std::string getMessages() const;

 protected:
  void writeMessage(const std::string& message) override;

 private:
  int writeCount_{0};
  std::string messages_{};
  bool callParent_{false};
};

const DataTableAllTables createAllMockTables(bool callParent = false);
size_t getWriteCount(LoggerEventType type);
const std::string getMessages(LoggerEventType type);
