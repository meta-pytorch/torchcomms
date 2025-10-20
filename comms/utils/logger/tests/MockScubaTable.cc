// Copyright (c) Meta Platforms, Inc. and affiliates.
#include "comms/utils/logger/tests/MockScubaTable.h"

const DataTableAllTables createAllMockTables(bool callParent) {
  DataTableAllTables allTables;
  for (const auto& [tableId, tableTypeName] : getAllTableNames()) {
    std::vector<std::string> tokens;
    if (tableTypeName.empty()) {
      continue;
    }
    folly::split(':', tableTypeName, tokens, true /* ignoreEmpty */);
    auto& tableType = tokens.at(0);
    auto& tableName = tokens.at(1);
    auto table =
        std::make_shared<MockScubaTable>(tableType, tableName, callParent);
    allTables.tables[std::string(tableId)] = table;
    allTables.tables[tableName] = table;
  }
  return allTables;
}

void MockScubaTable::writeMessage(const std::string& message) {
  writeCount_++;
  if (callParent_) {
    DataTable::writeMessage(message);
  } else {
    messages_ += message;
    messages_ += "\n";
  }
}

size_t MockScubaTable::getWriteCount() const {
  return writeCount_;
}

std::string MockScubaTable::getMessages() const {
  return messages_;
}

size_t getWriteCount(LoggerEventType type) {
  return std::dynamic_pointer_cast<MockScubaTable>(
             getTablePtrFromEvent(type)->getTable())
      ->getWriteCount();
}

const std::string getMessages(LoggerEventType type) {
  return std::dynamic_pointer_cast<MockScubaTable>(
             getTablePtrFromEvent(type)->getTable())
      ->getMessages();
}
