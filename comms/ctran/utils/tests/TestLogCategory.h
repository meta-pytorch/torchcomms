// Copyright (c) Meta Platforms, Inc. and affiliates.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include <folly/logging/LogCategory.h>
#include <folly/logging/LogFormatter.h>
#include <folly/logging/LogHandler.h>
#include <folly/logging/LogMessage.h>
#include <folly/logging/LogStream.h>
#include <folly/logging/LoggerDB.h>
#include <folly/logging/StandardLogHandler.h>
#include <folly/logging/StandardLogHandlerFactory.h>
#include <folly/logging/xlog.h>

// Custom log handler to capture log messages for testing
class CaptureLogHandler : public folly::LogHandler {
 public:
  explicit CaptureLogHandler(std::shared_ptr<folly::LogFormatter> formatter)
      : formatter_(std::move(formatter)){};

  void handleMessage(
      const folly::LogMessage& message,
      const folly::LogCategory* handlerCategory) override {
    printf("Processing Log: %s\n", message.getMessage().c_str());
    auto formattedMsg = formatter_->formatMessage(message, handlerCategory);
    printf("Formatted Log: %s\n", formattedMsg.c_str());
    messages_.push_back(formattedMsg);
  }

  void flush() override {}

  folly::LogHandlerConfig getConfig() const override {
    return folly::LogHandlerConfig();
  };

  const std::vector<std::string>& getMessages() const {
    return messages_;
  }

  void clearMessages() {
    messages_.clear();
  }

 private:
  std::vector<std::string> messages_;
  std::shared_ptr<folly::LogFormatter> formatter_;
};

/**
 * Helper base class for Ctran logging tests. Sets up a test log category and
 * handler to capture the log in test.
 */
class TestLogCategory {
 public:
  TestLogCategory() = default;

  bool setup(folly::LogCategory* category) {
    // Using the same formatter Ctran logging used
    const auto ctranCategory = getCtranCategory();
    if (!ctranCategory) {
      return false;
    }
    if (ctranCategory->getHandlers().size() <= 0) {
      return false;
    }

    std::shared_ptr<folly::LogFormatter> formatter = nullptr;
    auto& handler = *ctranCategory->getHandlers().front();
    if (typeid(handler) == typeid(folly::StandardLogHandler)) {
      formatter =
          static_cast<folly::StandardLogHandler&>(handler).getFormatter();
    } else {
      return false;
    }

    captureHandler_ = std::make_shared<CaptureLogHandler>(std::move(formatter));
    category->addHandler(captureHandler_);

    // Save the current log level and set it to INFO for testing
    oldLevel_ = category->getLevel();
    category->setLevel(folly::LogLevel::INFO);

    category_ = category;
    return true;
  }

  void reset() {
    captureHandler_->clearMessages();
    category_->setLevel(oldLevel_);
    category_->clearHandlers();
  }

  const std::vector<std::string>& getMessages() const {
    return captureHandler_->getMessages();
  }

  void clearMessages() {
    captureHandler_->clearMessages();
  }

 private:
  folly::LogCategory* getCtranCategory() {
    auto categoryName = XLOG_GET_CATEGORY_NAME();
    while (true) {
      auto category = folly::LoggerDB::get().getCategory(categoryName);
      if (category->getHandlers().size() > 0) {
        return category;
      }
      auto newCategoryName = folly::LogName::getParent(categoryName);
      if (categoryName == newCategoryName) {
        return nullptr;
      }
      categoryName = newCategoryName;
    }
    return nullptr;
  }

  folly::LogCategory* category_{nullptr};
  std::shared_ptr<CaptureLogHandler> captureHandler_{nullptr};
  folly::LogLevel oldLevel_;
};
