// Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#ifndef UTILS_LOG_H_
#define UTILS_LOG_H_

#include <stdlib.h>

#include <iostream>
#include <sstream>

namespace wenet {

const int INFO = 0, WARNING = 1, ERROR = 2, FATAL = 3;

class Logger {
 public:
  Logger(int severity, const char* func, const char* file, int line) {
    severity_ = severity;
    switch (severity) {
    case INFO:
      ss_ << "INFO (";
      break;
    case WARNING:
      ss_ << "WARNING (";
      break;
    case ERROR:
      ss_ << "ERROR (";
      break;
    case FATAL:
      ss_ << "FATAL (";
      break;
    default:
      severity_ = FATAL;
      ss_ << "FATAL (";
    }
    ss_ << func << "():" << file << ':' << line << ") ";
  }

  ~Logger() {
    std::cerr << ss_.str() << std::endl << std::flush;
    if (severity_ == FATAL) {
      abort();
    }
  }

  template <typename T> Logger& operator<<(const T &val) {
    ss_ << val;
    return *this;
  }

 private:
  int severity_;
  std::ostringstream ss_;
};

#define LOG(severity) ::wenet::Logger( \
    ::wenet::severity, __func__, __FILE__, __LINE__)

#define CHECK(test) \
do { \
  if (!(test)) { \
    std::cerr << "CHECK (" << __func__ << "():" << __FILE__ << ":" \
              << __LINE__ << ") " << #test << std::endl; \
    exit(-1); \
  } \
} while (0)

}  // namespace wenet

#endif  // UTILS_LOG_H_
