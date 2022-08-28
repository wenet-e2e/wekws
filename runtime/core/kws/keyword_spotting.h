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


#ifndef KWS_KEYWORD_SPOTTING_H_
#define KWS_KEYWORD_SPOTTING_H_

#include <memory>
#include <string>
#include <vector>

#include "onnxruntime_cxx_api.h"  // NOLINT

namespace wekws {

class KeywordSpotting {
 public:
  explicit KeywordSpotting(const std::string& model_path);
  // Call reset if keyword is detected
  void Reset();

  static void InitEngineThreads(int num_threads) {
    session_options_.SetIntraOpNumThreads(num_threads);
    session_options_.SetInterOpNumThreads(num_threads);
  }

  void Forward(const std::vector<std::vector<float>>& feats,
               std::vector<std::vector<float>>* prob);

 private:
  // session
  static Ort::Env env_;
  static Ort::SessionOptions session_options_;
  std::shared_ptr<Ort::Session> session_ = nullptr;
  // node names
  std::vector<const char*> in_names_;
  std::vector<const char*> out_names_;

  // meta info
  int cache_dim_ = 0;
  int cache_len_ = 0;
  // cache info
  Ort::Value cache_ort_{nullptr};
  std::vector<float> cache_;
};


}  // namespace wekws

#endif  // KWS_KEYWORD_SPOTTING_H_
