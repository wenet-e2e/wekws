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


#include "kws/keyword_spotting.h"

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace wekws {

Ort::Env KeywordSpotting::env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "");
Ort::SessionOptions KeywordSpotting::session_options_ = Ort::SessionOptions();

KeywordSpotting::KeywordSpotting(const std::string& model_path) {
  // 1. Load sessions
  session_ = std::make_shared<Ort::Session>(env_, model_path.c_str(),
                                            session_options_);
  // 2. Model info
  in_names_ = {"input", "cache"};
  out_names_ = {"output", "r_cache"};
  auto metadata = session_->GetModelMetadata();
  Ort::AllocatorWithDefaultOptions allocator;
  cache_dim_ = std::stoi(metadata.LookupCustomMetadataMap("cache_dim",
                                                          allocator));
  cache_len_ = std::stoi(metadata.LookupCustomMetadataMap("cache_len",
                                                          allocator));
  std::cout << "Kws Model Info:" << std::endl
            << "\tcache_dim: " << cache_dim_ << std::endl
            << "\tcache_len: " << cache_len_ << std::endl;
  Reset();
}


void KeywordSpotting::Reset() {
  Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  cache_.resize(cache_dim_ * cache_len_, 0.0);
  const int64_t cache_shape[] = {1, cache_dim_, cache_len_};
  cache_ort_ = Ort::Value::CreateTensor<float>(
      memory_info, cache_.data(), cache_.size(), cache_shape, 3);
}


void KeywordSpotting::Forward(
    const std::vector<std::vector<float>>& feats,
    std::vector<std::vector<float>>* prob) {
  prob->clear();
  if (feats.size() == 0) return;
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  // 1. Prepare input
  int num_frames = feats.size();
  int feature_dim = feats[0].size();
  std::vector<float> slice_feats;
  for (int i = 0; i < feats.size(); i++) {
    slice_feats.insert(slice_feats.end(), feats[i].begin(), feats[i].end());
  }
  const int64_t feats_shape[3] = {1, num_frames, feature_dim};
  Ort::Value feats_ort = Ort::Value::CreateTensor<float>(
      memory_info, slice_feats.data(), slice_feats.size(), feats_shape, 3);
  // 2. Ort forward
  std::vector<Ort::Value> inputs;
  inputs.emplace_back(std::move(feats_ort));
  inputs.emplace_back(std::move(cache_ort_));
  // ort_outputs.size() == 2
  std::vector<Ort::Value> ort_outputs = session_->Run(
      Ort::RunOptions{nullptr}, in_names_.data(), inputs.data(),
      inputs.size(), out_names_.data(), out_names_.size());

  // 3. Update cache
  cache_ort_ = std::move(ort_outputs[1]);

  // 4. Get keyword prob
  float* data = ort_outputs[0].GetTensorMutableData<float>();
  auto type_info = ort_outputs[0].GetTensorTypeAndShapeInfo();
  int num_outputs = type_info.GetShape()[1];
  int output_dim = type_info.GetShape()[2];
  prob->resize(num_outputs);
  for (int i = 0; i < num_outputs; i++) {
     (*prob)[i].resize(output_dim);
     memcpy((*prob)[i].data(), data + i * output_dim,
            sizeof(float) * output_dim);
  }
}

}  // namespace wekws
