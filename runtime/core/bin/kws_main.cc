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

#include <iostream>
#include <string>

#include "frontend/feature_pipeline.h"
#include "frontend/wav.h"
#include "kws/keyword_spotting.h"
#include "utils/log.h"

int main(int argc, char* argv[]) {
  if (argc != 5) {
    LOG(FATAL) << "Usage: kws_main fbank_dim(int) batch_size(int) "
               << "kws_model_path test_wav_path";
  }

  const int num_bins = std::stoi(argv[1]);  // Fbank feature dim
  const int batch_size = std::stoi(argv[2]);
  const std::string model_path = argv[3];
  const std::string wav_path = argv[4];

  wenet::WavReader wav_reader(wav_path);
  int num_samples = wav_reader.num_samples();
  wenet::FeaturePipelineConfig feature_config(num_bins, 16000);
  wenet::FeaturePipeline feature_pipeline(feature_config);
  std::vector<float> wav(wav_reader.data(), wav_reader.data() + num_samples);
  feature_pipeline.AcceptWaveform(wav);
  feature_pipeline.set_input_finished();

  wekws::KeywordSpotting spotter(model_path);

  // Simulate streaming, detect batch by batch
  int offset = 0;
  while (true) {
    std::vector<std::vector<float>> feats;
    bool ok = feature_pipeline.Read(batch_size, &feats);
    std::vector<std::vector<float>> prob;
    spotter.Forward(feats, &prob);
    for (int i = 0; i < prob.size(); i++) {
      std::cout << "frame " << offset + i << " prob";
      for (int j = 0; j < prob[i].size(); j++) {
        std::cout << " " << prob[i][j];
      }
      std::cout << std::endl;
    }
    // Reach the end of feature pipeline
    if (!ok) break;
    offset += prob.size();
  }
  return 0;
}
