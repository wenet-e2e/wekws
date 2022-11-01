// Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
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

#include <signal.h>
#include <iomanip>
#include <iostream>
#include <string>

#include "portaudio.h"  // NOLINT

#include "frontend/feature_pipeline.h"
#include "frontend/wav.h"
#include "kws/keyword_spotting.h"
#include "utils/log.h"

int g_exiting = 0;
std::shared_ptr<wenet::FeaturePipeline> g_feature_pipeline;

void SigRoutine(int dunno) {
  if (dunno == SIGINT) {
    g_exiting = 1;
  }
}

static int RecordCallback(const void* input, void* output,
                          unsigned long frames_count,  // NOLINT
                          const PaStreamCallbackTimeInfo* time_info,
                          PaStreamCallbackFlags status_flags, void* user_data) {
  const auto* pcm_data = static_cast<const int16_t*>(input);
  std::vector<int16_t> v(pcm_data, pcm_data + frames_count);
  g_feature_pipeline->AcceptWaveform(v);

  if (g_exiting) {
    LOG(INFO) << "Exiting loop.";
    g_feature_pipeline->set_input_finished();
    return paComplete;
  } else {
    return paContinue;
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    LOG(FATAL) << "Usage: kws_main fbank_dim batch_size kws_model_path";
  }
  const int num_bins = std::stoi(argv[1]);  // Fbank feature dim
  const int batch_size = std::stoi(argv[2]);
  const std::string model_path = argv[3];

  wenet::FeaturePipelineConfig feature_config(num_bins, 16000);
  g_feature_pipeline = std::make_shared<wenet::FeaturePipeline>(feature_config);
  wekws::KeywordSpotting spotter(model_path);

  signal(SIGINT, SigRoutine);
  PaError err = Pa_Initialize();
  PaStreamParameters params;
  std::cout << err << " " << Pa_GetDeviceCount() << std::endl;
  params.device = Pa_GetDefaultInputDevice();
  if (params.device == paNoDevice) {
    LOG(FATAL) << "Error: No default input device.";
  }
  params.channelCount = 1;
  params.sampleFormat = paInt16;
  params.suggestedLatency =
      Pa_GetDeviceInfo(params.device)->defaultLowInputLatency;
  params.hostApiSpecificStreamInfo = NULL;
  PaStream* stream;
  // Callback and spot pcm date each `interval` ms.
  int interval = 500;
  int frames_per_buffer = 16000 / 1000 * interval;
  Pa_OpenStream(&stream, &params, NULL, 16000, frames_per_buffer, paClipOff,
                RecordCallback, NULL);
  Pa_StartStream(stream);
  LOG(INFO) << "=== Now recording!! Please speak into the microphone. ===";

  std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(2);
  while (Pa_IsStreamActive(stream)) {
    Pa_Sleep(interval);
    std::vector<std::vector<float>> feats;
    g_feature_pipeline->Read(batch_size, &feats);
    std::vector<std::vector<float>> prob;
    spotter.Forward(feats, &prob);
    for (int t = 0; t < prob.size(); t++) {
      std::cout << "keywords prob:";
      for (int i = 0; i < prob[t].size(); i++) {
        std::cout << " kw[" << i << "] " << prob[t][i];
      }
      std::cout << std::endl;
    }
  }
  Pa_CloseStream(stream);
  Pa_Terminate();

  return 0;
}
