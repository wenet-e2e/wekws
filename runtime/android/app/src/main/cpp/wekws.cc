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
#include <jni.h>

#include <string>
#include <thread>

#include "frontend/feature_pipeline.h"
#include "kws/keyword_spotting.h"
#include "utils/log.h"

namespace wekws {
std::shared_ptr<KeywordSpotting> spotter;
std::shared_ptr<wenet::FeaturePipelineConfig> feature_config;
std::shared_ptr<wenet::FeaturePipeline> feature_pipeline;
std::string result;  // NOLINT
int offset;

void init(JNIEnv* env, jobject, jstring jModelDir) {
  const char* pModelDir = env->GetStringUTFChars(jModelDir, nullptr);

  std::string modelPath = std::string(pModelDir) + "/kws.ort";
  spotter = std::make_shared<KeywordSpotting>(modelPath);

  feature_config = std::make_shared<wenet::FeaturePipelineConfig>(40, 16000);
  feature_pipeline = std::make_shared<wenet::FeaturePipeline>(*feature_config);
}

void reset(JNIEnv* env, jobject) {
  offset = 0;
  result = "";
  spotter->Reset();
  feature_pipeline->Reset();
}

void accept_waveform(JNIEnv* env, jobject, jshortArray jWaveform) {
  jsize size = env->GetArrayLength(jWaveform);
  int16_t* waveform = env->GetShortArrayElements(jWaveform, 0);
  std::vector<int16_t> v(waveform, waveform + size);
  feature_pipeline->AcceptWaveform(v);
  env->ReleaseShortArrayElements(jWaveform, waveform, 0);

  LOG(INFO) << "wekws accept waveform in ms: " << int(size / 16);
}

void set_input_finished() {
  LOG(INFO) << "wekws input finished";
  feature_pipeline->set_input_finished();
}

// void spot_thread_func() {
//   while (true) {
//     std::vector<std::vector<float>> feats;
//     feature_pipeline->Read(80, &feats);
//     std::vector<std::vector<float>> prob;
//     spotter->Forward(feats, &prob);
//     float max_prob = 0.0;
//     for (int t = 0; t < prob.size(); t++) {
//       for (int j = 0; j < prob[t].size(); j++) {
//         max_prob = std::max(prob[t][j], max_prob);
//       }
//     }
//     result = std::to_string(offset) + " prob: " + std::to_string(max_prob);
//     offset += prob.size();
//   }
// }

// void start_spot() {
//   std::thread decode_thread(spot_thread_func);
//   decode_thread.detach();
// }

void start_spot() {
    std::vector<std::vector<float>> feats;
    feature_pipeline->Read(80, &feats);
    std::vector<std::vector<float>> prob;
    spotter->Forward(feats, &prob);
    float max_prob = 0.0;
    for (int t = 0; t < prob.size(); t++) {
      for (int j = 0; j < prob[t].size(); j++) {
        max_prob = std::max(prob[t][j], max_prob);
      }
    }
    result = std::to_string(offset) + " prob: " + std::to_string(max_prob);
    offset += prob.size();
}

jstring get_result(JNIEnv* env, jobject) {
  LOG(INFO) << "wekws ui result: " << result;
  return env->NewStringUTF(result.c_str());
}
}  // namespace wekws

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
  JNIEnv* env;
  if (vm->GetEnv(reinterpret_cast<void**>(&env), JNI_VERSION_1_6) != JNI_OK) {
    return JNI_ERR;
  }

  jclass c = env->FindClass("cn/org/wenet/wekws/Spot");
  if (c == nullptr) {
    return JNI_ERR;
  }

  static const JNINativeMethod methods[] = {
      {"init", "(Ljava/lang/String;)V", reinterpret_cast<void*>(wekws::init)},
      {"reset", "()V", reinterpret_cast<void*>(wekws::reset)},
      {"acceptWaveform", "([S)V",
       reinterpret_cast<void*>(wekws::accept_waveform)},
      {"setInputFinished", "()V",
       reinterpret_cast<void*>(wekws::set_input_finished)},
      {"startSpot", "()V", reinterpret_cast<void*>(wekws::start_spot)},
      {"getResult", "()Ljava/lang/String;",
       reinterpret_cast<void*>(wekws::get_result)},
  };
  int rc = env->RegisterNatives(c, methods,
                                sizeof(methods) / sizeof(JNINativeMethod));

  if (rc != JNI_OK) {
    return rc;
  }

  return JNI_VERSION_1_6;
}
