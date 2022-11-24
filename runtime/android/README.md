# Usage

Most of AI engineers are not familiar with Android development, this is a simple ‘how to’.

1. Train your model with your data

2. Export pytorch model to onnx model

3. Convert onnx model for mobile deployment

```bash
python -m onnxruntime.tools.convert_onnx_models_to_ort your-model.onnx
```
you will get `your-model.ort` and `your-model.with_runtime_opt.ort`


4. Install Android Studio and open path of wekws/runtime/android and build

*NOTE:* The default feature_dim in code is 40, if your model’s is 80, change it here `./app/src/main/cpp/wekws.cc`

```C++
  feature_config = std::make_shared<wenet::FeaturePipelineConfig>(40, 16000);  // 40 -> 80
```

It’s also can be built on Linux by runing `bash ./gradlew build`

5. Install `app/build/outputs/apk/debug/app-debug.apk` to your phone and try it.
