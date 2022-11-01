# WeKWS & Raspberry PI

There are two ways to build the runtime binaries for Raspberry PI.

1. Refer `runtime/onnxruntime/README.md` to build it in Raspberry PI.
2. Cross compile and `scp` the binaries and libraries to Raspberry PI.

## Cross Compile

* Step 1. Install cross compile tools in the PC.

``` sh
sudo apt-get install gcc-aarch64-linux-gnu g++-aarch64-linux-gnu
```

Or download, and install the binaries from: https://releases.linaro.org/components/toolchain/binaries/latest-7


* Step 2. Export your experiment model to ONNX by https://github.com/wenet-e2e/wekws/blob/main/wekws/bin/export_onnx.py

``` sh
exp=exp  # Change it to your experiment dir
python -m wekws.bin.export_onnx \
  --config $exp/train.yaml \
  --checkpoint $exp/final.pt \
  --output_dir final.onnx
```

* Step 3. Build. The build requires cmake 3.14 or above. and Send the binary and libraries to Raspberry PI.

``` sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=toolchains/aarch64-linux-gnu.toolchain.cmake
cmake --build build
scp -r build/bin pi@xxx.xxx.xxx:/path/to/wekws
scp fc_base/onnxruntime-src/lib/libonnxruntime.so* pi@xxx.xxx.xxx:/path/to/wekws
```

* Step 4. Run. The log will be shown in Raspberry PI's console.

``` sh
cd /path/to/wekws
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
./build/bin/stream_kws_main 40 80 final.onnx
```
