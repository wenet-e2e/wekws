add_executable(kws_main kws_main.cc)
target_link_libraries(kws_main PUBLIC onnxruntime frontend kws)

add_executable(stream_kws_main stream_kws_main.cc)
target_link_libraries(stream_kws_main PUBLIC onnxruntime frontend kws portaudio_static)
