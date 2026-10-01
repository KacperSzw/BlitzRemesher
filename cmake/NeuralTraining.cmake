if(BLITZ_NEURAL_TRAIN)
  set(BLITZ_LIBTORCH_ROOT "$ENV{BLITZ_LIBTORCH_ROOT}" CACHE PATH "Official LibTorch shared-with-deps directory")
  if(NOT EXISTS "${BLITZ_LIBTORCH_ROOT}/include/torch/csrc/api/include/torch/torch.h")
    message(FATAL_ERROR "Set BLITZ_LIBTORCH_ROOT to the pinned LibTorch C++ distribution")
  endif()

  # Training dependencies never enter the installed runtime target. Shared CUDA
  # updates compile once; command-line programs only compose the training code.
  add_library(blitz_neural_training STATIC training/update_cuda.cu training/fused.cu)
  target_include_directories(blitz_neural_training PUBLIC . src)
  target_include_directories(blitz_neural_training SYSTEM PUBLIC
    "${BLITZ_LIBTORCH_ROOT}/include"
    "${BLITZ_LIBTORCH_ROOT}/include/torch/csrc/api/include")
  target_compile_definitions(blitz_neural_training PUBLIC _GLIBCXX_USE_CXX11_ABI=1)
  target_link_directories(blitz_neural_training PUBLIC "${BLITZ_LIBTORCH_ROOT}/lib")
  target_link_libraries(blitz_neural_training PUBLIC blitz_io CUDA::cudart CUDA::cublas CUDA::cublasLt
    torch torch_cpu
    "$<$<PLATFORM_ID:Linux>:-Wl,--push-state,--no-as-needed>" torch_cuda
    "$<$<PLATFORM_ID:Linux>:-Wl,--pop-state>" c10 c10_cuda)
  set_target_properties(blitz_neural_training PROPERTIES CUDA_STANDARD 20 CUDA_STANDARD_REQUIRED ON)
  target_compile_options(blitz_neural_training PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false;--prec-div=true;--prec-sqrt=true>)

  add_executable(blitz-neural-train tools/neural/train.cpp)
  add_executable(blitz-neural-replay tools/neural/replay.cpp)
  add_executable(blitz-neural-action-train tools/neural/action_train.cpp)
  add_executable(blitz-neural-cycle tools/neural/cycle.cpp)
  add_executable(blitz-neural-diagnostics tools/neural/diagnostics.cpp)
  foreach(neural_tool blitz-neural-train blitz-neural-replay blitz-neural-action-train blitz-neural-cycle blitz-neural-diagnostics)
    target_link_libraries(${neural_tool} PRIVATE blitz_neural_training)
    set_target_properties(${neural_tool} PROPERTIES BUILD_RPATH "${BLITZ_LIBTORCH_ROOT}/lib")
  endforeach()
endif()
