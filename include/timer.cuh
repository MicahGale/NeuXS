#ifndef NEUXS_TIMER_CUH
#define NEUXS_TIMER_CUH

#include <cuda_runtime.h>

/*Simple stop watch to keep timing record of different kernels */

namespace neuxs {
struct StopWatch {

  cudaEvent_t _start;
  cudaEvent_t _stop;

  StopWatch() {
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
  }

  ~StopWatch() {
    cudaEventDestroy(_start);
    cudaEventDestroy(_stop);
  }

  void startClock() { cudaEventRecord(_start, 0); }

  void reset() {
    cudaEventDestroy(_start);
    cudaEventDestroy(_stop);
    cudaEventCreate(&_start);
    cudaEventCreate(&_stop);
  }

  float stopClock() {
    cudaEventRecord(_stop, 0);
    cudaEventSynchronize(_stop);
    float ms = 0.0;
    cudaEventElapsedTime(&ms, _start, _stop);
    return ms;
  }
};
} // namespace neuxs

#endif // NEUXS_TIMER_CUH
