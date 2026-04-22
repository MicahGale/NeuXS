#ifndef NEUXS_MEMORY_CUH
#define NEUXS_MEMORY_CUH

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace neuxs {

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

class MemoryManager {
public:
  MemoryManager(){};

  template <typename T> T *allocateLockedHost(size_t count) {
    T *ptr;
    CUDA_CHECK(cudaMallocHost(&ptr, count * sizeof(T)));
    return ptr;
  }

  template <typename T> T *allocateDevice(size_t count) {
    T *ptr;
    CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
  }

  template <typename T>
  void copyToDevice(const T *host_ptr, T *device_ptr, size_t count) {
    CUDA_CHECK(cudaMemcpy(device_ptr, host_ptr, count * sizeof(T),
                          cudaMemcpyHostToDevice));
  }

  template <typename T>
  void copyToHost(const T *device_ptr, T *host_ptr, size_t count) {
    CUDA_CHECK(cudaMemcpy(host_ptr, device_ptr, count * sizeof(T),
                          cudaMemcpyDeviceToHost));
  }

  template <typename T> void freeDevice(T *ptr) {
    if (ptr)
      CUDA_CHECK(cudaFree(ptr));
  }

  template <typename T> void freeHost(T *ptr) {
    if (ptr)
      CUDA_CHECK(cudaFreeHost(ptr));
  }

  //
  __host__ void deviceSynchronize() { CUDA_CHECK(cudaDeviceSynchronize()); }

  void printDeviceInfo() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    printf("GPU Device: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total Memory: %.1f GB\n", prop.totalGlobalMem / 1e9f);
    printf("  Shared Memory per Block: %zu KB\n",
           prop.sharedMemPerBlock / 1024);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max Block Dim: [%d, %d, %d]\n", prop.maxThreadsDim[0],
           prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  Max Grid Dim: [%d, %d, %d]\n", prop.maxGridSize[0],
           prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("  Warp Size: %d\n", prop.warpSize);
  }
};
} // namespace neuxs

#endif // NEUXS_MEMORY_CUH