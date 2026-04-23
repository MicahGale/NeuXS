#ifndef NEUXS_MEMORY_CUH
#define NEUXS_MEMORY_CUH

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <utility>

namespace neuxs {

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error: ") +                   \
                               cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

// ================== RAII wrapper for a device allocation ==================
// ===========================================================================
/*
 * DeviceBuffer<T> owns a cudaMalloc'd block. It is move-only so there is
 * never any ambiguity over who frees the memory. This is the primitive used
 * everywhere we would otherwise hand-roll cudaMalloc / cudaFree pairs.
 *
 * The raw device pointer is exposed via `get()` and is what we stuff into
 * the "view" structs that kernels see.
 *
 * Typical flow for uploading host data:
 *
 *   auto buf = DeviceBuffer<T>::makeFromHost(host_ptr, count);
 *   kernel<<<...>>>(buf.get(), count);   // buf frees itself at scope exit
 */
template <typename T> class DeviceBuffer {
public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(size_t count) { allocate(count); }

  ~DeviceBuffer() { reset(); }

  // Move-only: owning device memory must not be accidentally duplicated.
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  DeviceBuffer(DeviceBuffer &&other) noexcept
      : _ptr(other._ptr), _count(other._count) {
    other._ptr = nullptr;
    other._count = 0;
  }

  DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
    if (this != &other) {
      reset();
      _ptr = other._ptr;
      _count = other._count;
      other._ptr = nullptr;
      other._count = 0;
    }
    return *this;
  }

  // Allocate `count` elements. If we already owned something, free it first.
  void allocate(size_t count) {
    reset();
    if (count > 0) {
      CUDA_CHECK(cudaMalloc(&_ptr, count * sizeof(T)));
      _count = count;
    }
  }

  void copyFromHost(const T *host, size_t count) {
    CUDA_CHECK(
        cudaMemcpy(_ptr, host, count * sizeof(T), cudaMemcpyHostToDevice));
  }

  void copyToHost(T *host, size_t count) const {
    CUDA_CHECK(
        cudaMemcpy(host, _ptr, count * sizeof(T), cudaMemcpyDeviceToHost));
  }

  // Upload a single already-built value (e.g. a view struct).
  static DeviceBuffer<T> makeSingle(const T &value) {
    DeviceBuffer<T> buf(1);
    buf.copyFromHost(&value, 1);
    return buf;
  }

  // Upload an array of already-built values.
  static DeviceBuffer<T> makeFromHost(const T *host, size_t count) {
    DeviceBuffer<T> buf(count);
    if (count > 0)
      buf.copyFromHost(host, count);
    return buf;
  }

  T *get() { return _ptr; }
  const T *get() const { return _ptr; }
  size_t size() const { return _count; }
  bool empty() const { return _count == 0; }

  void reset() {
    if (_ptr) {
      // Intentionally don't CUDA_CHECK here: we don't want a throw out of a
      // destructor during stack unwinding. We do clear the error though so a
      // later CUDA call doesn't see a stale sticky error.
      cudaFree(_ptr);
      cudaGetLastError();
      _ptr = nullptr;
      _count = 0;
    }
  }

private:
  T *_ptr = nullptr;
  size_t _count = 0;
};

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
