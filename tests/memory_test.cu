#include "memory.cuh"
#include <iostream>

int main() {
  neuxs::MemoryManager memory_manager;
  constexpr int N = 1024;

  auto h = memory_manager.allocateLockedHost<float>(N);
  auto d = memory_manager.allocateDevice<float>(N);

  for (int i = 0; i < N; i++) {
    h[i] = static_cast<float>(i);
  }

  memory_manager.copyToDevice<float>(h, d, N);
  memory_manager.copyToHost<float>(d, h, N);
  memory_manager.deviceSynchronize();

  if (h[123] != 123.0f) {
    return 1;
  }

  std::cout << "Test PASSED\n";
  // be a good citizen
  memory_manager.freeDevice<float>(d);
  memory_manager.freeHost<float>(h);

  return 0;
}