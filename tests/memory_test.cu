#include "memory.cuh"
#include <iostream>

int main() {
  constexpr int N = 1024;

  auto h = neuxs::allocateLockedHost<float>(N);
  auto d = neuxs::allocateDevice<float>(N);

  for (int i = 0; i < N; i++) {
    h[i] = static_cast<float>(i);
  }

  neuxs::copyToDevice(h, d, N);
  neuxs::copyToHost(d, h, N);
  neuxs::deviceSynchronize();

  if (h[123] != 123.0f) {
    return 1;
  }

  std::cout << "Test PASSED\n";
  // be a good citizen
  neuxs::freeDevice(d);
  neuxs::freeHost(h);

  return 0;
}