#ifndef NEUXS_TRANSPORT_CUH
#define NEUXS_TRANSPORT_CUH

#include <cstdint>
#include <cuda_runtime.h>

template <typename XSType, typename FPrecision> struct CellView;

const double FISSION_ENERGY = 2.2e6; // [eV] just an approximation.
namespace neuxs {

enum class EventType { COLLIDE, ESCAPE, DIE };

// Simple 64-bit LCG (Knuth / MMIX constants). Adequate for a transport
// skeleton; a production code would use cuRAND or a counter-based generator
// (Philox / Threefry). We keep this header-only so we don't drag -lcurand
// into the link line just to write a couple of demo kernels.
struct SimpleRNG {
  uint64_t _state;

  __host__ __device__ explicit SimpleRNG(uint64_t seed = 1ULL) : _state(seed) {}

  // Uniform in (0, 1]. Top 53 bits → a double-precision mantissa.
  __device__ __forceinline__ double nextDouble() {
    _state = _state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint64_t hi = (_state >> 11);
    if (hi == 0)
      hi = 1; // exclude exact zero so log() is well-defined downstream
    return static_cast<double>(hi) * (1.0 / static_cast<double>(1ULL << 53));
  }

  // avoid going to the double modules
  __device__ __forceinline__ float nextFloat() {
    _state = _state * 6364136223846793005ULL + 1442695040888963407ULL;
    uint64_t hi = (_state >> 11);
    if (hi == 0)
      hi = 1; // exclude exact zero so log() is well-defined downstream
    return static_cast<float>(hi) * (1.0 / static_cast<float>(1ULL << 53));
  }
  __device__ __forceinline__ unsigned int nextInt(unsigned int max) {
    return static_cast<unsigned int>(this->nextFloat() *
                                     static_cast<float>(max));
  }
};

// Particle definition. Mostly  placeholder
template <typename FP> struct Particle {

  FP _energy = static_cast<FP>(0);
  unsigned int _cell_id = 0;
  bool _alive = true;
  SimpleRNG _rng{1ULL};
  Particle(FP energy, unsigned int cell_id, unsigned int seed)
      : _energy(energy), _cell_id(cell_id), _alive(true) {
    _rng = SimpleRNG(seed);
  };

  bool isAlive() { return this->_alive; }
  unsigned int getCellID() { return this->_cell_id; }
};

template <typename XS, typename FP>
__device__ void transport_particles(Particle<FP> *particles, size_t n_particles,
                                    CellView<XS, FP> *cells, size_t n_cells);

} // namespace neuxs

#endif // NEUXS_TRANSPORT_CUH
