#ifndef NEUXS_TRANSPORT_CUH
#define NEUXS_TRANSPORT_CUH

#include <cstdint>
#include <cuda_runtime.h>

#include "geometry.cuh"
#include "material.cuh"

namespace neuxs {

template <typename XSType, typename FPrecision> struct CellView;

const double FISSION_ENERGY = 1e6; // [eV] just an approximation.

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

  // avoid going to the double modulesunsigned int
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
  __host__ __device__ Particle() = default;
  __host__ __device__ Particle(FP energy, unsigned int id)
      : _energy(energy), _cell_id(id) {};

  __host__ __device__ Particle(FP energy, unsigned int cell_id,
                               unsigned int seed)
      : _energy(energy), _cell_id(cell_id), _alive(true),
        _rng(SimpleRNG(seed)) {}

  FP _energy = FISSION_ENERGY;
  unsigned int _cell_id = 0;
  bool _alive = true;
  SimpleRNG _rng{1ULL};

  __host__ __device__ bool isAlive() { return this->_alive; }
  __host__ __device__ unsigned int getCellID() { return this->_cell_id; }
};

template <typename FP>
__host__ Particle<FP> *
get_mono_energetic_particles(unsigned int number_of_particles,
                             unsigned int cell_id, unsigned int stride = 10000);

template <typename XSViewType, typename FP>
__global__ void transport_particles(Particle<FP> *particles, size_t n_particles,
                                    CellView<XSViewType, FP> **cells,
                                    size_t n_cells) {
  size_t part_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (part_idx >= n_particles)
    return;

  auto &part = particles[part_idx];
  if (part._cell_id >= n_cells)
    return;

  auto *cell = cells[part._cell_id];

  while (part.isAlive()) {
    if (cell->particleEscapesTheCell(&part)) {
      unsigned int next_idx = cell->getRandomNeighborCellIdx(&part);
      part._cell_id = next_idx;
      cell = cells[next_idx];
    } else {
      CollisionInfo collision = cell->_material->decideCollideType(part);
      switch (collision._type) {
      case CollisionType::CAPTURE: {
        part._alive = false;
        break;
      }
      case CollisionType::SCATTERING: {
        FP alpha = cell->_material->_nuclides[collision._nuclide_id]._alpha;
        part._energy *=
            (1 - static_cast<FP>(part._rng.nextFloat() * (1 - alpha)));
        break;
      }
      case CollisionType::FISSION:
        uint64_t new_seed = part._rng._state;
        part = Particle<FP>(static_cast<FP>(FISSION_ENERGY), part._cell_id,
                            new_seed);
        break;
      }
    }
  }
}

} // namespace neuxs

#endif // NEUXS_TRANSPORT_CUH
