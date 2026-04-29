#include "transport.cuh"

namespace neuxs {

template <typename XS, typename FP>
__device__ void transport_particles(Particle<FP> *particles, size_t n_particles,
                                    Cell<XS, FP> *cells, size_t n_cells) {
  size_t part_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (part_idx >= n_particles)
    return;
  Particle part = particles[part_idx];
  while (part.is_alive()) {
    // TODO
  }
}

}
