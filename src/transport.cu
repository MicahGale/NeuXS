#include "material.cuh"
#include "transport.cuh"

namespace neuxs {
template <typename FP> struct Collision;
template <typename XS, typename FP>
__device__ void transport_particles(Particle<FP> *particles, size_t n_particles,
                                    CellView<XS, FP> *cells, size_t n_cells) {
  size_t part_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (part_idx >= n_particles)
    return;
  Particle part = particles[part_idx];
  if (part.get_cell_id() >= n_cells)
    return;
  CellView<XS, FP> *cell = cells[part.get_cell_id()];
  while (part.is_alive()) {
    bool escaped = cell->particleEscapesTheCell(*part);
    if (escaped) {
      unsigned int next_neighbor_idx = part->_rng.nextInt(cell->_num_neighbors);
      unsigned int next_cell_idx = cell->getRandomNeighborCellIdx(part);
      part->_cell_id = next_cell_idx;
      cell = cells[next_cell_idx];
      // do the collision process
    } else {
      Collision<FP> collision = cell->_material->decideCollideType(part);
      switch (collision->_type) {
      case CollisionType::CAPTURE:
        part->_alive = false;
        break;
      case CollisionType::SCATTERING:
        part->_energy *=
            (1.0 - part->_rng->nextFloat() * (1 - collision->_nuclide->_alpha));
        break;
      case CollisionType::FISSION:
        // Only simulating one fission neutron to avoid infinite branching
        // Also avoids having to grow the particle bank
        part = Particle(FISSION_ENERGY, part.get_cell_id(), part._rng._state);
        particles[part_idx] = part;
      }
    }
  }
}

} // namespace neuxs
