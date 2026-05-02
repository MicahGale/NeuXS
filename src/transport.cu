#include "material.cuh"
#include "transport.cuh"

namespace neuxs {

template <typename XSViewType, typename FP>
__global__ void
transport_particles(Particle<FP> **particles, size_t n_particles,
                    CellView<XSViewType, FP> **cells, size_t n_cells) {
  size_t part_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (part_idx >= n_particles)
    return;
  Particle<FP> *part = &particles[part_idx];
  if (part->getCellID() >= n_cells)
    return;
  CellView<XSViewType, FP> *cell = cells[part->getCellID()];
  while (part->isAlive()) {
    bool escaped = cell->particleEscapesTheCell(*part);
    if (escaped) {
      unsigned int next_cell_idx = cell->getRandomNeighborCellIdx(*part);
      part->_cell_id = next_cell_idx;
      cell = cells[next_cell_idx];
    } else {
      CollisionInfo collision = cell->_material->decideCollideType(*part);
      switch (collision._type) {
      case CollisionType::CAPTURE:
        part->_alive = false;
        break;
      case CollisionType::SCATTERING: {
        auto alpha = cell->_material[collision._nuclide_id]._alpha;
        part->_energy *= (1.0f - part->_rng.nextFloat() * (1.0f - alpha));
        break;
      }
      case CollisionType::FISSION:
        *part = Particle<FP>(static_cast<FP>(FISSION_ENERGY), part->getCellID(),
                             part->_rng._state);
        particles[part_idx] = *part;
        break;
      }
    }
  }
}

template <typename FP>
Particle<FP> *get_mono_energetic_particles(unsigned int number_of_particles,
                                           unsigned int cell_id) {
  if (number_of_particles == 0) {
    return nullptr;
  }

  Particle<FP> *particles = new Particle<FP>[number_of_particles];

  for (unsigned int i = 0; i < number_of_particles; ++i) {
    particles[i] = Particle<FP>(static_cast<FP>(FISSION_ENERGY), cell_id);
  }

  return particles;
}

// don't make the compiler crazy
template struct Particle<float>;
template struct Particle<double>;

template neuxs::Particle<float> *
neuxs::get_mono_energetic_particles<float>(unsigned int, unsigned int);

template neuxs::Particle<double> *
neuxs::get_mono_energetic_particles<double>(unsigned int, unsigned int);

} // namespace neuxs