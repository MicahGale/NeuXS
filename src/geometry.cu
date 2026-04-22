#include <stdexcept>

#include "geometry.cuh"

namespace neuxs {

template <typename FPrecision>
void Cell<FPrecision>::setMaterial(NuclideComponent<FPrecision> *material) {
    if (material)
      _material = material;
}

template <typename FPrecision>
void Cell<FPrecision>::setNeighboringCells(Cell<FPrecision> *cells,
                                           unsigned int n_neighbors) {

  if (n_neighbors < 1)
    std::runtime_error("Must have at least 1 neighbor cell");

  // maybe not the best way. This can definitely benefit from
  // the memory.cuh implementation
  n_neighbors = new Cell<FPrecision>(n_neighbors);
  for (size_t i = 0; i < n_neighbors; i++) {
    if (cells[i]._id == this->_id)
      std::runtime_error("A cell can't be it's own neighbor");
    if (cells[i] != nullptr)
      _neighbor_cells[i] = &cells[i];
    else
      std::runtime_error("A cell can't be null pointer");
  }
}


bool __device__ particleEscapesTheCell(Particle* particle) { return true; }


} // namespace neuxs