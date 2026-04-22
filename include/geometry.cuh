#ifndef NEUXS_GEOMETRY_H
#define NEUXS_GEOMETRY_H

#include <cuda_runtime.h>

#include "cross_section.cuh"
#include "material.cuh"

namespace neuxs {

struct Particle;

template <typename FPrecision> struct Cell {

  Cell(FPrecision volume, unsigned int id) : _volume(volume), _id(id) {}
  FPrecision _volume;
  unsigned int _id;

  // Raw pointers as we don't to copy same materials everywhere.
  NuclideComponent<FPrecision> *_material = nullptr;

  // Raw pointers as we don't to copy same materials everywhere.
  Cell<FPrecision> *_neighbor_cells = nullptr;

  unsigned int _num_neighbors;

  void __host__ setMaterial(NuclideComponent<FPrecision> *material);
  void __host__ setNeighboringCells(Cell<FPrecision> *cell,
                                    unsigned int n_neighbors);

  bool __device__ particleEscapesTheCell(Particle* particle);
  Cell *__device__ getRandomNeighborCell(Particle* particle);

  const NuclideComponent<FPrecision> *getMaterial() const;
};

} // namespace neuxs

#endif