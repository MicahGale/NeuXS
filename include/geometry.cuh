#ifndef NEUXS_GEOMETRY_H
#define NEUXS_GEOMETRY_H

#include "material.cuh"
#include <cuda_runtime.h>

namespace neuxs {

template <typename XSType, typename FPrecision> struct Cell {
  FPrecision _volume;
  unsigned int _id;
  Material<XSType, FPrecision> *_material;
  Cell<XSType, FPrecision> *_neighbor_cells;
  unsigned int _num_neighbors;

  Cell(float volume, unsigned int id);

  void __host__ setMaterial(Material<XSType, FPrecision> *material);

  void __host__ setNeighboringCells(Cell<XSType, FPrecision> *cells,
                                    unsigned int number_of_neighbors);

  void __host__ checkNeighboringCellIDs();

  bool __device__ particleEscapesTheCell(float particle_energy);

  Cell *__device__ getRandomNeighborCell(float particle_energy);

  const Material<XSType, FPrecision> *getMaterial() const;
};

} // namespace neuxs

#endif
