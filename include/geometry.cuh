#ifndef NEUXS_GEOMETRY_H
#define NEUXS_GEOMETRY_H

#include <cuda_runtime.h>

#include "cross_section.cuh"
#include "material.cuh"
#include "memory.cuh"

namespace neuxs {

// Forward declaration of particle struct.
struct Particle;

/*
 * It's either I am not good enough or cuda is super annoying
 * But right now I am thinking about keeping the cell objects
 * purely in gpu. And only prepare the material class from CPU.
 * I will create an example later.
 */
template <typename XSType, typename FPrecision> struct Cell {

  Cell(FPrecision volume, unsigned int id) : _volume(volume), _id(id) {}

  ~Cell() {
    MemoryManager mm;
    mm.freeDevice(_material);
    mm.freeDevice(_neighbor_cells);
  }

  // setter methods for the cell from host side
  void __host__ setMaterial(Material<XSType, FPrecision> *material) {
    MemoryManager().allocateDevice<Material<XSType, FPrecision>>(_material);

    _material = &material;
  }
  void __host__ setNeighboringCells(Cell<XSType, FPrecision> *cell,
                                    unsigned int n_neighbors);

  // methods that will be used by the transport kernel
  // Making these method forcibly inlined as a way of optimization.
  // As each of these method involves
  bool __device__ __forceinline__ particleEscapesTheCell(Particle *particle);
  Cell *__device__ getRandomNeighborCell(Particle *particle);

  const Material<XSType, FPrecision> *getMaterial() const { return _material; };

  FPrecision _volume;
  unsigned int _id;

  // raw pointers that needs to be on GPU
  Material<XSType, FPrecision> *_material = nullptr;

  // raw pointers that needs to be on GPU
  Cell<XSType, FPrecision> *_neighbor_cells = nullptr;

  unsigned int _num_neighbors;
};

} // namespace neuxs

#endif