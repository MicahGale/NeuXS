#ifndef NEUXS_GEOMETRY_H
#define NEUXS_GEOMETRY_H

#include <cuda_runtime.h>
#include <stdexcept>

#include "cross_section.cuh"
#include "material.cuh"
#include "memory.cuh"

namespace neuxs {

struct Particle;

// ======================= CellView (device-facing POD) ======================
// ===========================================================================
/*
 * What kernels see. `_material` is a device pointer to an already-uploaded
 * MaterialView, so from inside a kernel you get the pointer chase you
 * wanted:
 *
 *   cell_view->_material->_xs_views[i].getCrossSection(e)
 *   cell_view->_material->_nuclides[i]._atom_dens
 *
 * Neighbor cells are referred to by ID (indexing into a global device array
 * of CellViews) — this avoids circular pointer uploads and keeps the view
 * POD.
 */
template <typename XSViewType, typename FPrecision> struct CellView {
  FPrecision _volume;
  unsigned int _id;
  MaterialView<XSViewType, FPrecision> *_material; // device ptr
  size_t *_neighbor_cell_ids; // device array, length = _num_neighbors
  unsigned int _num_neighbors;
};

template <typename XSType, typename FPrecision> struct Cell {

  // Derive view types directly from the XS class. Adding a new XS scheme
  // requires *zero* changes here.
  using XSViewType = typename XSType::ViewType;
  using ViewType = CellView<XSViewType, FPrecision>;

  Cell(FPrecision volume, unsigned int id) : _volume(volume), _id(id) {}

  ~Cell() {
    if (_neighbor_cells != nullptr) {
      delete[] _neighbor_cells;
    }
  }

  // Non-copyable (owns raw arrays and device buffers).
  Cell(const Cell &) = delete;
  Cell &operator=(const Cell &) = delete;

  void __host__ setMaterial(Material<XSType, FPrecision> *material) {
    if (material) {
      _material = material;
    } else {
      throw std::runtime_error("Null Material");
    }
  }

  void __host__ setNeighboringCells(Cell<XSType, FPrecision> *cells,
                                    unsigned int n_neighbors) {
    if (n_neighbors < 1)
      throw std::runtime_error("Must have at least 1 neighbor cell");

    _num_neighbors = n_neighbors;
    _neighbor_cells = new size_t[_num_neighbors];

    for (size_t i = 0; i < _num_neighbors; i++)
      _neighbor_cells[i] = cells[i]._id;
  }

  /*
   * Deep-copy this cell (and its material subtree, if not already uploaded)
   * onto the device. Returns a device pointer to the kernel-facing CellView.
   * Idempotent.
   *
   * Note: the neighbor IDs are uploaded as-is; dereferencing them against a
   * global array of CellViews is the caller's responsibility (that's how
   * you break the potential neighbor-of-neighbor pointer cycle cleanly).
   */
  __host__ ViewType *uploadToDevice() {
    if (_uploaded)
      return _d_self.get();

    if (!_material)
      throw std::runtime_error("Cell has no material");

    auto *material_view_device = _material->uploadToDevice();

    _d_neighbor_ids =
        DeviceBuffer<size_t>::makeFromHost(_neighbor_cells, _num_neighbors);

    ViewType v;
    v._volume = _volume;
    v._id = _id;
    v._material = material_view_device;
    v._neighbor_cell_ids = _d_neighbor_ids.get();
    v._num_neighbors = _num_neighbors;

    _d_self = DeviceBuffer<ViewType>::makeSingle(v);
    _uploaded = true;
    return _d_self.get();
  }

  bool __device__ __forceinline__ particleEscapesTheCell(Particle *particle);
  Cell *__device__ getRandomNeighborCell(Particle *particle);

  const Material<XSType, FPrecision> *getMaterial() const { return _material; }

  FPrecision _volume;
  unsigned int _id;
  Material<XSType, FPrecision> *_material = nullptr;
  size_t *_neighbor_cells = nullptr;
  unsigned int _num_neighbors = 0;

  // ---------- device-side backing storage (owned via RAII) ----------
  DeviceBuffer<size_t> _d_neighbor_ids;
  DeviceBuffer<ViewType> _d_self;

private:
  bool _uploaded = false;
};

} // namespace neuxs

#endif
