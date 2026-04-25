#ifndef NEUXS_GEOMETRY_H
#define NEUXS_GEOMETRY_H

#include <cuda_runtime.h>
#include <stdexcept>

#include "cross_section.cuh"
#include "material.cuh"
#include "memory.cuh"
#include "transport.cuh"

namespace neuxs {

// Forward declaration kept for backwards compat with any TU that pulls in
// geometry.cuh without transport.cuh — the full definition lives in
// transport.cuh, included above.
template <typename FPrecision> struct Particle;

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
 * POD. The caller passes that global array to getRandomNeighborCell.
 */
template <typename XSViewType, typename FPrecision> struct CellView {
  FPrecision _volume;
  unsigned int _id;
  MaterialView<XSViewType, FPrecision> *_material; // device ptr
  size_t *_neighbor_cell_ids; // device array, length = _num_neighbors
  unsigned int _num_neighbors;

  /*
   * Stochastic escape decision based on volume and macroscopic cross
   * section at the particle's energy.
   *
   * Physics:
   *   Σt(E) = Σᵢ Nᵢ · σt,ᵢ(E)               macroscopic total XS
   *   d     = −ln(ξ) / Σt(E)                 sampled distance to collision
   *   L     = ∛V                             characteristic cell length
   *
   * If d > L, the particle's next flight carries it past the cell boundary
   * before interacting → escape. Otherwise it collides inside the cell.
   *
   * This is the distance-to-collision test from standard stochastic
   * tracking, adapted to a volume-only cell description (no explicit
   * surfaces yet). The same Σt(E) computed here is what the collision
   * handler downstream will reuse.
   */
  __device__ __forceinline__ bool
  particleEscapesTheCell(Particle<FPrecision> *particle) const {
    const FPrecision sigma_t =
        _material->getMacroscopicSigmaT(particle->_energy);

    // Pathological case: no material interaction at this energy → escape.
    if (sigma_t <= static_cast<FPrecision>(0)) {
      return true;
    }

    // Sample a uniform ξ ∈ (0, 1] and convert to a distance to collision.
    const FPrecision xi = static_cast<FPrecision>(particle->_rng.nextDouble());
    const FPrecision d_collision = -device_log(xi) / sigma_t;

    const FPrecision char_length = device_cbrt(_volume);

    return d_collision > char_length;
  }

  /*
   * Uniformly pick one of this cell's neighbors for the particle's next
   * cell. Returns a pointer into the global CellView array so the caller
   * can keep chasing pointers.
   *
   * `all_cells` is the flat, device-resident array of every CellView in
   * the problem; the neighbor IDs stored on this view are indices into
   * that array. We need it because we deliberately don't store raw
   * neighbor pointers (that would require fixing up a pointer cycle
   * during upload, and would duplicate the neighbor info).
   */
  __device__ __forceinline__ CellView *
  getRandomNeighborCell(Particle<FPrecision> *particle,
                        CellView *all_cells) const {
    if (_num_neighbors == 0 || all_cells == nullptr)
      return nullptr;

    const FPrecision xi = static_cast<FPrecision>(particle->_rng.nextDouble());
    unsigned int i =
        static_cast<unsigned int>(xi * static_cast<FPrecision>(_num_neighbors));
    if (i >= _num_neighbors)
      i = _num_neighbors - 1;

    const size_t nid = _neighbor_cell_ids[i];
    return &all_cells[nid];
  }
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

  // NOTE: the physics methods (particleEscapesTheCell, getRandomNeighborCell)
  // were previously declared here but can't actually run on a Cell — Cell
  // lives in host memory and kernels have no valid pointer to one. They now
  // live on CellView, where kernels can reach them. See CellView above.

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
