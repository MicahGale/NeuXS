#ifndef NEUXS_GEOMETRY_H
#define NEUXS_GEOMETRY_H

#include <cuda_runtime.h>
#include <stdexcept>

#include "cross_section.cuh"
#include "material.cuh"
#include "memory.cuh"
#include "transport.cuh"

namespace neuxs {

// Forward declaration
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
   * Stochastic escape decision based on volume and macroscopic cross-section
   * at the particle's energy.
   *
   * Physics:
   *   d     = −ln(random_number) / Xs_t(E)           sampled distance to
   * collision L     = ∛V                             characteristic cell length
   *
   * If d > L, the particle's next flight carries it past the cell boundary
   * before interacting → escape. Otherwise, it collides inside the cell.
   *
   * This is the distance-to-collision test from standard stochastic
   * tracking, adapted to a volume-only cell description (no explicit
   * surfaces yet). The same Σt(E) computed here is what the collision
   * handler downstream will reuse.
   */
  __device__ bool particleEscapesTheCell(Particle<FPrecision> *particle) const;

  /*
   * Uniformly pick one of this cell's neighbors for the particle's next
   * cell. Returns a pointer into the global CellView array so the caller
   * can keep chasing pointers.
   */
  __device__ CellView *getRandomNeighborCell(Particle<FPrecision> *particle,
                                             CellView *all_cells) const;
};

template <typename XSType, typename FPrecision> struct Cell {

  using XSViewType = typename XSType::ViewType;
  using ViewType = CellView<XSViewType, FPrecision>;

  Cell(FPrecision volume, unsigned int id);

  ~Cell();

  // Forcing it to be non-copyable.
  Cell(const Cell &) = delete;
  Cell &operator=(const Cell &) = delete;

  void __host__ setMaterial(Material<XSType, FPrecision> *material);

  void __host__ setNeighboringCells(Cell<XSType, FPrecision> *cells,
                                    unsigned int n_neighbors);

  /*
   * Deep-copy this cell (and its material subtree, if not already uploaded)
   * onto the device. Returns a device pointer to the kernel-facing CellView.
   * Idempotent.
   */
  __host__ ViewType *uploadToDevice();

  const Material<XSType, FPrecision> *getMaterial() const;

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
