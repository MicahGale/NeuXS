#include <stdexcept>

#include "geometry.cuh"
#include "material.cuh"
#include "transport.cuh"

namespace neuxs {

// ============================================================================
// CellView
// ============================================================================

template <typename XSViewType, typename FPrecision>
__device__ bool CellView<XSViewType, FPrecision>::particleEscapesTheCell(
    Particle<FPrecision> *particle) const {
  const FPrecision sigma_t = _material->getMacroscopicSigmaT(particle->_energy);

  // Pathological case: no material interaction at this energy → escape.
  if (sigma_t <= static_cast<FPrecision>(0))
    return true;

  const FPrecision xi = static_cast<FPrecision>(particle->_rng.nextDouble());
  const FPrecision d_collision = -device_log(xi) / sigma_t;

  const FPrecision char_length = device_cbrt(_volume);

  return d_collision > char_length;
}

template <typename XSViewType, typename FPrecision>
__device__ CellView<XSViewType, FPrecision> *
CellView<XSViewType, FPrecision>::getRandomNeighborCell(
    Particle<FPrecision> *particle, CellView *all_cells) const {
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

// ============================================================================
// Cell
// ============================================================================

template <typename XSType, typename FPrecision>
Cell<XSType, FPrecision>::Cell(FPrecision volume, unsigned int id)
    : _volume(volume), _id(id) {}

template <typename XSType, typename FPrecision>
Cell<XSType, FPrecision>::~Cell() {
  if (_neighbor_cells != nullptr) {
    delete[] _neighbor_cells;
  }
}

template <typename XSType, typename FPrecision>
void __host__
Cell<XSType, FPrecision>::setMaterial(Material<XSType, FPrecision> *material) {
  if (material) {
    _material = material;
  } else {
    throw std::runtime_error("Null Material");
  }
}

template <typename XSType, typename FPrecision>
void __host__ Cell<XSType, FPrecision>::setNeighboringCells(
    Cell<XSType, FPrecision> *cells, unsigned int n_neighbors) {
  if (n_neighbors < 1)
    throw std::runtime_error("Must have at least 1 neighbor cell");

  _num_neighbors = n_neighbors;
  _neighbor_cells = new size_t[_num_neighbors];

  for (size_t i = 0; i < _num_neighbors; i++)
    _neighbor_cells[i] = cells[i]._id;
}

template <typename XSType, typename FPrecision>
__host__ typename Cell<XSType, FPrecision>::ViewType *
Cell<XSType, FPrecision>::uploadToDevice() {
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

template <typename XSType, typename FPrecision>
const Material<XSType, FPrecision> *
Cell<XSType, FPrecision>::getMaterial() const {
  return _material;
}

// ============================================================================
// Explicit template instantiations
// ============================================================================

// CellView — combinations of {AoSLinearView, SoALinearView,
// LogarithmicHashAoSView} × {float, double}
template struct CellView<AoSLinearView<float>, float>;
template struct CellView<AoSLinearView<double>, double>;
template struct CellView<SoALinearView<float>, float>;
template struct CellView<SoALinearView<double>, double>;
template struct CellView<LogarithmicHashAoSView<float>, float>;
template struct CellView<LogarithmicHashAoSView<double>, double>;

// Cell — combinations of {AoSLinear, SoALinear, LogarithmicHashAoS} ×
// {float, double}
template struct Cell<AoSLinear<float>, float>;
template struct Cell<AoSLinear<double>, double>;
template struct Cell<SoALinear<float>, float>;
template struct Cell<SoALinear<double>, double>;
template struct Cell<LogarithmicHashAoS<float>, float>;
template struct Cell<LogarithmicHashAoS<double>, double>;

} // namespace neuxs
