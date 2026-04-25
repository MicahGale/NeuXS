#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "cross_section.cuh"
#include "cross_section_reader.h"

namespace neuxs {

template <typename FPrecision>
void AoSLinear<FPrecision>::setCrossSection(
    const OpenMCCrossSectionReader &reader,
    NuclideComponent<FPrecision> &nuclide) {

  auto nuclide_name = std::string(nuclide._name);
  auto energy_host = reader.getEnergyDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature);
  auto scattering = reader.getCrossSectionDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::SCATTERING);
  auto capture = reader.getCrossSectionDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::CAPTURE);

  const auto size = energy_host.size();
  std::vector<FPrecision> fission;
  if (nuclide._allows_fission) {
    fission = reader.getCrossSectionDataPoints<FPrecision>(
        nuclide_name, nuclide._temperature, CrossSectionDataType::FISSION);
    if (fission.empty())
      fission.reserve(size);

    if (fission.empty()) {
      fission.assign(size, static_cast<FPrecision>(0));
    }
  } else {
    fission.assign(size, static_cast<FPrecision>(0));
  }

  // let do some manual memory allocation
  this->_energy = new FPrecision[size];
  this->_xs_data = new CrossSectionGridPoint<FPrecision>[size];
  this->_size = size;

  for (size_t i = 0; i < size; i++) {
    this->_energy[i] = energy_host[i];
    CrossSectionGridPoint<FPrecision> grid(scattering[i], fission[i],
                                           capture[i]);
    this->_xs_data[i] = grid;
  }
}

template <typename FPrecision>
typename AoSLinear<FPrecision>::ViewType
AoSLinear<FPrecision>::uploadToDevice() {
  // Idempotent: if we've already uploaded, hand back the cached view.
  if (this->_uploaded)
    return this->_cached_view;

  const size_t n = this->_size;

  // Allocate + copy both host arrays to device in one shot each.
  _d_energy = DeviceBuffer<FPrecision>::makeFromHost(this->_energy, n);
  _d_xs_data = DeviceBuffer<CrossSectionGridPoint<FPrecision>>::makeFromHost(
      this->_xs_data, n);

  this->_cached_view._energy = _d_energy.get();
  this->_cached_view._grid = _d_xs_data.get();
  this->_cached_view._size = n;
  this->_uploaded = true;
  return this->_cached_view;
}

template <typename FPrecision>
void SoALinear<FPrecision>::setCrossSection(
    const OpenMCCrossSectionReader &reader,
    NuclideComponent<FPrecision> &nuclide) {
  auto nuclide_name = std::string(nuclide._name);

  auto energy_host = reader.getEnergyDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature);
  auto scattering_host = reader.getCrossSectionDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::SCATTERING);
  auto capture_host = reader.getCrossSectionDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::CAPTURE);
  std::vector<FPrecision> fission_host;
  const auto size = energy_host.size();
  if (nuclide._allows_fission) {
    fission_host = reader.getCrossSectionDataPoints<FPrecision>(
        nuclide_name, nuclide._temperature, CrossSectionDataType::FISSION);

    if (fission_host.empty())
      fission_host.assign(size, static_cast<FPrecision>(0));

  } else
    fission_host.assign(size, static_cast<FPrecision>(0));

  // let do some manual memory allocation
  this->_energy = new FPrecision[size];
  this->_xs_data._sigma_s = new FPrecision[size];
  this->_xs_data._sigma_f = new FPrecision[size];
  this->_xs_data._sigma_c = new FPrecision[size];
  this->_xs_data._sigma_t = new FPrecision[size];
  this->_size = size;

  for (size_t i = 0; i < size; i++) {
    this->_energy[i] = energy_host[i];
    this->_xs_data._sigma_s[i] = scattering_host[i];
    this->_xs_data._sigma_f[i] = fission_host[i];
    this->_xs_data._sigma_c[i] = capture_host[i];
    this->_xs_data._sigma_t[i] =
        scattering_host[i] + fission_host[i] + capture_host[i];
  }
}

template <typename FPrecision>
typename SoALinear<FPrecision>::ViewType
SoALinear<FPrecision>::uploadToDevice() {
  if (_uploaded)
    return _cached_view;

  const size_t n = this->_size;

  _d_energy = DeviceBuffer<FPrecision>::makeFromHost(this->_energy, n);
  _d_sigma_s =
      DeviceBuffer<FPrecision>::makeFromHost(this->_xs_data._sigma_s, n);
  _d_sigma_f =
      DeviceBuffer<FPrecision>::makeFromHost(this->_xs_data._sigma_f, n);
  _d_sigma_c =
      DeviceBuffer<FPrecision>::makeFromHost(this->_xs_data._sigma_c, n);
  _d_sigma_t =
      DeviceBuffer<FPrecision>::makeFromHost(this->_xs_data._sigma_t, n);

  _cached_view._energy = _d_energy.get();
  _cached_view._data._sigma_s = _d_sigma_s.get();
  _cached_view._data._sigma_f = _d_sigma_f.get();
  _cached_view._data._sigma_c = _d_sigma_c.get();
  _cached_view._data._sigma_t = _d_sigma_t.get();
  _cached_view._size = n;
  _uploaded = true;
  return _cached_view;
}

// ======================= LogarithmicHashAoS =============================
/*
 * Hash-table construction: for each of (n_bins + 1) evenly spaced bin
 * boundaries in ln(E) space, record the largest grid index k such that
 * _energy[k] <= bin_boundary. Lookup at runtime then uses
 *     [hash_table[bin], hash_table[bin + 1]]
 * as the already-narrowed window for binary search.
 *
 * We walk a cursor through _energy, so construction is O(size + n_bins)
 * rather than O(size * n_bins).
 */
template <typename FPrecision>
void LogarithmicHashAoS<FPrecision>::setLogarithmicHashGrid(size_t n_bins) {
  if (!this->_energy || this->_size < 2) {
    throw std::runtime_error(
        "setLogarithmicHashGrid: call setCrossSection first");
  }

  _n_bins = n_bins;

  FPrecision E_min = this->_energy[0];
  FPrecision E_max = this->_energy[this->_size - 1];
  if (E_min <= static_cast<FPrecision>(0) || E_max <= E_min) {
    throw std::runtime_error(
        "setLogarithmicHashGrid: degenerate energy grid for log hash");
  }

  FPrecision log_min = std::log(E_min);
  FPrecision log_max = std::log(E_max);

  _hash_info._grid_energy_minimum = log_min;
  _hash_info._grid_energy_delta =
      static_cast<FPrecision>(_n_bins) / (log_max - log_min);

  delete[] _hash_table_host;
  _hash_table_host = new size_t[_n_bins + 1];

  // Single cursor sweep: k is the largest index with _energy[k] <= bin_e.
  _hash_table_host[0] = 0;
  size_t k = 0;
  for (size_t i = 1; i < _n_bins; i++) {
    FPrecision bin_log_e =
        log_min + static_cast<FPrecision>(i) / _hash_info._grid_energy_delta;
    FPrecision bin_e = std::exp(bin_log_e);

    while (k + 1 < this->_size && this->_energy[k + 1] <= bin_e)
      k++;

    _hash_table_host[i] = k;
  }
  _hash_table_host[_n_bins] = this->_size - 1;
}

template <typename FPrecision>
typename LogarithmicHashAoS<FPrecision>::ViewType
LogarithmicHashAoS<FPrecision>::uploadToDevice() {
  if (_hash_uploaded)
    return _cached_hash_view;

  if (!_hash_table_host) {
    throw std::runtime_error(
        "uploadToDevice: call setLogarithmicHashGrid first");
  }

  const size_t n = this->_size;

  // Reuse the base class's device buffers for energy + grid.
  this->_d_energy = DeviceBuffer<FPrecision>::makeFromHost(this->_energy, n);
  this->_d_xs_data =
      DeviceBuffer<CrossSectionGridPoint<FPrecision>>::makeFromHost(
          this->_xs_data, n);

  // Upload the hash table.
  _d_hash_table =
      DeviceBuffer<size_t>::makeFromHost(_hash_table_host, _n_bins + 1);

  // Build the composite view.
  _cached_hash_view._base._energy = this->_d_energy.get();
  _cached_hash_view._base._grid = this->_d_xs_data.get();
  _cached_hash_view._base._size = n;
  _cached_hash_view._log_energy_min = _hash_info._grid_energy_minimum;
  _cached_hash_view._hash_delta = _hash_info._grid_energy_delta;
  _cached_hash_view._hash_table = _d_hash_table.get();
  _cached_hash_view._n_bins = _n_bins;

  // Mark both caches populated so either path returns a valid view.
  this->_cached_view = _cached_hash_view._base;
  this->_uploaded = true;
  _hash_uploaded = true;

  return _cached_hash_view;
}

// need explicit definition otherwise compiler goes wild

template void
AoSLinear<float>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                  NuclideComponent<float> &nuclide);

template void
AoSLinear<double>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                   NuclideComponent<double> &nuclide);

template void
SoALinear<double>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                   NuclideComponent<double> &nuclide);

template void
SoALinear<float>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                  NuclideComponent<float> &nuclide);

template AoSLinear<float>::ViewType AoSLinear<float>::uploadToDevice();
template AoSLinear<double>::ViewType AoSLinear<double>::uploadToDevice();
template SoALinear<float>::ViewType SoALinear<float>::uploadToDevice();
template SoALinear<double>::ViewType SoALinear<double>::uploadToDevice();

template void LogarithmicHashAoS<float>::setLogarithmicHashGrid(size_t);
template void LogarithmicHashAoS<double>::setLogarithmicHashGrid(size_t);
template LogarithmicHashAoS<float>::ViewType
LogarithmicHashAoS<float>::uploadToDevice();
template LogarithmicHashAoS<double>::ViewType
LogarithmicHashAoS<double>::uploadToDevice();

} // namespace neuxs
