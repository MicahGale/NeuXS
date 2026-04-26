#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "cross_section.cuh"
#include "cross_section_reader.h"

namespace neuxs {

// ======================== AoSLinearView =================================
template <typename FPrecision>
__device__ size_t
AoSLinearView<FPrecision>::searchEnergyGrid(FPrecision energy) const {
  if (energy <= _energy[0])
    return 0;
  if (energy >= _energy[_size - 1])
    return _size - 2;

  size_t lo = 0;
  size_t hi = _size - 1;
  while (hi - lo > 1) {
    size_t mid = (lo + hi) >> 1;
    if (_energy[mid] <= energy)
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}

template <typename FPrecision>
__device__ CrossSectionGridPoint<FPrecision>
AoSLinearView<FPrecision>::getCrossSection(FPrecision energy) const {
  size_t idx = searchEnergyGrid(energy);
  FPrecision E_lo = _energy[idx];
  FPrecision E_hi = _energy[idx + 1];
  FPrecision f = (energy - E_lo) / (E_hi - E_lo);

  const auto &p_lo = _grid[idx];
  const auto &p_hi = _grid[idx + 1];

  CrossSectionGridPoint<FPrecision> r;
  r._sigma_s = p_lo._sigma_s + f * (p_hi._sigma_s - p_lo._sigma_s);
  r._sigma_f = p_lo._sigma_f + f * (p_hi._sigma_f - p_lo._sigma_f);
  r._sigma_c = p_lo._sigma_c + f * (p_hi._sigma_c - p_lo._sigma_c);
  r._sigma_t = p_lo._sigma_t + f * (p_hi._sigma_t - p_lo._sigma_t);
  return r;
}

// ======================== SoALinearView =================================
template <typename FPrecision>
__device__ size_t
SoALinearView<FPrecision>::searchEnergyGrid(FPrecision energy) const {
  if (energy <= _energy[0])
    return 0;
  if (energy >= _energy[_size - 1])
    return _size - 2;

  size_t lo = 0;
  size_t hi = _size - 1;
  while (hi - lo > 1) {
    size_t mid = (lo + hi) >> 1;
    if (_energy[mid] <= energy)
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}

template <typename FPrecision>
__device__ CrossSectionGridPoint<FPrecision>
SoALinearView<FPrecision>::getCrossSection(FPrecision energy) const {
  size_t idx = searchEnergyGrid(energy);
  FPrecision E_lo = _energy[idx];
  FPrecision E_hi = _energy[idx + 1];
  FPrecision f = (energy - E_lo) / (E_hi - E_lo);

  CrossSectionGridPoint<FPrecision> r;
  r._sigma_s =
      _data._sigma_s[idx] + f * (_data._sigma_s[idx + 1] - _data._sigma_s[idx]);
  r._sigma_f =
      _data._sigma_f[idx] + f * (_data._sigma_f[idx + 1] - _data._sigma_f[idx]);
  r._sigma_c =
      _data._sigma_c[idx] + f * (_data._sigma_c[idx + 1] - _data._sigma_c[idx]);
  r._sigma_t =
      _data._sigma_t[idx] + f * (_data._sigma_t[idx + 1] - _data._sigma_t[idx]);
  return r;
}

// ======================== LogarithmicHashAoSView ========================
template <typename FPrecision>
__device__ size_t
LogarithmicHashAoSView<FPrecision>::searchEnergyGrid(FPrecision energy) const {
  if (energy <= _base._energy[0])
    return 0;
  if (energy >= _base._energy[_base._size - 1])
    return _base._size - 2;

  FPrecision log_e = device_log(energy);
  long bin = static_cast<long>((log_e - _log_energy_min) * _hash_delta);
  if (bin < 0)
    bin = 0;
  if (bin >= static_cast<long>(_n_bins))
    bin = static_cast<long>(_n_bins) - 1;

  size_t lo = _hash_table[bin];
  size_t hi = _hash_table[bin + 1];
  if (hi >= _base._size)
    hi = _base._size - 1;
  if (hi <= lo)
    return lo;

  while (hi - lo > 1) {
    size_t mid = (lo + hi) >> 1;
    if (_base._energy[mid] <= energy)
      lo = mid;
    else
      hi = mid;
  }
  return lo;
}

template <typename FPrecision>
__device__ CrossSectionGridPoint<FPrecision>
LogarithmicHashAoSView<FPrecision>::getCrossSection(FPrecision energy) const {
  size_t idx = searchEnergyGrid(energy);
  FPrecision E_lo = _base._energy[idx];
  FPrecision E_hi = _base._energy[idx + 1];
  FPrecision f = (energy - E_lo) / (E_hi - E_lo);

  const auto &p_lo = _base._grid[idx];
  const auto &p_hi = _base._grid[idx + 1];

  CrossSectionGridPoint<FPrecision> r;
  r._sigma_s = p_lo._sigma_s + f * (p_hi._sigma_s - p_lo._sigma_s);
  r._sigma_f = p_lo._sigma_f + f * (p_hi._sigma_f - p_lo._sigma_f);
  r._sigma_c = p_lo._sigma_c + f * (p_hi._sigma_c - p_lo._sigma_c);
  r._sigma_t = p_lo._sigma_t + f * (p_hi._sigma_t - p_lo._sigma_t);
  return r;
}

// ======================== CrossSection (base) ===========================
template <typename XSType, typename FPrecision>
CrossSection<XSType, FPrecision>::~CrossSection() {
  delete[] _energy;
}

// ======================== AoSLinear =====================================
template <typename FPrecision> AoSLinear<FPrecision>::~AoSLinear() {
  delete[] _xs_data;
}

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
  if (this->_uploaded)
    return this->_cached_view;

  const size_t n = this->_size;

  _d_energy = DeviceBuffer<FPrecision>::makeFromHost(this->_energy, n);
  _d_xs_data = DeviceBuffer<CrossSectionGridPoint<FPrecision>>::makeFromHost(
      this->_xs_data, n);

  this->_cached_view._energy = _d_energy.get();
  this->_cached_view._grid = _d_xs_data.get();
  this->_cached_view._size = n;
  this->_uploaded = true;
  return this->_cached_view;
}

// ======================== SoALinear =====================================
template <typename FPrecision> SoALinear<FPrecision>::~SoALinear() {
  delete[] _xs_data._sigma_s;
  delete[] _xs_data._sigma_f;
  delete[] _xs_data._sigma_c;
  delete[] _xs_data._sigma_t;
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

// ======================== LogarithmicHashAoS ============================
template <typename FPrecision>
LogarithmicHashAoS<FPrecision>::~LogarithmicHashAoS() {
  delete[] _hash_table_host;
}

/*
 * Hash-table construction: for each of (n_bins + 1) evenly spaced bin
 * boundaries in ln(E) space, record the largest grid index k such that
 * _energy[k] <= bin_boundary. Lookup at runtime then uses
 *     [hash_table[bin], hash_table[bin + 1]]
 * as the already-narrowed window for binary search. O(size + n_bins).
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

  this->_d_energy = DeviceBuffer<FPrecision>::makeFromHost(this->_energy, n);
  this->_d_xs_data =
      DeviceBuffer<CrossSectionGridPoint<FPrecision>>::makeFromHost(
          this->_xs_data, n);

  _d_hash_table =
      DeviceBuffer<size_t>::makeFromHost(_hash_table_host, _n_bins + 1);

  _cached_hash_view._base._energy = this->_d_energy.get();
  _cached_hash_view._base._grid = this->_d_xs_data.get();
  _cached_hash_view._base._size = n;
  _cached_hash_view._log_energy_min = _hash_info._grid_energy_minimum;
  _cached_hash_view._hash_delta = _hash_info._grid_energy_delta;
  _cached_hash_view._hash_table = _d_hash_table.get();
  _cached_hash_view._n_bins = _n_bins;

  this->_cached_view = _cached_hash_view._base;
  this->_uploaded = true;
  _hash_uploaded = true;

  return _cached_hash_view;
}

// need explicit definition otherwise compiler goes wild
template struct CrossSectionGridPoint<float>;
template struct CrossSectionGridPoint<double>;

template struct AoSLinearView<float>;
template struct AoSLinearView<double>;
template struct SoALinearView<float>;
template struct SoALinearView<double>;
template struct LogarithmicHashAoSView<float>;
template struct LogarithmicHashAoSView<double>;

template class CrossSection<CrossSectionGridPoint<float>, float>;
template class CrossSection<CrossSectionGridPoint<double>, double>;
template class CrossSection<CrossSectionArray<float>, float>;
template class CrossSection<CrossSectionArray<double>, double>;

template class AoSLinear<float>;
template class AoSLinear<double>;
template class SoALinear<float>;
template class SoALinear<double>;
template class LogarithmicHashAoS<float>;
template class LogarithmicHashAoS<double>;

} // namespace neuxs
