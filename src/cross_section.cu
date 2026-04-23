#include <algorithm>
#include <filesystem>
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
  // check if already uploaded or not
  if (_uploaded)
    return _cached_view;

  const size_t n = this->_size;
  // Allocate + copy both host arrays to device in one shot each.
  _d_energy = DeviceBuffer<FPrecision>::makeFromHost(this->_energy, n);
  _d_xs_data = DeviceBuffer<CrossSectionGridPoint<FPrecision>>::makeFromHost(
      this->_xs_data, n);

  _cached_view._energy = _d_energy.get();
  _cached_view._grid = _d_xs_data.get();
  _cached_view._size = n;
  _uploaded = true;
  return _cached_view;
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

} // namespace neuxs
