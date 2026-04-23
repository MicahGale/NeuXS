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
  this->_device_data = new CrossSectionGridPoint<FPrecision>[size];

  for (size_t i = 0; i < size; i++) {
    this->_energy[i] = energy_host[i];
    CrossSectionGridPoint<FPrecision> grid(scattering[i], fission[i],
                                           capture[i]);
    this->_device_data[i] = grid;
  }
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
  this->_device_data._sigma_s = new FPrecision[size];
  this->_device_data._sigma_f = new FPrecision[size];
  this->_device_data._sigma_c = new FPrecision[size];
  this->_device_data._sigma_t = new FPrecision[size];

  for (size_t i = 0; i < size; i++) {
    this->_energy[i] = energy_host[i];
    this->_device_data._sigma_s[i] = scattering_host[i];
    this->_device_data._sigma_f[i] = fission_host[i];
    this->_device_data._sigma_c[i] = capture_host[i];
    this->_device_data._sigma_t[i] =
        scattering_host[i] + fission_host[i] + capture_host[i];
  }
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

} // namespace neuxs