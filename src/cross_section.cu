#include <algorithm>
#include <filesystem>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "cross_section.cuh"
#include "cross_section_reader.h"

namespace neuxs {

template <typename FPrecision>
void AoSLinear<FPrecision>::setCrossSection(
    const OpenMCCrossSectionReader &reader, NuclideComponent &nuclide) {

  auto nuclide_name = std::string(nuclide._name);
  auto energy_host = reader.getEnergyDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature);
  const auto size = energy_host.size();

  this->_energy =
      DeviceVector<FPrecision>(energy_host.begin(), energy_host.end());

  auto scattering = reader.getCrossSectionDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::SCATTERING);

  auto capture = reader.getCrossSectionDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::CAPTURE);

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

  for (std::size_t i = 0; i < size; i++) {
    this->_device_data.push_back(CrossSectionGridPoint<FPrecision>(
        scattering[i], fission[i], capture[i]));
  }
}

template <typename FPrecision>
void SoALinear<FPrecision>::setCrossSection(
    const OpenMCCrossSectionReader &reader, NuclideComponent &nuclide) {
  auto nuclide_name = std::string(nuclide._name);

  auto energy_host = reader.getEnergyDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature);
  const auto size = energy_host.size();

  this->_energy =
      DeviceVector<FPrecision>(energy_host.begin(), energy_host.end());

  auto scattering_host = reader.getCrossSectionDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::SCATTERING);

  auto capture_host = reader.getCrossSectionDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::CAPTURE);

  std::vector<FPrecision> fission_host;
  if (nuclide._allows_fission) {
    fission_host = reader.getCrossSectionDataPoints<FPrecision>(
        nuclide_name, nuclide._temperature, CrossSectionDataType::FISSION);

    if (fission_host.empty())
      fission_host.assign(size, static_cast<FPrecision>(0));

  } else
    fission_host.assign(size, static_cast<FPrecision>(0));

  // I kinda agree now that when I made the cross-section reader I made some
  // poor choice
  DeviceVector<FPrecision> scattering_device(scattering_host.begin(),
                                             scattering_host.end());
  DeviceVector<FPrecision> fission_device(fission_host.begin(),
                                          fission_host.end());
  DeviceVector<FPrecision> capture_device(capture_host.begin(),
                                          capture_host.end());

  // The constructor will compute sigma_t = sigma_s + sigma_f + sigma_c
  this->_device_data = CrossSectionArray<FPrecision>(
      scattering_device, fission_device, capture_device);
}

// need explicit definition otherwise compiler goes wild

template void
AoSLinear<float>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                  NuclideComponent &nuclide);

template void
AoSLinear<double>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                   NuclideComponent &nuclide);

template void
SoALinear<double>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                   NuclideComponent &nuclide);

template void
SoALinear<float>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                  NuclideComponent &nuclide);

} // namespace neuxs