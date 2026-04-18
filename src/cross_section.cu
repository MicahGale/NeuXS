#include <algorithm>
#include <filesystem>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "cross_section.cuh"
#include "cross_section_reader.h"

namespace neuxs {

template <typename T>
void AoSLinear<T>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                   NuclideComponent &nuclide) {

  auto nuclide_name = std::string(nuclide._name);
  auto energy_host =
      reader.getEnergyDataPoints<T>(nuclide_name, nuclide._temperature);
  const auto size = energy_host.size();

  this->_energy = DeviceVector<T>(energy_host.begin(), energy_host.end());

  auto scattering = reader.getCrossSectionDataPoints<T>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::SCATTERING);

  auto capture = reader.getCrossSectionDataPoints<T>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::CAPTURE);

  std::vector<T> fission;
  if (nuclide._allows_fission) {
    fission = reader.getCrossSectionDataPoints<T>(
        nuclide_name, nuclide._temperature, CrossSectionDataType::FISSION);
    if (fission.empty())
      fission.reserve(size);

    if (fission.empty()) {
      fission.assign(size, static_cast<T>(0));
    }
  } else {
    fission.assign(size, static_cast<T>(0));
  }

  for (std::size_t i = 0; i < size; i++) {
    this->_device_data.push_back(
        CrossSectionGridPoint<T>(scattering[i], fission[i], capture[i]));
  }
}

template <typename T>
void SoALinear<T>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                   NuclideComponent &nuclide) {
  auto nuclide_name = std::string(nuclide._name);

  auto energy_host =
      reader.getEnergyDataPoints<T>(nuclide_name, nuclide._temperature);
  const auto size = energy_host.size();

  this->_energy = DeviceVector<T>(energy_host.begin(), energy_host.end());

  auto scattering_host = reader.getCrossSectionDataPoints<T>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::SCATTERING);

  auto capture_host = reader.getCrossSectionDataPoints<T>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::CAPTURE);

  std::vector<T> fission_host;
  if (nuclide._allows_fission) {
    fission_host = reader.getCrossSectionDataPoints<T>(
        nuclide_name, nuclide._temperature, CrossSectionDataType::FISSION);

    if (fission_host.empty())
      fission_host.assign(size, static_cast<T>(0));

  } else
    fission_host.assign(size, static_cast<T>(0));

  // I kinda agree now that when I made the cross-section reader I made some
  // poor choice
  DeviceVector<T> scattering_device(scattering_host.begin(),
                                    scattering_host.end());
  DeviceVector<T> fission_device(fission_host.begin(), fission_host.end());
  DeviceVector<T> capture_device(capture_host.begin(), capture_host.end());

  // The constructor will compute sigma_t = sigma_s + sigma_f + sigma_c
  this->_device_data =
      CrossSectionArray<T>(scattering_device, fission_device, capture_device);
}

// need explicit definition otherwise compiler goes wild

template void
AoSLinear<float>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                  NuclideComponent &nuclide);

template void
AoSLinear<double>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                   NuclideComponent &nuclide);

} // namespace neuxs