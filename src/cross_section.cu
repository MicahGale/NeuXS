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

  std::vector<CrossSectionGridPoint<T>> grid_points_host;
  grid_points_host.reserve(size);

  for (std::size_t i = 0; i < size; i++) {
    grid_points_host.emplace_back(scattering[i], fission[i], capture[i]);
  }

  this->_device_data = DeviceVector<CrossSectionGridPoint<T>>(
      grid_points_host.begin(), grid_points_host.end());
}



// need explicit definition otherwise compiler goes wild

template void
AoSLinear<float>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                  NuclideComponent &nuclide);

template void
AoSLinear<double>::setCrossSection(const OpenMCCrossSectionReader &reader,
                                   NuclideComponent &nuclide);



} // namespace neuxs