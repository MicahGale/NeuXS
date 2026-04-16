#include "cross_section.cuh"
#include <algorithm>
#include <filesystem>

namespace neuxs {

template <typename T>
__device__ void CrossSection<T>::getCrossSection(DeviceVector<float> *energy,
                                                 DeviceVector<float> *sigma_s,
                                                 DeviceVector<float> *sigma_c,
                                                 DeviceVector<float> *sigma_f,
                                                 DeviceVector<float> *sigma_t) {
  return static_cast<T>(this)->get_sigma(energy, sigma_s, sigma_c, sigma_f,
                                         sigma_t);
}

template <typename T>
__device__ void CrossSection<T>::interpolate(float x1, float x2, float x_val,
                                             float y1, float y2, float *y_val) {
  float x_delta = x2 - x1;
  float y_delta = y2 - y1;
  *y_val = y1 + y_delta * (x_val - x1) / x_delta;
}

NuclideCrossSectionSet::NuclideCrossSectionSet(
    const std::vector<float> &energy, const std::vector<float> &sigma_s,
    const std::vector<float> &sigma_f, const std::vector<float> &sigma_t,
    const std::vector<float> &sigma_c) {

  preCheck(energy, sigma_s, sigma_f, sigma_t, sigma_c);

  const auto size = energy.size();
  _cross_section_grids.reserve(size);

  for (size_t i = 0; i < size; i++)
    _cross_section_grids.push_back(CrossSectionGridPoint(
        energy[i], sigma_s[i], sigma_f[i], sigma_t[i], sigma_c[i]));

  _cross_section_grids.shrink_to_fit();
}

void NuclideCrossSectionSet::preCheck(const std::vector<float> &energy,
                                      const std::vector<float> &sigma_s,
                                      const std::vector<float> &sigma_f,
                                      const std::vector<float> &sigma_t,
                                      const std::vector<float> &sigma_c) {

  const auto n = energy.size();
  const bool check = (sigma_s.size() == n && sigma_f.size() == n &&
                      sigma_t.size() == n && sigma_c.size() == n);

  if (!check)
    throw std::runtime_error(" Invalid cross section data point. Size of "
                             "reaction data don't match with each other");
}
}; // namespace neuxs
