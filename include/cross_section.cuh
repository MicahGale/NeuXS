#ifndef NEUXS_CROSS_SECTION_CUH
#define NEUXS_CROSS_SECTION_CUH

#include <cuda_runtime.h>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuco/dynamic_map.cuh>

#include "hdf5.h"

#include "cross_section_reader.h"
#include "material.cuh"

namespace neuxs {

class OpenMCCrossSectionReader;

template <typename T> using DeviceVector = thrust::device_vector<T>;
template <typename T> using HostVector = thrust::host_vector<T>;

// ============= Main play ground for different data structure ==========
// =======================================================================
/*
 * CrossSectionGridPoint is the basic singular templated data structure we will
 * explore for AoS data structure. Energy grid needs to be separate.
 */

template <typename T> struct CrossSectionGridPoint {

  CrossSectionGridPoint(T sigma_s, T sigma_f, T sigma_c)
      : _sigma_s(sigma_s), _sigma_f(sigma_f), _sigma_c(sigma_c),
        _sigma_t(_sigma_c + _sigma_f + _sigma_s) {}

  T _sigma_s;
  T _sigma_f;
  T _sigma_c;
  T _sigma_t;
};

/*
 * CrossSectionGridPoint is the basic singular templated data structure we will
 * explore for SoA data structure. Cross-section data will be stored in device
 * vector.
 */

template <typename T> struct CrossSectionArray {

  CrossSectionArray(DeviceVector<T> sigma_s, DeviceVector<T> sigma_f,
                    DeviceVector<T> sigma_c)
      : _sigma_s(sigma_s), _sigma_f(sigma_f), _sigma_c(sigma_c) {

    _sigma_t.resize(_sigma_s.size());
    for (size_t i = 0; i < _sigma_s.size(); i++)
      _sigma_t[i] = _sigma_s[i] + _sigma_f[i] + _sigma_c[i];
  }

  DeviceVector<T> _sigma_s;
  DeviceVector<T> _sigma_f;
  DeviceVector<T> _sigma_c;
  DeviceVector<T> _sigma_t;
};

template <typename T> struct HashMap {};

// ===================Cross section Base class ====================
//              This is only for a single nuclide
// ================================================================

/*
 * Base class for cross-section data type. T1 will be cross-section data
 * structure.T2 will numeric data type. either float of double. All the daughter
 * class will just declare the T1 based on which cross-section data structure
 * they are using.
 */

template <typename T1, typename T2> class CrossSection {
public:
  /* cross setter from OpenMCCrossSectionReader object
   * this method will use the OpenMCCrossSectionReader object
   * to set the custom data type associated with that cross_section
   * class.
   */

  __host__ virtual void setCrossSection(const OpenMCCrossSectionReader &reader,
                                        NuclideComponent &nuclide) = 0;

  /* It needs to be abstract as we will implement
   * different kind of interpolation methods
   */
  __device__ virtual CrossSectionGridPoint<T2> getCrossSection(T2 *energy) = 0;

  /* play ground for different grid search methodology
   */
  __device__ virtual size_t searchEnergyGrid(T2 *energy) = 0;

  // Energy grid. I need to talk to Micah about this
  // design and there could be a future change if we decide to  use energy
  // grid's lethargy value for search. Just an idea.

  DeviceVector<T2> _energy;
};

template <typename T>
class AoSLinear : public CrossSection<CrossSectionGridPoint<T>, T> {

public:
  __host__ virtual void setCrossSection(const OpenMCCrossSectionReader &reader,
                                        NuclideComponent &nuclide) override;

  __device__ virtual size_t searchEnergyGrid(T *energy) override {};
  // linear-linear interpolation methods here
  __device__ virtual CrossSectionGridPoint<T>
  getCrossSection(T *energy) override {};

  DeviceVector<CrossSectionGridPoint<T>> _device_data;
};

template <typename T> class SoALinear : CrossSection<CrossSectionArray<T>, T> {

public:
  __host__ virtual void setCrossSection(const OpenMCCrossSectionReader &reader,
                                        NuclideComponent &nuclide) override;

  __device__ virtual size_t searchEnergyGrid(T *energy) override;
  // linear-linear interpolation methods here
  __device__ virtual CrossSectionGridPoint<T>
  getCrossSection(T *energy) override;

  CrossSectionArray<T> _device_data;
};

} // namespace neuxs

#endif // NEUXS_CROSS_SECTION_CUH
