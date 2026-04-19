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

template <typename FPrecision>
using DeviceVector = thrust::device_vector<FPrecision>;
template <typename FPrecision>
using HostVector = thrust::host_vector<FPrecision>;

// ============= Main play ground for different data structure ==========
// =======================================================================
/*
 * CrossSectionGridPoint is the basic singular templated data structure we will
 * explore for AoS data structure. Energy grid needs to be separate.
 */

template <typename FPrecision> struct CrossSectionGridPoint {

  CrossSectionGridPoint(FPrecision sigma_s, FPrecision sigma_f,
                        FPrecision sigma_c)
      : _sigma_s(sigma_s), _sigma_f(sigma_f), _sigma_c(sigma_c),
        _sigma_t(_sigma_c + _sigma_f + _sigma_s) {}

  FPrecision _sigma_s;
  FPrecision _sigma_f;
  FPrecision _sigma_c;
  FPrecision _sigma_t;
};

/*
 * CrossSectionArray is the basic singular templated data structure we'll use
 * explore for SoA data structure. Cross-section data will be stored in device
 * vector.
 */

template <typename FPrecision> struct CrossSectionArray {
  CrossSectionArray() = default;
  CrossSectionArray(DeviceVector<FPrecision> sigma_s,
                    DeviceVector<FPrecision> sigma_f,
                    DeviceVector<FPrecision> sigma_c)
      : _sigma_s(sigma_s), _sigma_f(sigma_f), _sigma_c(sigma_c) {

    _sigma_t.resize(_sigma_s.size());
    for (size_t i = 0; i < _sigma_s.size(); i++)
      _sigma_t[i] = _sigma_s[i] + _sigma_f[i] + _sigma_c[i];
  }

  DeviceVector<FPrecision> _sigma_s;
  DeviceVector<FPrecision> _sigma_f;
  DeviceVector<FPrecision> _sigma_c;
  DeviceVector<FPrecision> _sigma_t;
};

template <typename FPrecision> struct HashMap {};

// ===================Cross section Base class ====================
//              This is only for a single nuclide
// ================================================================

/*
 * Base class for cross-section data type. XSType will be cross-section data
 * structure. FPrecision will numeric data type. either float of double. All the
 * daughter class will just declare the T1 based on which cross-section data
 * structure they are using.
 */

template <typename XSType, typename FPrecision> class CrossSection {
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
  __device__ virtual void
  getCrossSection(FPrecision *energy,
                  CrossSectionGridPoint<FPrecision> *xs_grid) = 0;

  /* play ground for different grid search methodology
   */
  __device__ virtual size_t searchEnergyGrid(FPrecision *energy) = 0;

  // Energy grid. I need to talk to Micah about this
  // design and there could be a future change if we decide to  use energy
  // grid's lethargy value for search. Just an idea.

  DeviceVector<FPrecision> _energy;
};

template <typename FPrecision>
class AoSLinear
    : public CrossSection<CrossSectionGridPoint<FPrecision>, FPrecision> {

public:
  __host__ virtual void setCrossSection(const OpenMCCrossSectionReader &reader,
                                        NuclideComponent &nuclide) override;

  __device__ virtual size_t searchEnergyGrid(FPrecision *energy) override {};

  // linear-linear interpolation methods here
  __device__ virtual void
  getCrossSection(FPrecision *energy,
                  CrossSectionGridPoint<FPrecision> *xs_grid) override {};

  DeviceVector<CrossSectionGridPoint<FPrecision>> _device_data;
};

template <typename FPrecision>
class SoALinear : CrossSection<CrossSectionArray<FPrecision>, FPrecision> {

public:
  __host__ virtual void setCrossSection(const OpenMCCrossSectionReader &reader,
                                        NuclideComponent &nuclide) override;

  __device__ virtual size_t searchEnergyGrid(FPrecision *energy) override {};

  // linear-linear interpolation methods here
  __device__ virtual void
  getCrossSection(FPrecision *energy,
                  CrossSectionGridPoint<FPrecision> *xs_grid) override {};

  CrossSectionArray<FPrecision> _device_data;
};

} // namespace neuxs

#endif // NEUXS_CROSS_SECTION_CUH
