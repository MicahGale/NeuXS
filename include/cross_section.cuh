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

template <typename FPrecision>
using DynamicMap = cuco::dynamic_map<int, FPrecision>;

template <typename T> __device__ T device_log(T val);
template <> __device__ __forceinline__ float device_log<float>(float val) { return logf(val); }
template <> __device__ __forceinline__ double device_log<double>(double val) {
  return log(val);
}

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

template <typename FPrecision> struct HashGrid {
  // credit: https://github.com/ANL-CESAR/XSBench
  // Logarithmic_Hash_Grid_Search( Energy E, Material M ):
  //	macroscopic XS = 0
  //	hash_index = grid_delta * (ln(E) - grid_minimum_energy)
  //	for each nuclide in M do:
  //		i_low  = unionized_grid[nuclide, hash_index]
  //		i_high = unionized_grid[nuclide, hash_index+1]
  //		index = binary search in range(i_low, i_high) to find E in
  // nuclide grid 		interpolate data from grid[nuclide, index]
  // macroscopic XS += data
  __device__ void getHashIndex(FPrecision *energy, size_t *hash_index) {
    *hash_index = _grid_energy_delta *
                  (device_log<FPrecision>(energy) - _grid_energy_minimum);
  }

  FPrecision _grid_energy_minimum;
  FPrecision _grid_energy_delta;
};

// ===================Cross section Base class ====================
//              This is only for a single nuclide
// ================================================================

/*
 * Base class for cross-section data type. XSType will be cross-section data
 * structure. FPrecision will numeric data type. either float of double. All the
 * daughter class will just declare the T1 based on which cross-section data
 * structure they are using.
 *
 * The over all data structure map is given bellow. For the sake our own
 * sanity this needs to be changed every time we implement a new data
 * struct or change the hierarchy
 *
 *                     CrossSection<XSType, FPrecision>
 *              (This is the base class. Any new data structure class
 *               needs to inherit either from this or any of the
 *               daughter class. In the inherited class we implement
 *               the lookup and interpolation methods)
 *                                    │
 *              ┌─────────────────────┴─────────────────────┐
 *              │                                           │
 *     AoSLinear<FPrecision>                    SoALinear<FPrecision>
 *              │
 *              │
 *  LogarithmicHashAoS<FPrecision>
 */

template <typename XSType, typename FPrecision> class CrossSection {
public:
  /* cross setter from OpenMCCrossSectionReader object
   * this method will use the OpenMCCrossSectionReader object
   * to set the custom data type associated with that cross_section
   * class.
   */

  __host__ virtual void
  setCrossSection(const OpenMCCrossSectionReader &reader,
                  NuclideComponent<FPrecision> &nuclide) = 0;

  DeviceVector<FPrecision> _energy;
};

template <typename FPrecision>
class AoSLinear
    : public CrossSection<CrossSectionGridPoint<FPrecision>, FPrecision> {

public:
  __host__ virtual void
  setCrossSection(const OpenMCCrossSectionReader &reader,
                  NuclideComponent<FPrecision> &nuclide) override;

  /* It needs to be abstract as we will implement
   * different kind of interpolation methods
   */
  __device__ void getCrossSection(FPrecision *energy,
                                  CrossSectionGridPoint<FPrecision> *xs_grid) {
  };

  /* play ground for different grid search methodology
   */
  __device__ size_t searchEnergyGrid(FPrecision *energy) {};

  // Energy grid. I need to talk to Micah about this
  // design and there could be a future change if we decide to  use energy
  // grid's lethargy value for search. Just an idea.

  DeviceVector<CrossSectionGridPoint<FPrecision>> _device_data;
};

template <typename FPrecision>
class SoALinear
    : public CrossSection<CrossSectionArray<FPrecision>, FPrecision> {

public:
  __host__ virtual void
  setCrossSection(const OpenMCCrossSectionReader &reader,
                  NuclideComponent<FPrecision> &nuclide) override;

  /* It needs to be abstract as we will implement
   * different kind of interpolation methods
   */
  __device__ void getCrossSection(FPrecision *energy,
                                  CrossSectionGridPoint<FPrecision> *xs_grid);

  /* play ground for different grid search methodology
   */
  __device__ size_t searchEnergyGrid(FPrecision *energy);

  CrossSectionArray<FPrecision> _device_data;
};

template <typename FPrecision>
class LogarithmicHashAoS : public AoSLinear<FPrecision> {
public:
  // setter method _hash_info
  __host__ void setLogarithmicHashGrid();

  // we implement the interpolation method here
  __device__ void getCrossSection(FPrecision *energy,
                                  CrossSectionGridPoint<FPrecision> *xs_grid);
  // This method will call the getHashIndex
  __device__ size_t searchEnergyGrid(FPrecision *energy);

protected:
  HashGrid<FPrecision> _hash_info;
};

} // namespace neuxs

#endif // NEUXS_CROSS_SECTION_CUH
