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
#include "memory.cuh"

namespace neuxs {

class OpenMCCrossSectionReader;

template <typename FPrecision>
using DeviceVector = thrust::device_vector<FPrecision>;
template <typename FPrecision>
using HostVector = thrust::host_vector<FPrecision>;

template <typename FPrecision>
using DynamicMap = cuco::dynamic_map<int, FPrecision>;

// ================ Math wrappers =====================
template <typename T> __device__ __forceinline__ T device_log(T val);
template <> __device__ __forceinline__ float device_log<float>(float val) {
  return logf(val);
};
template <> __device__ __forceinline__ double device_log<double>(double val) {
  return log(val);
};

template <typename T> __device__ __forceinline__ T device_cbrt(T val);
template <> __device__ __forceinline__ float device_cbrt<float>(float val) {
  return cbrtf(val);
};
template <> __device__ __forceinline__ double device_cbrt<double>(double val) {
  return cbrt(val);
};

// ============= Main play ground for different data structure ==========
// =======================================================================
/*
 * CrossSectionGridPoint is the basic singular templated data structure we will
 * explore for AoS data structure. Energy grid needs to be separate.
 */
template <typename FPrecision> struct CrossSectionGridPoint {
  __host__ __device__ CrossSectionGridPoint() {};
  __host__ __device__ CrossSectionGridPoint(FPrecision sigma_s,
                                            FPrecision sigma_f,
                                            FPrecision sigma_c)
      : _sigma_s(sigma_s), _sigma_f(sigma_f), _sigma_c(sigma_c),
        _sigma_t(sigma_c + sigma_f + sigma_s) {}

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

  FPrecision *_sigma_s;
  FPrecision *_sigma_f;
  FPrecision *_sigma_c;
  FPrecision *_sigma_t;
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
  __device__ void getHashIndex(FPrecision *energy, size_t *hash_index);

  // Scalar hash parameters. `_grid_energy_minimum` is ln(E_min) despite the
  // name (kept for backwards compat).
  FPrecision _grid_energy_minimum;
  FPrecision _grid_energy_delta;
};

// ================ Device-facing "view" structs (POD) =====================
// ===========================================================================
/*
 * Separating the HOST and Device object. Since we are only doing lookups
 * in GPU methods like searchEnergyGrid() and getCrossSection() should be exposed
 * the mirror object in GPU. This should save us some complexity regarding deep copying
 * object from H2D.
 */

template <typename FPrecision> struct AoSLinearView {
  FPrecision *_energy;                      // device ptr
  CrossSectionGridPoint<FPrecision> *_grid; // device ptr
  size_t _size;

  __device__ size_t searchEnergyGrid(FPrecision energy) const;
  __device__ CrossSectionGridPoint<FPrecision>
  getCrossSection(FPrecision energy) const;
};

template <typename FPrecision> struct SoALinearView {
  FPrecision *_energy;                 // device ptr
  CrossSectionArray<FPrecision> _data; // each pointer inside is device
  size_t _size;

  __device__ size_t searchEnergyGrid(FPrecision energy) const;
  __device__ CrossSectionGridPoint<FPrecision>
  getCrossSection(FPrecision energy) const;
};

/*
 * Log-hash accelerated AoS view. A pre-computed hash table narrows the
 * binary search range before we do the final lookup. Composition rather
 * than inheritance: embeds an `AoSLinearView` so it can share the
 * interpolation code.
 */
template <typename FPrecision> struct LogarithmicHashAoSView {
  AoSLinearView<FPrecision> _base; // energy + grid live here
  // Hash parameters
  FPrecision _log_energy_min; // ln(E_min)
  FPrecision _hash_delta;     // n_bins / (ln(E_max) - ln(E_min))
  size_t *_hash_table;        // device ptr, length = _n_bins + 1
  size_t _n_bins;

  __device__ size_t searchEnergyGrid(FPrecision energy) const;
  __device__ CrossSectionGridPoint<FPrecision>
  getCrossSection(FPrecision energy) const;
};

// =================== Cross section base class ===========================
// ========================================================================
/*
 *                     CrossSection<XSType, FPrecision>
 *                                    │
 *              ┌─────────────────────┴─────────────────────┐
 *              │                                           │
 *     AoSLinear<FPrecision>                    SoALinear<FPrecision>
 *              │
 *  LogarithmicHashAoS<FPrecision>
 *
 * Each concrete class has an associated `ViewType` (POD) and an
 * `uploadToDevice()` method. Kernels see views; host classes own backing
 * storage via DeviceBuffers.
 */
template <typename XSType, typename FPrecision> class CrossSection {
public:
  CrossSection() = default;
  virtual ~CrossSection();

  CrossSection(const CrossSection &) = delete;
  CrossSection &operator=(const CrossSection &) = delete;

  __host__ virtual void
  setCrossSection(const OpenMCCrossSectionReader &reader,
                  NuclideComponent<FPrecision> &nuclide) = 0;

  FPrecision *_energy = nullptr;
  size_t _size = 0;
};

template <typename FPrecision>
class AoSLinear
    : public CrossSection<CrossSectionGridPoint<FPrecision>, FPrecision> {

public:

  using ViewType = AoSLinearView<FPrecision>;

  ~AoSLinear() override;

  __host__ void setCrossSection(const OpenMCCrossSectionReader &reader,
                                NuclideComponent<FPrecision> &nuclide) override;

  __host__ ViewType uploadToDevice();

  // Host-side cross-section data.
  CrossSectionGridPoint<FPrecision> *_xs_data = nullptr;

  // Device-resident backing storage (owned via RAII).
  DeviceBuffer<FPrecision> _d_energy;
  DeviceBuffer<CrossSectionGridPoint<FPrecision>> _d_xs_data;

protected:
  bool _uploaded = false;
  ViewType _cached_view{};
};

template <typename FPrecision>
class SoALinear
    : public CrossSection<CrossSectionArray<FPrecision>, FPrecision> {
public:
  using ViewType = SoALinearView<FPrecision>;

  ~SoALinear() override;

  __host__ void setCrossSection(const OpenMCCrossSectionReader &reader,
                                NuclideComponent<FPrecision> &nuclide) override;

  __host__ ViewType uploadToDevice();

  // Host-side SoA storage.
  CrossSectionArray<FPrecision> _xs_data{};

  // Device-resident backing storage.
  DeviceBuffer<FPrecision> _d_energy;
  DeviceBuffer<FPrecision> _d_sigma_s;
  DeviceBuffer<FPrecision> _d_sigma_f;
  DeviceBuffer<FPrecision> _d_sigma_c;
  DeviceBuffer<FPrecision> _d_sigma_t;

private:
  bool _uploaded = false;
  ViewType _cached_view{};
};

template <typename FPrecision>
class LogarithmicHashAoS : public AoSLinear<FPrecision> {
public:
  // Shadow the base ViewType: kernels using a LogarithmicHashAoS will see
  // the hashed view. Everything upstream (Material, Cell) will pick this up
  // automatically via `XSClass::ViewType`.
  using ViewType = LogarithmicHashAoSView<FPrecision>;

  ~LogarithmicHashAoS() override;

  /*
   * Build the log-hash table from the already-populated energy grid. Call
   * *after* setCrossSection(). `n_bins` trades off table size vs. lookup
   * speed; 10k–100k is typical (XSBench default is ~10k).
   */
  __host__ void setLogarithmicHashGrid(size_t n_bins = 10000);

  __host__ ViewType uploadToDevice();

  // we implement the interpolation method here
  __device__ void getCrossSection(FPrecision *energy,
                                  CrossSectionGridPoint<FPrecision> *xs_grid);
  // This method will call the getHashIndex
  __device__ size_t searchEnergyGrid(FPrecision *energy);

protected:
  HashGrid<FPrecision> _hash_info;

  // Host-side hash table (length = _n_bins + 1) and its size.
  size_t *_hash_table_host = nullptr;
  size_t _n_bins = 0;

  DeviceBuffer<size_t> _d_hash_table;

private:
  bool _hash_uploaded = false;
  ViewType _cached_hash_view{};
};

} // namespace neuxs

#endif // NEUXS_CROSS_SECTION_CUH
