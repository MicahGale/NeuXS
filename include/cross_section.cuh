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

template <typename T> __device__ T device_log(T val);
template <> __device__ __forceinline__ float device_log<float>(float val) {
  return logf(val);
}
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
  CrossSectionGridPoint() {}
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

  FPrecision _grid_energy_minimum;
  FPrecision _grid_energy_delta;
};

// ================ Device-facing "view" structs (POD) =====================
// ===========================================================================
/*
 * A *View* is a lightweight, trivially-copyable POD struct that kernels
 * operate on. It holds **device** pointers only, so pointer-chasing from a
 * View lands on more device memory — not garbage.
 *
 * Naming convention: the host class X has an associated view type named
 * XView, exposed via `using ViewType = XView<...>`. Kernels only ever see
 * Views; they never see host classes. Host classes own the underlying
 * DeviceBuffer storage — the views are non-owning references.
 *
 * Adding a new lookup/interpolation scheme means:
 *   1) write a new host class deriving from CrossSection<...>
 *   2) write its View struct here with __device__ lookup methods
 *   3) expose `using ViewType = NewView<FP>;` and an `uploadToDevice()`
 * Nothing in Material, Cell, or the kernels needs to change.
 */

template <typename FPrecision> struct AoSLinearView {
  FPrecision *_energy;                      // device ptr
  CrossSectionGridPoint<FPrecision> *_grid; // device ptr
  size_t _size;

  // Linear (binary) search on the energy grid. Returns the index of the
  // lower bracket for interpolation.
  __device__ size_t searchEnergyGrid(FPrecision energy) const;

  // Look up the (interpolated) cross-section grid point at `energy`.
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
 *
 * Device story: each concrete class also defines an associated `ViewType`
 * (a POD) and an `uploadToDevice()` method. The view is what kernels see;
 * it holds device pointers to the backing storage owned by DeviceBuffers
 * on the host object. This is how we get a true deep copy onto the GPU
 * without losing the OOP structure on the host side.
 */

template <typename XSType, typename FPrecision> class CrossSection {
public:
  CrossSection() = default;
  virtual ~CrossSection() { delete[] _energy; }

  // Non-copyable: owns raw arrays. (Move could be added later if needed.)
  CrossSection(const CrossSection &) = delete;
  CrossSection &operator=(const CrossSection &) = delete;

  /* cross setter from OpenMCCrossSectionReader object
   * this method will use the OpenMCCrossSectionReader object
   * to set the custom data type associated with that cross_section
   * class.
   */

  __host__ virtual void
  setCrossSection(const OpenMCCrossSectionReader &reader,
                  NuclideComponent<FPrecision> &nuclide) = 0;

  FPrecision *_energy = nullptr;
  size_t _size = 0; // length of _energy and of the derived class's xs array
};

template <typename FPrecision>
class AoSLinear
    : public CrossSection<CrossSectionGridPoint<FPrecision>, FPrecision> {

public:
  // Associated view type — this is what kernels see.
  using ViewType = AoSLinearView<FPrecision>;

  ~AoSLinear() override { delete[] _xs_data; }

  __host__ virtual void
  setCrossSection(const OpenMCCrossSectionReader &reader,
                  NuclideComponent<FPrecision> &nuclide) override;

  /*
   * Deep-copy host data onto the device and produce a kernel-facing View.
   *
   * The device memory is owned by the DeviceBuffer members below, so the
   * lifetime of the returned view is tied to *this* object. The call is
   * idempotent: repeated calls return the same view without re-uploading.
   */
  __host__ ViewType uploadToDevice();

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

  // Host-side cross-section data (used during construction and for host
  // verification / unit tests). NOTE: previously named `_device_data`, but
  // that was misleading — the memory was plain `new[]` on the host. Renamed
  // to reflect reality. The actual device copy lives in `_d_xs_data` below.
  CrossSectionGridPoint<FPrecision> *_xs_data = nullptr;

  // Device-resident backing storage (owned here via RAII).
  DeviceBuffer<FPrecision> _d_energy;
  DeviceBuffer<CrossSectionGridPoint<FPrecision>> _d_xs_data;

private:
  bool _uploaded = false;
  ViewType _cached_view{};
};

template <typename FPrecision>
class SoALinear
    : public CrossSection<CrossSectionArray<FPrecision>, FPrecision> {

public:
  using ViewType = SoALinearView<FPrecision>;

  ~SoALinear() override {
    delete[] _xs_data._sigma_s;
    delete[] _xs_data._sigma_f;
    delete[] _xs_data._sigma_c;
    delete[] _xs_data._sigma_t;
  };
  __host__ virtual void
  setCrossSection(const OpenMCCrossSectionReader &reader,
                  NuclideComponent<FPrecision> &nuclide) override;

  __host__ ViewType uploadToDevice();

  /* It needs to be abstract as we will implement
   * different kind of interpolation methods
   */
  __device__ void getCrossSection(FPrecision *energy,
                                  CrossSectionGridPoint<FPrecision> *xs_grid);

  /* play ground for different grid search methodology
   */
  __device__ size_t searchEnergyGrid(FPrecision *energy);

  // Host-side SoA storage (renamed from `_device_data` for clarity — the
  // underlying pointers are host memory allocated with `new[]`).
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
  // Inherits ViewType from AoSLinear. If the hashed lookup needs extra
  // fields (e.g. grid_delta, min energy), define a new view struct here
  // and shadow `using ViewType = ...;` — nothing upstream changes.

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
