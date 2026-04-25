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

// device_cbrt follows the same dispatch pattern as device_log; used by the
// geometry's `particleEscapesTheCell` to compute a characteristic length
// from a cell volume.
template <typename T> __device__ T device_cbrt(T val);
template <> __device__ __forceinline__ float device_cbrt<float>(float val) {
  return cbrtf(val);
}
template <> __device__ __forceinline__ double device_cbrt<double>(double val) {
  return cbrt(val);
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

  // Scalar hash parameters. `_grid_energy_minimum` is ln(E_min) despite the
  // name (kept for backwards compat with the original comment above).
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

  // Binary search. Returns the index of the lower bracket; caller uses
  // (idx, idx+1) for interpolation.
  __device__ __forceinline__ size_t searchEnergyGrid(FPrecision energy) const {
    if (energy <= _energy[0])
      return 0;
    if (energy >= _energy[_size - 1])
      return _size - 2;

    size_t lo = 0;
    size_t hi = _size - 1;
    while (hi - lo > 1) {
      size_t mid = (lo + hi) >> 1;
      if (_energy[mid] <= energy)
        lo = mid;
      else
        hi = mid;
    }
    return lo;
  }

  // Lin-lin interpolation of all four reaction channels at `energy`.
  __device__ __forceinline__ CrossSectionGridPoint<FPrecision>
  getCrossSection(FPrecision energy) const {
    size_t idx = searchEnergyGrid(energy);
    FPrecision E_lo = _energy[idx];
    FPrecision E_hi = _energy[idx + 1];
    FPrecision f = (energy - E_lo) / (E_hi - E_lo);

    const auto &p_lo = _grid[idx];
    const auto &p_hi = _grid[idx + 1];

    CrossSectionGridPoint<FPrecision> r;
    r._sigma_s = p_lo._sigma_s + f * (p_hi._sigma_s - p_lo._sigma_s);
    r._sigma_f = p_lo._sigma_f + f * (p_hi._sigma_f - p_lo._sigma_f);
    r._sigma_c = p_lo._sigma_c + f * (p_hi._sigma_c - p_lo._sigma_c);
    r._sigma_t = p_lo._sigma_t + f * (p_hi._sigma_t - p_lo._sigma_t);
    return r;
  }
};

template <typename FPrecision> struct SoALinearView {
  FPrecision *_energy;                 // device ptr
  CrossSectionArray<FPrecision> _data; // each pointer inside is device
  size_t _size;

  __device__ __forceinline__ size_t searchEnergyGrid(FPrecision energy) const {
    if (energy <= _energy[0])
      return 0;
    if (energy >= _energy[_size - 1])
      return _size - 2;

    size_t lo = 0;
    size_t hi = _size - 1;
    while (hi - lo > 1) {
      size_t mid = (lo + hi) >> 1;
      if (_energy[mid] <= energy)
        lo = mid;
      else
        hi = mid;
    }
    return lo;
  }

  __device__ __forceinline__ CrossSectionGridPoint<FPrecision>
  getCrossSection(FPrecision energy) const {
    size_t idx = searchEnergyGrid(energy);
    FPrecision E_lo = _energy[idx];
    FPrecision E_hi = _energy[idx + 1];
    FPrecision f = (energy - E_lo) / (E_hi - E_lo);

    CrossSectionGridPoint<FPrecision> r;
    r._sigma_s = _data._sigma_s[idx] +
                 f * (_data._sigma_s[idx + 1] - _data._sigma_s[idx]);
    r._sigma_f = _data._sigma_f[idx] +
                 f * (_data._sigma_f[idx + 1] - _data._sigma_f[idx]);
    r._sigma_c = _data._sigma_c[idx] +
                 f * (_data._sigma_c[idx + 1] - _data._sigma_c[idx]);
    r._sigma_t = _data._sigma_t[idx] +
                 f * (_data._sigma_t[idx + 1] - _data._sigma_t[idx]);
    return r;
  }
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

  __device__ __forceinline__ size_t searchEnergyGrid(FPrecision energy) const {
    // Locate hash bin. Clamp to the valid range.
    if (energy <= _base._energy[0])
      return 0;
    if (energy >= _base._energy[_base._size - 1])
      return _base._size - 2;

    FPrecision log_e = device_log(energy);
    long bin = static_cast<long>((log_e - _log_energy_min) * _hash_delta);
    if (bin < 0)
      bin = 0;
    if (bin >= static_cast<long>(_n_bins))
      bin = static_cast<long>(_n_bins) - 1;

    size_t lo = _hash_table[bin];
    size_t hi = _hash_table[bin + 1];
    if (hi >= _base._size)
      hi = _base._size - 1;
    if (hi <= lo)
      return lo;

    // Short binary search within the narrowed window.
    while (hi - lo > 1) {
      size_t mid = (lo + hi) >> 1;
      if (_base._energy[mid] <= energy)
        lo = mid;
      else
        hi = mid;
    }
    return lo;
  }

  __device__ __forceinline__ CrossSectionGridPoint<FPrecision>
  getCrossSection(FPrecision energy) const {
    size_t idx = searchEnergyGrid(energy);
    FPrecision E_lo = _base._energy[idx];
    FPrecision E_hi = _base._energy[idx + 1];
    FPrecision f = (energy - E_lo) / (E_hi - E_lo);

    const auto &p_lo = _base._grid[idx];
    const auto &p_hi = _base._grid[idx + 1];

    CrossSectionGridPoint<FPrecision> r;
    r._sigma_s = p_lo._sigma_s + f * (p_hi._sigma_s - p_lo._sigma_s);
    r._sigma_f = p_lo._sigma_f + f * (p_hi._sigma_f - p_lo._sigma_f);
    r._sigma_c = p_lo._sigma_c + f * (p_hi._sigma_c - p_lo._sigma_c);
    r._sigma_t = p_lo._sigma_t + f * (p_hi._sigma_t - p_lo._sigma_t);
    return r;
  }
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

protected:
  // `protected` so LogarithmicHashAoS can reuse the cache after overriding
  // its own ViewType.
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
  // Shadow the base ViewType: kernels using a LogarithmicHashAoS will see
  // the hashed view. Everything upstream (Material, Cell) picks this up
  // automatically via `XSClass::ViewType`.
  using ViewType = LogarithmicHashAoSView<FPrecision>;

  ~LogarithmicHashAoS() override { delete[] _hash_table_host; }

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

  // Device-side backing storage for the hash table (energy + grid reuse
  // the base class's DeviceBuffers).
  DeviceBuffer<size_t> _d_hash_table;

private:
  bool _hash_uploaded = false;
  ViewType _cached_hash_view{};
};

} // namespace neuxs

#endif // NEUXS_CROSS_SECTION_CUH
