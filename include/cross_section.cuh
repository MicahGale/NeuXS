#ifndef NEUXS_CROSS_SECTION_CUH
#define NEUXS_CROSS_SECTION_CUH

#include <cmath>
#include <cuda_runtime.h>
#include <string>
#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuco/dynamic_map.cuh>

#include "faddeeva.cuh"
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

template <typename FPrecision> class PiecewiseSlbwModelView {
  /**
    Implementation based on the "Lulu Li notes."
    Li, Lulu. 22.211 Nuclear Reactor Physics I Notes. 2012. archived:
    <https://archive.org/details/ne-mit-notes-lulu>.
   */
private:
  // https://en.wikipedia.org/wiki/Planck_constant
  static constexpr FPrecision PLANCK_CONST = 6.582119e-16; //[eVs]
  // assume fission cross section is offset of absorption
  static constexpr FPrecision FISSION_MULTIPLIER = 0.1;
  FPrecision _A{0};
  FPrecision _kT{0};
  FPrecision _sigma_pot{0};
  FPrecision *_res_E0{nullptr};
  FPrecision *_res_gamma_n{nullptr};
  FPrecision *_res_gamma_g{nullptr};
  size_t _n_res{0};
  bool _fissile{false};

public:
  PiecewiseSlbwModelView() = default;
  PiecewiseSlbwModelView(FPrecision A, FPrecision kT, FPrecision sigma_pot,
                         FPrecision *_res_E0, FPrecision *res_gamma_n,
                         FPrecision *res_gamma_g, size_t n_res, bool fissile)
      : _A(A), _kT(kT), _sigma_pot(sigma_pot), _res_E0(_res_E0),
        _res_gamma_n(res_gamma_n), _res_gamma_g(res_gamma_g), _n_res(n_res),
        _fissile(fissile) {}

  __device__ size_t searchEnergyGrid(FPrecision energy) const {
    if (energy <= _res_E0[0])
      return 0;
    if (energy >= _res_E0[_n_res - 1])
      return _n_res - 2;

    size_t lo = 0;
    size_t hi = _n_res - 1;
    while (hi - lo > 1) {
      size_t mid = (lo + hi) >> 1;
      if (_res_E0[mid] <= energy)
        lo = mid;
      else
        hi = mid;
    }
    // find nearest resonance
    if ((energy - _res_E0[lo]) <= (_res_E0[hi] - energy))
      return lo;
    return hi;
  };
  __device__ thrust::complex<FPrecision> ugly_psi_xi(FPrecision x,
                                                     FPrecision xsi) {
    thrust::complex<FPrecision> inner_term =
        thrust::complex<FPrecision>(0.0, 1.0) *
        thrust::complex<FPrecision>(xsi, static_cast<FPrecision>(0.0)) *
        thrust::complex<FPrecision>(x, static_cast<FPrecision>(1.0)) / 2.0;
    thrust::complex<FPrecision> exp_term = thrust::exp(inner_term);
    return exp_term * exp_term * Faddeeva::erfc(-inner_term);
  }

  __device__ CrossSectionGridPoint<FPrecision>
  getCrossSection(FPrecision energy) {
    size_t E0_idx = this->searchEnergyGrid(energy);
    FPrecision E0 = this->_res_E0[E0_idx];
    FPrecision gg, gn, gamma;
    gg = this->_res_gamma_g[E0_idx];
    gn = this->_res_gamma_n[E0_idx];
    gamma = gg + gn;
    FPrecision x = 2 * (energy - E0) / gamma;
    FPrecision xsi = gamma * sqrt(this->_A / (4.0 * this->_kT * E0));
    thrust::complex<FPrecision> ugly_part =
        xsi / (2.0 * sqrt(M_PI)) * this->ugly_psi_xi(x, xsi);
    FPrecision psi = ugly_part.real();
    FPrecision xi = ugly_part.imag();
    FPrecision r_inner = (this->_A + 1.0) / this->_A;
    FPrecision r = 2603911.0 / energy * r_inner * r_inner;
    FPrecision q = sqrt(r * this->_sigma_pot);
    FPrecision sigma_capture = sqrt(E0 / energy) * gn / gamma * gg * r * psi;
    FPrecision sigma_scatter =
        gn * gn / (gamma * gamma) * (r * psi + q * xi) + this->_sigma_pot;
    FPrecision sigma_f =
        (this->_fissile) ? sigma_capture * this->FISSION_MULTIPLIER : 0.0;
    return CrossSectionGridPoint<FPrecision>(sigma_scatter, sigma_f,
                                             sigma_capture);
  }
};

template <typename FPrecision> class PiecewiseSlbwModel {
  /**
    Implementation based on the "Lulu Li notes."
    Li, Lulu. 22.211 Nuclear Reactor Physics I Notes. 2012. archived:
    <https://archive.org/details/ne-mit-notes-lulu>.

   * Smith, Kord. 22.212 Reactor Physics I: Lecture 2: Resonance Absorption.
   2017.
   */
private:
  static constexpr FPrecision R_0 = 1.2e-15; // 1.2 fm
  // https://en.wikipedia.org/wiki/Boltzmann_constant
  static constexpr FPrecision BOLTZMANN_CONST = 8.617333e-5; // eV/K

  FPrecision _A{0};
  FPrecision _kT{0};
  FPrecision _sigma_pot{0};
  size_t _n_res{0};
  bool _fissile{false};

  FPrecision *_res_E0_host{nullptr};
  FPrecision *_res_gamma_n_host{nullptr};
  FPrecision *_res_gamma_g_host{nullptr};

  DeviceBuffer<FPrecision> _d_res_E0;
  DeviceBuffer<FPrecision> _d_res_gamma_n;
  DeviceBuffer<FPrecision> _d_res_gamma_g;

  bool _uploaded{false};
  PiecewiseSlbwModelView<FPrecision> _cached_view;

public:
  using ViewType = PiecewiseSlbwModelView<FPrecision>;

  PiecewiseSlbwModel() = default;
  ~PiecewiseSlbwModel() {
    delete[] _res_E0_host;
    delete[] _res_gamma_n_host;
    delete[] _res_gamma_g_host;
  }

  void setCrossSection(const OpenMCCrossSectionReader &reader,
                       NuclideComponent<FPrecision> &nuclide);
  ViewType uploadToDevice();
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
 * in GPU methods like searchEnergyGrid() and getCrossSection() should be
 * exposed the mirror object in GPU. This should save us some complexity
 * regarding deep copying object from H2D.
 */

template <typename FPrecision> struct AoSLinearView {
  FPrecision *_energy;                      // device ptr
  CrossSectionGridPoint<FPrecision> *_grid; // device ptr
  size_t _size;

  __device__ size_t searchEnergyGrid(FPrecision energy) const {
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
  };
  __device__ CrossSectionGridPoint<FPrecision>
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
  };
};

template <typename FPrecision> struct SoALinearView {
  FPrecision *_energy;                 // device ptr
  CrossSectionArray<FPrecision> _data; // each pointer inside is device
  size_t _size;

  __device__ size_t searchEnergyGrid(FPrecision energy) const {
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
  };
  __device__ CrossSectionGridPoint<FPrecision>
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

  __device__ size_t searchEnergyGrid(FPrecision energy) const {
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

    while (hi - lo > 1) {
      size_t mid = (lo + hi) >> 1;
      if (_base._energy[mid] <= energy)
        lo = mid;
      else
        hi = mid;
    }
    return lo;
  }
  __device__ CrossSectionGridPoint<FPrecision>
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

  __host__ void prepareCrossSection(const OpenMCCrossSectionReader &reader,
                                    NuclideComponent<FPrecision> &nuclide);

  __host__ virtual void
  setCrossSection(const OpenMCCrossSectionReader &reader,
                  NuclideComponent<FPrecision> &nuclide) override {
    this->prepareCrossSection(reader, nuclide);
  };

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

  __host__ virtual void
  setCrossSection(const OpenMCCrossSectionReader &reader,
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

  __host__ virtual void setCrossSection(const OpenMCCrossSectionReader &reader,
                                        NuclideComponent<FPrecision> &nuclide) {
    this->prepareCrossSection(reader, nuclide);
    this->setLogarithmicHashGrid();
  };
  /*
   * Build the log-hash table from the already-populated energy grid. Call
   * *after* setCrossSection(). `n_bins` trades off table size vs.
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
