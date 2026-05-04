#include <algorithm>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "cross_section.cuh"
#include "cross_section_reader.h"

namespace neuxs {

// ======================== CrossSection (base) ===========================
template <typename XSType, typename FPrecision>
CrossSection<XSType, FPrecision>::~CrossSection() {
  delete[] _energy;
}

// ======================== AoSLinear =====================================
template <typename FPrecision> AoSLinear<FPrecision>::~AoSLinear() {
  delete[] _xs_data;
}

template <typename FPrecision>
void AoSLinear<FPrecision>::prepareCrossSection(
    const OpenMCCrossSectionReader &reader,
    NuclideComponent<FPrecision> &nuclide) {

  auto nuclide_name = std::string(nuclide._name);
  auto energy_host = reader.getEnergyDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature);
  auto scattering = reader.getCrossSectionDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::SCATTERING);
  auto capture = reader.getCrossSectionDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::CAPTURE);

  const auto size = energy_host.size();
  std::vector<FPrecision> fission;
  if (nuclide._allows_fission) {
    fission = reader.getCrossSectionDataPoints<FPrecision>(
        nuclide_name, nuclide._temperature, CrossSectionDataType::FISSION);
    if (fission.empty())
      fission.reserve(size);

    if (fission.empty()) {
      fission.assign(size, static_cast<FPrecision>(0));
    }
  } else {
    fission.assign(size, static_cast<FPrecision>(0));
  }

  this->_energy = new FPrecision[size];
  this->_xs_data = new CrossSectionGridPoint<FPrecision>[size];
  this->_size = size;

  for (size_t i = 0; i < size; i++) {
    this->_energy[i] = energy_host[i];
    CrossSectionGridPoint<FPrecision> grid(scattering[i], fission[i],
                                           capture[i]);
    this->_xs_data[i] = grid;
  }
}

template <typename FPrecision>
typename AoSLinear<FPrecision>::ViewType
AoSLinear<FPrecision>::uploadToDevice() {
  if (this->_uploaded)
    return this->_cached_view;

  const size_t n = this->_size;

  _d_energy = DeviceBuffer<FPrecision>::makeFromHost(this->_energy, n);
  _d_xs_data = DeviceBuffer<CrossSectionGridPoint<FPrecision>>::makeFromHost(
      this->_xs_data, n);

  this->_cached_view._energy = _d_energy.get();
  this->_cached_view._grid = _d_xs_data.get();
  this->_cached_view._size = n;
  this->_uploaded = true;
  return this->_cached_view;
}

// ======================== SoALinear =====================================
template <typename FPrecision> SoALinear<FPrecision>::~SoALinear() {
  delete[] _xs_data._sigma_s;
  delete[] _xs_data._sigma_f;
  delete[] _xs_data._sigma_c;
  delete[] _xs_data._sigma_t;
}

template <typename FPrecision>
void SoALinear<FPrecision>::setCrossSection(
    const OpenMCCrossSectionReader &reader,
    NuclideComponent<FPrecision> &nuclide) {
  auto nuclide_name = std::string(nuclide._name);

  auto energy_host = reader.getEnergyDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature);
  auto scattering_host = reader.getCrossSectionDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::SCATTERING);
  auto capture_host = reader.getCrossSectionDataPoints<FPrecision>(
      nuclide_name, nuclide._temperature, CrossSectionDataType::CAPTURE);
  std::vector<FPrecision> fission_host;
  const auto size = energy_host.size();
  if (nuclide._allows_fission) {
    fission_host = reader.getCrossSectionDataPoints<FPrecision>(
        nuclide_name, nuclide._temperature, CrossSectionDataType::FISSION);

    if (fission_host.empty())
      fission_host.assign(size, static_cast<FPrecision>(0));

  } else
    fission_host.assign(size, static_cast<FPrecision>(0));

  this->_energy = new FPrecision[size];
  this->_xs_data._sigma_s = new FPrecision[size];
  this->_xs_data._sigma_f = new FPrecision[size];
  this->_xs_data._sigma_c = new FPrecision[size];
  this->_xs_data._sigma_t = new FPrecision[size];
  this->_size = size;

  for (size_t i = 0; i < size; i++) {
    this->_energy[i] = energy_host[i];
    this->_xs_data._sigma_s[i] = scattering_host[i];
    this->_xs_data._sigma_f[i] = fission_host[i];
    this->_xs_data._sigma_c[i] = capture_host[i];
    this->_xs_data._sigma_t[i] =
        scattering_host[i] + fission_host[i] + capture_host[i];
  }
}

template <typename FPrecision>
typename SoALinear<FPrecision>::ViewType
SoALinear<FPrecision>::uploadToDevice() {
  if (_uploaded)
    return _cached_view;

  const size_t n = this->_size;

  _d_energy = DeviceBuffer<FPrecision>::makeFromHost(this->_energy, n);
  _d_sigma_s =
      DeviceBuffer<FPrecision>::makeFromHost(this->_xs_data._sigma_s, n);
  _d_sigma_f =
      DeviceBuffer<FPrecision>::makeFromHost(this->_xs_data._sigma_f, n);
  _d_sigma_c =
      DeviceBuffer<FPrecision>::makeFromHost(this->_xs_data._sigma_c, n);
  _d_sigma_t =
      DeviceBuffer<FPrecision>::makeFromHost(this->_xs_data._sigma_t, n);

  _cached_view._energy = _d_energy.get();
  _cached_view._data._sigma_s = _d_sigma_s.get();
  _cached_view._data._sigma_f = _d_sigma_f.get();
  _cached_view._data._sigma_c = _d_sigma_c.get();
  _cached_view._data._sigma_t = _d_sigma_t.get();
  _cached_view._size = n;
  _uploaded = true;
  return _cached_view;
}

// ======================== LogarithmicHashAoS ============================
template <typename FPrecision>
LogarithmicHashAoS<FPrecision>::~LogarithmicHashAoS() {
  delete[] _hash_table_host;
}

/*
 * Hash-table construction: for each of (n_bins + 1) evenly spaced bin
 * boundaries in ln(E) space, record the largest grid index k such that
 * _energy[k] <= bin_boundary. Lookup at runtime then uses
 *     [hash_table[bin], hash_table[bin + 1]]
 * as the already-narrowed window for binary search. O(size + n_bins).
 */
template <typename FPrecision>
void LogarithmicHashAoS<FPrecision>::setLogarithmicHashGrid(size_t n_bins) {
  if (!this->_energy || this->_size < 2) {
    throw std::runtime_error(
        "setLogarithmicHashGrid: call setCrossSection first");
  }

  _n_bins = n_bins;

  FPrecision E_min = this->_energy[0];
  FPrecision E_max = this->_energy[this->_size - 1];
  if (E_min <= static_cast<FPrecision>(0) || E_max <= E_min) {
    throw std::runtime_error(
        "setLogarithmicHashGrid: degenerate energy grid for log hash");
  }

  FPrecision log_min = std::log(E_min);
  FPrecision log_max = std::log(E_max);

  _hash_info._grid_energy_minimum = log_min;
  _hash_info._grid_energy_delta =
      static_cast<FPrecision>(_n_bins) / (log_max - log_min);

  delete[] _hash_table_host;
  _hash_table_host = new size_t[_n_bins + 1];

  // Single cursor sweep: k is the largest index with _energy[k] <= bin_e.
  _hash_table_host[0] = 0;
  size_t k = 0;
  for (size_t i = 1; i < _n_bins; i++) {
    FPrecision bin_log_e =
        log_min + static_cast<FPrecision>(i) / _hash_info._grid_energy_delta;
    FPrecision bin_e = std::exp(bin_log_e);

    while (k + 1 < this->_size && this->_energy[k + 1] <= bin_e)
      k++;

    _hash_table_host[i] = k;
  }
  _hash_table_host[_n_bins] = this->_size - 1;
}

template <typename FPrecision>
typename LogarithmicHashAoS<FPrecision>::ViewType
LogarithmicHashAoS<FPrecision>::uploadToDevice() {
  if (_hash_uploaded)
    return _cached_hash_view;

  if (!_hash_table_host) {
    throw std::runtime_error(
        "uploadToDevice: call setLogarithmicHashGrid first");
  }

  const size_t n = this->_size;

  this->_d_energy = DeviceBuffer<FPrecision>::makeFromHost(this->_energy, n);
  this->_d_xs_data =
      DeviceBuffer<CrossSectionGridPoint<FPrecision>>::makeFromHost(
          this->_xs_data, n);

  _d_hash_table =
      DeviceBuffer<size_t>::makeFromHost(_hash_table_host, _n_bins + 1);

  _cached_hash_view._base._energy = this->_d_energy.get();
  _cached_hash_view._base._grid = this->_d_xs_data.get();
  _cached_hash_view._base._size = n;
  _cached_hash_view._log_energy_min = _hash_info._grid_energy_minimum;
  _cached_hash_view._hash_delta = _hash_info._grid_energy_delta;
  _cached_hash_view._hash_table = _d_hash_table.get();
  _cached_hash_view._n_bins = _n_bins;

  this->_cached_view = _cached_hash_view._base;
  this->_uploaded = true;
  _hash_uploaded = true;

  return _cached_hash_view;
}

//======================= SLBW ====================================

template <typename FPrecision>
void PiecewiseSlbwModel<FPrecision>::setCrossSection(
    const OpenMCCrossSectionReader &, NuclideComponent<FPrecision> &nuclide) {
  _A = static_cast<FPrecision>(nuclide._A);
  _fissile = nuclide._allows_fission;
  _kT = nuclide._temperature * BOLTZMANN_CONST;
  FPrecision radius = R_0 * std::cbrt(_A);
  _sigma_pot = 4 * M_PI * radius * radius;

  std::string name(nuclide._name);

  if (name == "U235") {
    // First 20 positive s-wave resonances, ENDF/B-VIII.1 MF2/MT151
    static const double e0[] = {
        7.8908290e-03, 2.7013550e-01, 1.1223460e+00, 1.2618570e+00,
        2.0319390e+00, 2.7450120e+00, 3.1376980e+00, 3.6053370e+00,
        3.6261870e+00, 4.8477750e+00, 5.3910210e+00, 6.1724320e+00,
        6.3862980e+00, 6.8694720e+00, 7.0786970e+00, 7.5805380e+00,
        8.7565890e+00, 8.8096490e+00, 9.2692040e+00, 9.7512930e+00};
    static const double gn[] = {
        1.0511240e-09, 4.2853550e-06, 1.5131490e-05, 2.2334800e-07,
        8.8961560e-06, 4.5515900e-07, 2.6182410e-05, 4.4677540e-05,
        4.5568100e-07, 5.4090560e-05, 2.3033990e-05, 6.4539770e-05,
        2.2348260e-04, 8.4515380e-09, 1.1492460e-04, 3.1176370e-06,
        9.3514780e-04, 1.8867260e-04, 1.1177060e-04, 3.5823870e-05};
    static const double gg[] = {
        3.3188730e-02, 4.3357920e-02, 3.1256750e-02, 1.0823480e-01,
        4.3163060e-02, 2.3241890e-02, 3.7210630e-02, 3.6130830e-02,
        3.7618710e-02, 4.4763620e-02, 7.0347840e-02, 3.6633490e-02,
        4.3018400e-02, 3.2890450e-02, 3.8176510e-02, 7.6744090e-02,
        3.3992710e-02, 7.7680170e-02, 3.5665100e-02, 5.0280830e-02};
    _n_res = 20;
    _res_E0_host = new FPrecision[20];
    _res_gamma_n_host = new FPrecision[20];
    _res_gamma_g_host = new FPrecision[20];
    for (size_t i = 0; i < 20; ++i) {
      _res_E0_host[i] = static_cast<FPrecision>(e0[i]);
      _res_gamma_n_host[i] = static_cast<FPrecision>(gn[i]);
      _res_gamma_g_host[i] = static_cast<FPrecision>(gg[i]);
    }

  } else if (name == "U238") {
    // First 20 positive s-wave resonances, ENDF/B-VIII.1 MF2/MT151
    static const double e0[] = {
        6.6742800e+00, 2.0871770e+01, 3.6682830e+01, 6.6030210e+01,
        8.0751570e+01, 1.0256310e+02, 1.1690350e+02, 1.4566620e+02,
        1.6529710e+02, 1.8967510e+02, 2.0852260e+02, 2.3739870e+02,
        2.7367020e+02, 2.9100910e+02, 3.1132120e+02, 3.4781490e+02,
        3.5363660e+02, 3.7693420e+02, 3.9762020e+02, 4.1024910e+02};
    static const double gn[] = {
        1.4922140e-03, 1.0075630e-02, 3.3591510e-02, 2.4275730e-02,
        1.8553450e-03, 7.0872000e-02, 2.5412640e-02, 9.3993120e-04,
        3.4687730e-03, 1.7111770e-01, 5.0475840e-02, 2.6835650e-02,
        2.5270090e-02, 1.6605250e-02, 1.0602890e-03, 8.0215600e-02,
        2.1728210e-05, 1.1327670e-03, 6.2491830e-03, 1.9203250e-02};
    static const double gg[] = {
        2.2708240e-02, 2.2751040e-02, 2.2261450e-02, 2.2417210e-02,
        2.2584190e-02, 2.3171990e-02, 2.2060080e-02, 2.4522710e-02,
        2.4522710e-02, 2.1835790e-02, 2.2767750e-02, 2.4432020e-02,
        2.2615230e-02, 2.3219750e-02, 2.3092490e-02, 2.1653570e-02,
        2.3092490e-02, 2.3492810e-02, 2.3492810e-02, 2.6458120e-02};
    _n_res = 20;
    _res_E0_host = new FPrecision[20];
    _res_gamma_n_host = new FPrecision[20];
    _res_gamma_g_host = new FPrecision[20];
    for (size_t i = 0; i < 20; ++i) {
      _res_E0_host[i] = static_cast<FPrecision>(e0[i]);
      _res_gamma_n_host[i] = static_cast<FPrecision>(gn[i]);
      _res_gamma_g_host[i] = static_cast<FPrecision>(gg[i]);
    }

  } else if (name == "H1") {
    // No resolved resonances; single dummy at 20 MeV to give 1/v-like tail
    static const double e0[] = {2.0e+07};
    static const double gn[] = {1.0e+06};
    static const double gg[] = {1.0e-02};
    _n_res = 1;
    _res_E0_host = new FPrecision[1];
    _res_gamma_n_host = new FPrecision[1];
    _res_gamma_g_host = new FPrecision[1];
    _res_E0_host[0] = static_cast<FPrecision>(e0[0]);
    _res_gamma_n_host[0] = static_cast<FPrecision>(gn[0]);
    _res_gamma_g_host[0] = static_cast<FPrecision>(gg[0]);

  } else if (name == "O16") {
    // No resolved resonances in thermal range; single dummy at 20 MeV
    static const double e0[] = {2.0e+07};
    static const double gn[] = {1.0e+06};
    static const double gg[] = {1.0e-04};
    _n_res = 1;
    _res_E0_host = new FPrecision[1];
    _res_gamma_n_host = new FPrecision[1];
    _res_gamma_g_host = new FPrecision[1];
    _res_E0_host[0] = static_cast<FPrecision>(e0[0]);
    _res_gamma_n_host[0] = static_cast<FPrecision>(gn[0]);
    _res_gamma_g_host[0] = static_cast<FPrecision>(gg[0]);

  } else if (name == "Fe56") {
    // First 20 positive s-wave resonances, ENDF/B-VIII.1 MF2/MT151
    static const double e0[] = {
        2.7791000e+04, 7.4029000e+04, 8.3628000e+04, 1.2986100e+05,
        1.4047900e+05, 1.6927500e+05, 1.8773700e+05, 2.2058600e+05,
        2.4499100e+05, 2.7720600e+05, 3.1790900e+05, 3.3144700e+05,
        3.5726300e+05, 3.6107800e+05, 3.8136000e+05, 4.0540800e+05,
        4.3829600e+05, 4.6993400e+05, 5.0019400e+05, 5.3592100e+05};
    static const double gn[] = {
        1.4093000e+03, 6.1150000e+02, 1.2151000e+03, 5.8800000e+02,
        2.7350000e+03, 9.6200000e+02, 3.6200000e+03, 1.2670000e+03,
        4.8700000e+02, 3.6500000e+03, 7.1179000e+03, 3.2758000e+02,
        2.2050000e+03, 7.7750000e+03, 1.2330000e+04, 2.3290000e+03,
        1.9180000e+03, 2.5660000e+03, 1.7260000e+03, 2.5550000e+02};
    static const double gg[] = {
        1.2886600e+00, 6.9561000e-01, 5.2883000e-01, 6.2000000e-01,
        1.6100000e+00, 9.4000000e-01, 1.0500000e+00, 1.5000000e+00,
        7.0517000e-01, 1.1500000e+00, 3.3615900e+00, 5.1171000e-01,
        1.1000000e+00, 1.1000000e+00, 9.6000000e-01, 9.6000000e-01,
        9.6000000e-01, 9.6000000e-01, 9.6000000e-01, 9.6000000e-01};
    _n_res = 20;
    _res_E0_host = new FPrecision[20];
    _res_gamma_n_host = new FPrecision[20];
    _res_gamma_g_host = new FPrecision[20];
    for (size_t i = 0; i < 20; ++i) {
      _res_E0_host[i] = static_cast<FPrecision>(e0[i]);
      _res_gamma_n_host[i] = static_cast<FPrecision>(gn[i]);
      _res_gamma_g_host[i] = static_cast<FPrecision>(gg[i]);
    }

  } else if (name == "Cr52") {
    // First 20 positive s-wave resonances, ENDF/B-VIII.1 MF2/MT151
    static const double e0[] = {
        3.1638290e+04, 5.0325640e+04, 9.6693420e+04, 1.2194000e+05,
        1.4058940e+05, 2.6514150e+05, 3.2618590e+05, 3.6667490e+05,
        4.0236960e+05, 4.2289970e+05, 4.6331560e+05, 4.9393820e+05,
        5.0116420e+05, 5.3297710e+05, 6.1286100e+05, 7.1223590e+05,
        7.3817030e+05, 7.7245900e+05, 7.9487180e+05, 8.4199250e+05};
    static const double gn[] = {
        2.6101440e+01, 1.6200000e+03, 6.4674640e+03, 6.4019900e+02,
        6.0557540e+03, 2.2030880e+02, 8.3800830e+03, 6.1520480e+03,
        2.0685050e+04, 2.8296380e+03, 1.1757050e+04, 2.7930190e+02,
        1.2335690e+01, 6.0403630e+03, 2.0061000e+04, 6.1050000e+03,
        2.9883000e+04, 5.6860000e+03, 1.7622000e+04, 1.8478000e+03};
    static const double gg[] = {
        2.2055480e-01, 4.9000000e-01, 5.2000000e+00, 9.5181780e-01,
        1.3231860e+00, 2.1000090e-01, 3.7001380e-01, 1.3244470e+00,
        1.3200430e+00, 1.0905520e+00, 1.6744770e+00, 7.2797210e-01,
        1.1096760e+00, 1.4326450e+00, 1.3200000e+00, 1.3200000e+00,
        1.3200000e+00, 1.3200000e+00, 1.3200000e+00, 1.3200000e+00};
    _n_res = 20;
    _res_E0_host = new FPrecision[20];
    _res_gamma_n_host = new FPrecision[20];
    _res_gamma_g_host = new FPrecision[20];
    for (size_t i = 0; i < 20; ++i) {
      _res_E0_host[i] = static_cast<FPrecision>(e0[i]);
      _res_gamma_n_host[i] = static_cast<FPrecision>(gn[i]);
      _res_gamma_g_host[i] = static_cast<FPrecision>(gg[i]);
    }

  } else if (name == "Zr90") {
    // First 20 positive s-wave resonances, ENDF/B-VIII.1 MF2/MT151
    static const double e0[] = {
        3.8612000e+03, 1.3365000e+04, 1.3444000e+04, 1.7402000e+04,
        3.5356000e+04, 4.2250000e+04, 4.2455000e+04, 5.3264000e+04,
        5.3371000e+04, 5.7790000e+04, 6.5362000e+04, 7.0840000e+04,
        7.3330000e+04, 8.5730000e+04, 9.0370000e+04, 1.1410000e+05,
        1.2960000e+05, 1.4430000e+05, 1.5120000e+05, 1.5970000e+05};
    static const double gn[] = {
        1.0800000e+01, 3.0000000e+01, 5.3000000e+01, 2.2100000e+02,
        3.9000000e+01, 2.5000000e+02, 4.9000000e-01, 2.3000000e+01,
        1.5000000e+00, 2.1200000e+02, 7.0000000e-01, 2.0500000e+02,
        1.7000000e+02, 6.5000000e+00, 1.9000000e+01, 1.3600000e+02,
        1.4000000e+02, 3.2500000e+02, 2.2500000e+02, 4.9000000e+01};
    static const double gg[] = {
        7.8000000e-02, 6.8000000e-02, 5.3000000e-02, 1.6200000e-01,
        5.9000000e-01, 2.0000000e-01, 7.1000000e-02, 2.5000000e-02,
        9.4000000e-02, 4.2000000e-01, 3.0000000e-01, 1.3000000e-01,
        1.3000000e-01, 1.3000000e-01, 1.3000000e-01, 1.3000000e-01,
        1.3000000e-01, 1.3000000e-01, 1.3000000e-01, 1.3000000e-01};
    _n_res = 20;
    _res_E0_host = new FPrecision[20];
    _res_gamma_n_host = new FPrecision[20];
    _res_gamma_g_host = new FPrecision[20];
    for (size_t i = 0; i < 20; ++i) {
      _res_E0_host[i] = static_cast<FPrecision>(e0[i]);
      _res_gamma_n_host[i] = static_cast<FPrecision>(gn[i]);
      _res_gamma_g_host[i] = static_cast<FPrecision>(gg[i]);
    }

  } else if (name == "Sn120") {
    // First 20 positive s-wave resonances, ENDF/B-VIII.1 MF2/MT151
    static const double e0[] = {
        9.4860000e+02, 3.1184000e+03, 4.3556000e+03, 7.1185000e+03,
        7.3060000e+03, 8.6671000e+03, 8.9477000e+03, 1.1489000e+04,
        1.1791000e+04, 1.3053000e+04, 1.8029000e+04, 1.8950000e+04,
        1.9271000e+04, 2.0977000e+04, 2.3040000e+04, 2.3669000e+04,
        2.6224000e+04, 2.7960000e+04, 2.8610000e+04, 2.9334000e+04};
    static const double gn[] = {
        1.2390000e-01, 7.3490000e-01, 4.1400000e-01, 1.2040000e-01,
        5.4100000e-01, 9.2500000e-02, 7.5610000e+00, 3.3400000e-01,
        6.4600000e-01, 5.8900000e-01, 1.0130000e+00, 1.0930000e+00,
        2.0150000e+00, 3.7320000e+00, 6.1160000e+00, 1.3010000e+00,
        4.9890000e+00, 6.3000000e-01, 5.3700000e-01, 1.6050000e+00};
    static const double gg[] = {
        4.4300000e-02, 3.2400000e-02, 3.1400000e-02, 3.6700000e-02,
        3.1600000e-02, 3.4100000e-02, 3.0500000e-02, 3.0100000e-02,
        3.4400000e-02, 3.0100000e-02, 3.8700000e-02, 4.6400000e-02,
        4.3100000e-02, 4.0000000e-02, 4.0300000e-02, 4.5400000e-02,
        4.6900000e-02, 2.8000000e-02, 1.8900000e-02, 1.9200000e-02};
    _n_res = 20;
    _res_E0_host = new FPrecision[20];
    _res_gamma_n_host = new FPrecision[20];
    _res_gamma_g_host = new FPrecision[20];
    for (size_t i = 0; i < 20; ++i) {
      _res_E0_host[i] = static_cast<FPrecision>(e0[i]);
      _res_gamma_n_host[i] = static_cast<FPrecision>(gn[i]);
      _res_gamma_g_host[i] = static_cast<FPrecision>(gg[i]);
    }

  } else {
    throw std::runtime_error(
        "PiecewiseSlbwModel::setCrossSection: unknown nuclide: " + name);
  }
}

template <typename FPrecision>
typename PiecewiseSlbwModel<FPrecision>::ViewType
PiecewiseSlbwModel<FPrecision>::uploadToDevice() {
  if (_uploaded)
    return _cached_view;

  _d_res_E0 = DeviceBuffer<FPrecision>::makeFromHost(_res_E0_host, _n_res);
  _d_res_gamma_n =
      DeviceBuffer<FPrecision>::makeFromHost(_res_gamma_n_host, _n_res);
  _d_res_gamma_g =
      DeviceBuffer<FPrecision>::makeFromHost(_res_gamma_g_host, _n_res);

  _cached_view =
      ViewType(_A, _kT, _sigma_pot, _d_res_E0.get(), _d_res_gamma_n.get(),
               _d_res_gamma_g.get(), _n_res, _fissile);
  _uploaded = true;
  return _cached_view;
}

// need explicit definition otherwise compiler goes wild
template struct CrossSectionGridPoint<float>;
template struct CrossSectionGridPoint<double>;

template struct AoSLinearView<float>;
template struct AoSLinearView<double>;
template struct SoALinearView<float>;
template struct SoALinearView<double>;
template struct LogarithmicHashAoSView<float>;
template struct LogarithmicHashAoSView<double>;

template class PiecewiseSlbwModelView<float>;
template class PiecewiseSlbwModelView<double>;
template class PiecewiseSlbwModel<float>;
template class PiecewiseSlbwModel<double>;

template class CrossSection<CrossSectionGridPoint<float>, float>;
template class CrossSection<CrossSectionGridPoint<double>, double>;
template class CrossSection<CrossSectionArray<float>, float>;
template class CrossSection<CrossSectionArray<double>, double>;

template class AoSLinear<float>;
template class AoSLinear<double>;
template class SoALinear<float>;
template class SoALinear<double>;
template class LogarithmicHashAoS<float>;
template class LogarithmicHashAoS<double>;

} // namespace neuxs
