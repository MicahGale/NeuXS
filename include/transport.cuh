#ifndef NEUXS_TRANSPORT_CUH
#define NEUXS_TRANSPORT_CUH

#endif // NEUXS_TRANSPORT_CUH
#include "geometry.cuh";
#include <thrust/device_vector.h>

namespace neuxs {
enum class EventType { COLLIDE, ESCAPE, DIE };

template <typename F> struct Particle {
  Particle(F energy, Cell *cell) : _energy(energy), _cell(cell) {}

  F _energy;
  Cell *_cell;
};

template <typename XSType, typename FPrecision>
__device__ void transport_history(device_vector<Particle<FPrecision>>,
                                  device_vector<Cell<XSType, FPrecision>>);
} // namespace neuxs
