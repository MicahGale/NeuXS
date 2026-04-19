#ifndef NEUXS_TRANSPORT_CUH
#define NEUXS_TRANSPORT_CUH

#endif // NEUXS_TRANSPORT_CUH
#include "geometry.cuh";

namespace neuxs {
enum class EventType { COLLIDE, ESCAPE, DIE };

template <typename F> struct Particle {
  Particle(F energy, Cell *cell) : _energy(energy), _cell(cell) {}

  F _energy;
  Cell *_cell;
};
} // namespace neuxs
