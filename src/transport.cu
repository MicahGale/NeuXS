#include "transport.cuh"

namespace neuxs {

template <typename FP>
Particle<FP> *get_mono_energetic_particles(unsigned int number_of_particles,
                                           unsigned int cell_id) {
  if (number_of_particles == 0) {
    return nullptr;
  }

  Particle<FP> *particles = new Particle<FP>[number_of_particles];

  for (unsigned int i = 0; i < number_of_particles; ++i) {
    particles[i] = Particle<FP>(static_cast<FP>(FISSION_ENERGY), cell_id);
  }

  return particles;
}

// don't make the compiler crazy
template struct Particle<float>;
template struct Particle<double>;

template neuxs::Particle<float> *
neuxs::get_mono_energetic_particles<float>(unsigned int, unsigned int);

template neuxs::Particle<double> *
neuxs::get_mono_energetic_particles<double>(unsigned int, unsigned int);

} // namespace neuxs